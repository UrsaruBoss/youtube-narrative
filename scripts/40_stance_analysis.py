# 40_stance_analysis_optimized.py
# ------------------------------
# PURPOSE
# -------
# Stance / narrative classification stage using NLI (XNLI) with candidate labels.
#
# Approach:
#   - Treat each stance label as an NLI hypothesis:
#       premise   = comment text
#       hypothesis = "This text expresses: <label_description>."
#   - For each text, compute entailment probability for each hypothesis.
#   - Pick the label with highest entailment, but gate to "neutral" when the model
#     is uncertain (low best score or low margin vs 2nd best).
#
# Responsibilities:
#   - Load dataset containing text + sentiment (optionally translated text_en).
#   - Load stance candidate labels + descriptions from JSON config.
#   - Run GPU-optimized NLI scoring in batches with checkpoint/resume support.
#   - Write stance_label, stance_score, and risk_assessment columns.
#
# Input:
#   data/processed/master_dataset_with_sentiment_en.csv
#
# Output:
#   data/processed/master_dataset_with_stance.csv
#
# Notes:
#   - This is an inference-only script (no training).
#   - The "risk_assessment" here is stance+sentiment driven (heuristic).
#   - For throughput, we bucket texts by token length to reduce padding waste.

import os
import json
import math
import time
from collections import Counter

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------
# FILES
# -------------------
INPUT_FILE = "data/processed/master_dataset_with_sentiment_en.csv"
OUTPUT_FILE = "data/processed/master_dataset_with_stance.csv"
STANCE_CONFIG_FILE = "config/config_candidates_stance.json"

# Resume / checkpoints
CHECKPOINT_EVERY_BATCHES = 50
CHECKPOINT_FILE = "data/processed/_stance_checkpoint.csv"

# -------------------
# MODEL
# -------------------
MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
LOCAL_DIR = "./model_stance_local"

# Perf knobs
BATCH_SIZE_START = 192      # try 192; auto-reduce on OOM (96/64/32)
MAX_LEN = 96                # 96–192 typical sweet spot for short comments
USE_FP16 = True

# Gating stance
STANCE_MIN_SCORE = 0.36
STANCE_MIN_MARGIN = 0.06

# Diagnostics
PRINT_EVERY = 25
EMPTY_CACHE_EVERY = 50

HYP_TEMPLATE = "This text expresses: {}."

# Prefer translated column if available
TEXT_COL_PRIMARY = "text_en"
TEXT_COL_FALLBACK = "text"

# Length bucketing
BUCKET_SIZES = [32, 64, 96, 128, 160, 192, 256]


# -------------------
# HELPERS
# -------------------
def load_config(path: str):
    """
    Load stance candidate labels and optional descriptions.

    Expected JSON shape:
      {
        "candidate_labels": ["pro-nato", "anti_west", ...],
        "descriptions": {
          "pro-nato": "support for NATO / the West",
          ...
        }
      }

    Ensures:
      - candidate_labels is non-empty
      - "neutral" is included
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing config: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = data.get("candidate_labels", [])
    desc = data.get("descriptions", {}) or {}

    if not labels:
        raise ValueError(f"No candidate_labels in {path}")

    if "neutral" not in labels:
        labels.append("neutral")

    return labels, desc


def build_hypotheses(labels, descriptions):
    """
    Convert labels into natural-language NLI hypotheses.

    Using descriptions improves stability:
      - label key can be short (anti_west)
      - description makes hypothesis semantically rich (anti-West sentiment)
    """
    return [HYP_TEMPLATE.format(descriptions.get(lbl, lbl)) for lbl in labels]


def ensure_model_exists():
    """
    Ensure tokenizer + model are present locally.
    Downloads to LOCAL_DIR if missing.
    """
    if os.path.exists(LOCAL_DIR) and len(os.listdir(LOCAL_DIR)) > 0:
        return True

    print(f"Downloading model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.save_pretrained(LOCAL_DIR)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.save_pretrained(LOCAL_DIR)
    return True


def get_entailment_index(model):
    """
    Identify which logit index corresponds to the "entailment" class.

    XNLI models typically have labels like:
      - contradiction
      - neutral
      - entailment

    We try config.label2id/id2label to find the right one; fallback to 2.
    """
    label2id = {k.lower(): v for k, v in (model.config.label2id or {}).items()}
    if "entailment" in label2id:
        return int(label2id["entailment"])

    id2label = {int(k): str(v).lower() for k, v in (model.config.id2label or {}).items()}
    for i, lab in id2label.items():
        if "entail" in lab:
            return int(i)

    return 2


def pick_text_col(df: pd.DataFrame) -> str:
    """
    Prefer translated text if present, otherwise use original.
    """
    if TEXT_COL_PRIMARY in df.columns and df[TEXT_COL_PRIMARY].notna().any():
        return TEXT_COL_PRIMARY
    return TEXT_COL_FALLBACK


def quick_token_len(tokenizer, text: str) -> int:
    """
    Approximate token length for bucketing (single-sequence encoding).
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def bucketize_lengths(lengths):
    """
    Assign each length to the smallest bucket >= length.
    """
    out = []
    for L in lengths:
        b = None
        for s in BUCKET_SIZES:
            if L <= s:
                b = s
                break
        out.append(b or BUCKET_SIZES[-1])
    return out


def run_batch(
    model,
    tokenizer,
    batch_texts,
    hypotheses,
    entail_idx,
    max_len,
    use_fp16,
    min_score,
    min_margin,
    neutral_idx
):
    """
    Run NLI scoring for one batch.

    For each text:
      - Evaluate entailment probability against each hypothesis.
      - Select best hypothesis label, but gate to neutral if:
          best_score < min_score OR margin(best - second_best) < min_margin

    Implementation:
      - Expand texts to pairs (premise, hypothesis) sized B * L
      - Tokenize with truncation on the premise only (comments may be long/noisy)
      - Compute entailment probs, reshape back to (B, L)
    """
    B = len(batch_texts)
    L = len(hypotheses)

    # Repeat each premise L times to pair with all hypotheses
    premises = sum(([t] * L for t in batch_texts), [])
    hyps = hypotheses * B

    enc = tokenizer(
        premises,
        hyps,
        padding=True,
        truncation="only_first",
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to("cuda", non_blocking=True) for k, v in enc.items()}

    with torch.inference_mode():
        if use_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**enc).logits
        else:
            logits = model(**enc).logits

        probs = torch.softmax(logits, dim=-1)
        entail = probs[:, entail_idx].view(B, L)

        # Extract best and second-best entailment for gating
        top2 = torch.topk(entail, k=2, dim=1).values
        best_score = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        best_idx = torch.argmax(entail, dim=1)

        # Gate uncertain predictions to neutral
        unsure = (best_score < min_score) | (margin < min_margin)
        best_idx = torch.where(
            unsure,
            torch.tensor(neutral_idx, device=best_idx.device),
            best_idx
        )

    return best_idx.detach().cpu().tolist(), best_score.detach().cpu().tolist()


def calculate_risk(row):
    """
    Risk scoring (heuristic), driven by stance and sentiment.

    Design goals:
      - Avoid over-flagging generic "peace" comments.
      - Only elevate "peace_call" when context clearly references conflict/NATO topics.
      - Require stance_score gating for non-neutral labels (except hard overrides).

    Returns:
      - A coarse risk category string.
    """
    sentiment = row.get("sentiment_label", "neutral")
    stance = row.get("stance_label", "neutral")
    sscore = float(row.get("stance_score", 0.0) or 0.0)

    text = str(row.get("text_en") or row.get("text") or "").lower()

    # Hard override: highest severity irrespective of score
    if stance == "dehumanization":
        return "DEHUMANIZATION"

    # If the model is uncertain on a non-neutral label, down-rank to low risk
    if stance != "neutral" and sscore < STANCE_MIN_SCORE:
        return "LOW_RISK"

    # Hostility buckets
    if stance in ["anti_ukraine", "anti_russia", "anti_west"]:
        return "HOSTILE_NARRATIVE" if sentiment == "negative" else "HOSTILE_FRAMING"

    # Alignment buckets (positive sentiment + alignment stance)
    if sentiment == "positive" and stance == "pro-russia":
        return "CRITICAL_PROPAGANDA"
    if sentiment == "positive" and stance == "pro-ukraine":
        return "PRO_UKRAINE_SUPPORT"
    if sentiment == "positive" and stance == "pro-nato":
        return "PRO_WEST_SUPPORT"

    # Peace narrative (strict context check)
    if stance == "peace_call":
        peace_context = [
            "ukrain", "ukraine", "russia", "russian", "putin",
            "zelens", "zelensky", "nato", "war", "invasion",
            "ceasefire", "negotiat", "truce"
        ]
        has_context = any(k in text for k in peace_context)

        # Optional stricter gating for peace_call to reduce false positives
        if has_context and sscore >= max(0.45, STANCE_MIN_SCORE):
            return "PEACE_NARRATIVE"
        return "LOW_RISK"

    return "LOW_RISK"


# -------------------
# MAIN
# -------------------
def main():
    print("40 (STANCE) | optimized | uses text_en if present")

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("GPU:", torch.cuda.get_device_name(0))
    ensure_model_exists()

    labels, descriptions = load_config(STANCE_CONFIG_FILE)
    hypotheses = build_hypotheses(labels, descriptions)
    neutral_idx = labels.index("neutral")

    print(f"Stance labels ({len(labels)}): {labels}")
    print("Example hypothesis:", hypotheses[0])

    if not os.path.exists(INPUT_FILE):
        print(f"Missing input: {INPUT_FILE}")
        return

    print("Load tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR, local_files_only=True)
    model.eval().to("cuda")
    entail_idx = get_entailment_index(model)

    df = pd.read_csv(INPUT_FILE)
    text_col = pick_text_col(df)
    print(f"Using text column: {text_col}")
    df[text_col] = df[text_col].fillna("").astype(str)

    # Only analyze non-trivial text
    mask_valid = df[text_col].str.len() > 3
    valid_indices = df.index[mask_valid].to_list()
    valid_texts = df.loc[mask_valid, text_col].to_list()
    n = len(valid_texts)
    print(f"Texts to analyze: {n}")

    # ------------------------------------------------------------
    # Resume if checkpoint exists
    # ------------------------------------------------------------
    stance_label_out = ["neutral"] * len(df)
    stance_score_out = [0.0] * len(df)
    done_mask = set()

    if os.path.exists(CHECKPOINT_FILE):
        try:
            ck = pd.read_csv(CHECKPOINT_FILE)
            if len(ck) == len(df) and "stance_label" in ck.columns and "stance_score" in ck.columns:
                stance_label_out = ck["stance_label"].fillna("neutral").astype(str).tolist()
                stance_score_out = ck["stance_score"].fillna(0.0).astype(float).tolist()
                done_mask = set(
                    i for i in valid_indices
                    if stance_score_out[i] != 0.0 or stance_label_out[i] != "neutral"
                )
                print(f"Resume: {len(done_mask)} rows already filled from checkpoint.")
        except Exception:
            pass

    # ------------------------------------------------------------
    # Bucket texts by length to reduce padding
    # ------------------------------------------------------------
    print("Bucketing by length...")
    lengths = [quick_token_len(tokenizer, t[:2000]) for t in tqdm(valid_texts, total=n)]
    buckets = bucketize_lengths(lengths)

    # Process shorter texts first
    order = sorted(range(n), key=lambda i: (buckets[i], lengths[i]))
    ordered_indices = [valid_indices[i] for i in order]
    ordered_texts = [valid_texts[i] for i in order]

    # Remove already processed rows if resuming
    if done_mask:
        filtered = [(idx, txt) for idx, txt in zip(ordered_indices, ordered_texts) if idx not in done_mask]
        if filtered:
            ordered_indices, ordered_texts = zip(*filtered)
            ordered_indices, ordered_texts = list(ordered_indices), list(ordered_texts)
        else:
            ordered_indices, ordered_texts = [], []

    n2 = len(ordered_texts)
    print(f"To process after resume: {n2}")

    # Diagnostics
    stance_counter = Counter()
    stance_score_sum = 0.0
    seen_samples = 0

    # Warmup
    warm = ordered_texts[:min(8, n2)]
    if warm:
        _ = run_batch(
            model, tokenizer, warm, hypotheses, entail_idx,
            MAX_LEN, USE_FP16, STANCE_MIN_SCORE, STANCE_MIN_MARGIN, neutral_idx
        )
        torch.cuda.synchronize()

    batch_size = BATCH_SIZE_START
    t0 = time.time()

    num_batches = math.ceil(n2 / batch_size) if batch_size > 0 else 0
    pbar = tqdm(range(0, n2, batch_size), total=num_batches)

    for bi, start in enumerate(pbar, start=1):
        end = min(start + batch_size, n2)
        batch_texts = ordered_texts[start:end]
        batch_df_idx = ordered_indices[start:end]

        # Try current batch_size; on OOM auto-reduce and retry
        while True:
            try:
                idxs, scores = run_batch(
                    model, tokenizer, batch_texts, hypotheses, entail_idx,
                    MAX_LEN, USE_FP16, STANCE_MIN_SCORE, STANCE_MIN_MARGIN, neutral_idx
                )
                break
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda" in msg:
                    torch.cuda.empty_cache()
                    if batch_size <= 32:
                        raise
                    batch_size = max(32, batch_size // 2)
                    end = min(start + batch_size, n2)
                    batch_texts = ordered_texts[start:end]
                    batch_df_idx = ordered_indices[start:end]
                    pbar.write(f"OOM/CUDA stall -> batch_size now {batch_size}")
                    continue
                else:
                    raise

        # Write batch outputs back into full-length arrays
        for df_i, li, sc in zip(batch_df_idx, idxs, scores):
            stance_label_out[df_i] = labels[int(li)]
            stance_score_out[df_i] = float(sc)

        # Update diagnostics counters
        batch_lbls = [labels[int(li)] for li in idxs]
        stance_counter.update(batch_lbls)
        stance_score_sum += sum(scores)
        seen_samples += len(batch_texts)

        # Speed / ETA (based on processed count in this ordered list)
        processed = start + len(batch_texts)
        dt = time.time() - t0
        speed = processed / dt if dt > 0 else 0.0
        eta_min = ((n2 - processed) / speed) / 60.0 if speed > 0 else float("inf")
        pbar.set_description(f"GPU NLI | bs={batch_size} | {speed:.1f} txt/s | ETA {eta_min:.1f} min")

        # Periodic diagnostics print
        if bi % PRINT_EVERY == 0:
            total = sum(stance_counter.values()) or 1
            avg_score = (stance_score_sum / seen_samples) if seen_samples else 0.0
            print("\n=== INTERIM DISTRIBUTION (STANCE) ===")
            for lbl, cnt in stance_counter.most_common(10):
                print(f"  {lbl:20s} {cnt/total:6.2%}")
            print(f"Avg stance score: {avg_score:.3f}")
            print("=====================================\n")

        if bi % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()

        # Periodic checkpoint save for resume
        if bi % CHECKPOINT_EVERY_BATCHES == 0:
            ck = df.copy()
            ck["stance_label"] = stance_label_out
            ck["stance_score"] = stance_score_out
            ck.to_csv(CHECKPOINT_FILE, index=False)

    # ------------------------------------------------------------
    # Save final output
    # ------------------------------------------------------------
    df["stance_label"] = stance_label_out
    df["stance_score"] = stance_score_out
    df["risk_assessment"] = df.apply(calculate_risk, axis=1)

    df.to_csv(OUTPUT_FILE, index=False)

    # Cleanup checkpoint after success
    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
    except Exception:
        pass

    print(f"Done: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
