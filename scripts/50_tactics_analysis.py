# 50_tactics_analysis.py
# ----------------------
# PURPOSE
# -------
# Tactics detection stage (NLI-based) that runs AFTER sentiment + stance.
#
# Approach:
#   - Zero-shot / NLI classification with XLM-RoBERTa XNLI.
#   - Each candidate tactic is expressed as an NLI hypothesis:
#       premise   = comment text
#       hypothesis = "This text uses the following tactic: <tactic_description>."
#   - We compute entailment probability for each candidate tactic and choose the best one,
#     but we gate uncertain predictions to "neutral" using:
#       - minimum best score threshold (TACTIC_MIN_SCORE)
#       - minimum margin threshold (TACTIC_MIN_MARGIN)
#
# Responsibilities:
#   - Load dataset containing at least: text/text_en, stance_label/score, sentiment_label.
#   - Load tactic candidate labels + descriptions from JSON config.
#   - Run GPU-optimized batched inference with checkpoint/resume support.
#   - Write tactic_label, tactic_score, and (optionally) recompute risk_assessment.
#
# Input:
#   data/processed/master_dataset_with_sentiment_stance.csv
#
# Output:
#   data/processed/master_dataset_with_sentiment_stance_tactics.csv
#
# Notes:
#   - Keep LOCAL_DIR separate from stance if you want isolated model caches.
#     In practice you can reuse the same XNLI weights; separation avoids accidental overwrites.
#   - Prefer running on text_en (translated) to make hypotheses consistent in English.

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
INPUT_FILE = "data/processed/master_dataset_with_sentiment_stance.csv"
OUTPUT_FILE = "data/processed/master_dataset_with_sentiment_stance_tactics.csv"

TACTICS_CONFIG_FILE = "config/config_candidates_tactics.json"

# Resume / checkpoints
CHECKPOINT_EVERY_BATCHES = 50
CHECKPOINT_FILE = "data/processed/_tactics_checkpoint.csv"

# -------------------
# MODEL
# -------------------
MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
LOCAL_DIR = "./model_xlm_local"

# Perf knobs
BATCH_SIZE_START = 192
MAX_LEN = 160
USE_FP16 = True

# Gating tactics (typically stricter than stance)
TACTIC_MIN_SCORE = 0.44
TACTIC_MIN_MARGIN = 0.08

# Diagnostics
PRINT_EVERY = 25
EMPTY_CACHE_EVERY = 50

HYP_TEMPLATE = "This text uses the following tactic: {}."

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
    Load tactic candidate labels and optional descriptions.

    Expected JSON:
      {
        "candidate_labels": ["propaganda", "conspiracy", ...],
        "descriptions": { "propaganda": "coordinated persuasive framing", ... }
      }

    Ensures "neutral" is included.
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
    Convert candidate tactic labels into NLI hypotheses.
    Descriptions are preferred because they are more semantically explicit than short keys.
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
    Identify which output index corresponds to entailment.
    Fallback to 2 (common for XNLI: contradiction, neutral, entailment).
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
    Prefer translated text if present; otherwise fallback to original.
    """
    if TEXT_COL_PRIMARY in df.columns and df[TEXT_COL_PRIMARY].notna().any():
        return TEXT_COL_PRIMARY
    return TEXT_COL_FALLBACK


def quick_token_len(tokenizer, text: str) -> int:
    """
    Approximate token length for bucketing.
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
    NLI entailment scoring over tactic hypotheses.

    For each text, compute entailment scores for each tactic hypothesis and select best.
    Gate uncertain results to "neutral" if:
      - best_score < min_score, or
      - best_score - second_best < min_margin
    """
    B = len(batch_texts)
    L = len(hypotheses)

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

        top2 = torch.topk(entail, k=2, dim=1).values
        best_score = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        best_idx = torch.argmax(entail, dim=1)

        unsure = (best_score < min_score) | (margin < min_margin)
        best_idx = torch.where(
            unsure,
            torch.tensor(neutral_idx, device=best_idx.device),
            best_idx
        )

    return best_idx.detach().cpu().tolist(), best_score.detach().cpu().tolist()


def calculate_risk(row):
    """
    RISK (stance + tactics).

    Simple heuristic:
      - dehumanization tactic (confident) => highest severity
      - propaganda/conspiracy (confident) => manipulative signal, severity increases if hostile stance too
      - hostile stance (confident) => hostile narrative/framing
      - otherwise LOW_RISK
    """
    sentiment = row.get("sentiment_label", "neutral")
    stance = row.get("stance_label", "neutral")
    sscore = float(row.get("stance_score", 0.0) or 0.0)

    tactic = row.get("tactic_label", "neutral")
    tscore = float(row.get("tactic_score", 0.0) or 0.0)

    # Hard severity: dehumanization tactic
    if tactic == "dehumanization" and tscore >= TACTIC_MIN_SCORE:
        return "DEHUMANIZATION"

    # Manipulative tactics
    if tactic in ["conspiracy", "propaganda"] and tscore >= TACTIC_MIN_SCORE:
        if stance in ["anti_ukraine", "anti_russia", "anti_west"] and sscore >= 0.36:
            return "HOSTILE_MANIPULATION"
        return "MANIPULATIVE_NARRATIVE"

    # Stance-only hostility
    if stance in ["anti_ukraine", "anti_russia", "anti_west"] and sscore >= 0.36:
        return "HOSTILE_NARRATIVE" if sentiment == "negative" else "HOSTILE_FRAMING"

    return "LOW_RISK"


# -------------------
# MAIN
# -------------------
def main():
    print("50 (TACTICS) | optimized | uses text_en if present")

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

    labels, descriptions = load_config(TACTICS_CONFIG_FILE)
    hypotheses = build_hypotheses(labels, descriptions)
    neutral_idx = labels.index("neutral")

    print(f"Tactic labels ({len(labels)}): {labels}")
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

    mask_valid = df[text_col].str.len() > 3
    valid_indices = df.index[mask_valid].to_list()
    valid_texts = df.loc[mask_valid, text_col].to_list()
    n = len(valid_texts)
    print(f"Texts to analyze: {n}")

    # ------------------------------------------------------------
    # Resume if checkpoint exists
    # ------------------------------------------------------------
    tactic_label_out = ["neutral"] * len(df)
    tactic_score_out = [0.0] * len(df)
    done_mask = set()

    if os.path.exists(CHECKPOINT_FILE):
        try:
            ck = pd.read_csv(CHECKPOINT_FILE)
            if len(ck) == len(df) and "tactic_label" in ck.columns and "tactic_score" in ck.columns:
                tactic_label_out = ck["tactic_label"].fillna("neutral").astype(str).tolist()
                tactic_score_out = ck["tactic_score"].fillna(0.0).astype(float).tolist()
                done_mask = set(
                    i for i in valid_indices
                    if tactic_score_out[i] != 0.0 or tactic_label_out[i] != "neutral"
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
    tactic_counter = Counter()
    tactic_score_sum = 0.0
    seen_samples = 0

    # Warmup
    warm = ordered_texts[:min(8, n2)]
    if warm:
        _ = run_batch(
            model, tokenizer, warm, hypotheses, entail_idx,
            MAX_LEN, USE_FP16, TACTIC_MIN_SCORE, TACTIC_MIN_MARGIN, neutral_idx
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

        while True:
            try:
                idxs, scores = run_batch(
                    model, tokenizer, batch_texts, hypotheses, entail_idx,
                    MAX_LEN, USE_FP16, TACTIC_MIN_SCORE, TACTIC_MIN_MARGIN, neutral_idx
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

        for df_i, li, sc in zip(batch_df_idx, idxs, scores):
            tactic_label_out[df_i] = labels[int(li)]
            tactic_score_out[df_i] = float(sc)

        batch_lbls = [labels[int(li)] for li in idxs]
        tactic_counter.update(batch_lbls)
        tactic_score_sum += sum(scores)
        seen_samples += len(batch_texts)

        processed = start + len(batch_texts)
        dt = time.time() - t0
        speed = processed / dt if dt > 0 else 0.0
        eta_min = ((n2 - processed) / speed) / 60.0 if speed > 0 else float("inf")
        pbar.set_description(f"GPU NLI | bs={batch_size} | {speed:.1f} txt/s | ETA {eta_min:.1f} min")

        if bi % PRINT_EVERY == 0:
            total = sum(tactic_counter.values()) or 1
            avg_score = (tactic_score_sum / seen_samples) if seen_samples else 0.0
            print("\n=== INTERIM DISTRIBUTION (TACTICS) ===")
            for lbl, cnt in tactic_counter.most_common(10):
                print(f"  {lbl:20s} {cnt/total:6.2%}")
            print(f"Avg tactic score: {avg_score:.3f}")
            print("======================================\n")

        if bi % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()

        if bi % CHECKPOINT_EVERY_BATCHES == 0:
            ck = df.copy()
            ck["tactic_label"] = tactic_label_out
            ck["tactic_score"] = tactic_score_out
            ck.to_csv(CHECKPOINT_FILE, index=False)

    # ------------------------------------------------------------
    # Save final
    # ------------------------------------------------------------
    df["tactic_label"] = tactic_label_out
    df["tactic_score"] = tactic_score_out

    # Optional: recompute risk with tactics included
    df["risk_assessment"] = df.apply(calculate_risk, axis=1)

    df.to_csv(OUTPUT_FILE, index=False)

    # Cleanup checkpoint
    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
    except Exception:
        pass

    print(f"Done: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
