# 30_sentiment_analysis.py
# ---------------------------------
# PURPOSE
# -------
# Sentiment analysis stage for the pipeline
#
# Responsibilities:
#   - Load the translated (or original) master dataset.
#   - Run multilingual sentiment classification in efficient GPU batches.
#   - Support resume via checkpoint file (so long runs can be continued).
#   - Save final dataset with sentiment_label and sentiment_score.
#
# Input:
#   data/processed/master_dataset_translated.csv  (or a variant you set below)
#
# Output:
#   data/processed/master_dataset_with_sentiment.csv
#
# Model:
#   cardiffnlp/twitter-xlm-roberta-base-sentiment
#
# Notes:
#   - This model was trained on Twitter-like short text; it is still useful for
#     YouTube comments, but expect some domain shift.
#   - If text_en is present, we prefer that (consistent language space for downstream logic).
#   - We run the model directly (tokenizer + model) instead of HF pipeline for better
#     control and predictable batching/resume behavior.

import os
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
INPUT_FILE = "data/processed/master_dataset_translated.csv"
OUTPUT_FILE = "data/processed/master_dataset_with_sentiment.csv"

# Resume / checkpoints
CHECKPOINT_EVERY_BATCHES = 50
CHECKPOINT_FILE = "data/processed/_sentiment_checkpoint.csv"

# -------------------
# MODEL
# -------------------
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
LOCAL_DIR = "./model_sentiment_local"

# Perf knobs
BATCH_SIZE_START = 256      # will auto-reduce on OOM (128/96/64/32)
MAX_LEN = 128               # sentiment usually works well with 128–256; 512 is slower
USE_FP16 = True

# Diagnostics
PRINT_EVERY = 25
EMPTY_CACHE_EVERY = 50

# Prefer translated column if available
TEXT_COL_PRIMARY = "text_en"
TEXT_COL_FALLBACK = "text"

# Optional: length bucketing to reduce padding waste
BUCKET_SIZES = [32, 64, 96, 128, 160, 192, 256, 320, 384, 512]


# -------------------
# HELPERS
# -------------------
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


def pick_text_col(df: pd.DataFrame) -> str:
    """
    Prefer translated text if present (text_en), else fallback to original text.
    """
    if TEXT_COL_PRIMARY in df.columns and df[TEXT_COL_PRIMARY].notna().any():
        return TEXT_COL_PRIMARY
    return TEXT_COL_FALLBACK


def quick_token_len(tokenizer, text: str) -> int:
    """
    Approximate token length for bucketing.
    Uses tokenizer.encode (single sequence) without special tokens.
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


def map_sentiment_id2label(model) -> dict:
    """
    Build a robust id->label mapping for the model.

    For cardiffnlp/twitter-xlm-roberta-base-sentiment:
      LABEL_0 -> negative
      LABEL_1 -> neutral
      LABEL_2 -> positive

    If model has custom id2label, we still normalize.
    """
    # Default for this model family
    default = {0: "negative", 1: "neutral", 2: "positive"}

    id2label = {}
    if getattr(model.config, "id2label", None):
        for k, v in model.config.id2label.items():
            try:
                i = int(k)
            except Exception:
                continue
            id2label[i] = str(v).lower().strip()

    # Normalize common label names
    normalized = {}
    for i in range(model.config.num_labels):
        lab = id2label.get(i, f"label_{i}").lower()
        if lab in ("label_0", "lbl_0", "negative"):
            normalized[i] = "negative"
        elif lab in ("label_1", "lbl_1", "neutral"):
            normalized[i] = "neutral"
        elif lab in ("label_2", "lbl_2", "positive"):
            normalized[i] = "positive"
        else:
            # fallback to model default mapping if weird
            normalized[i] = default.get(i, lab)

    # If still looks wrong, use known default for this exact model (3-class)
    if model.config.num_labels == 3 and set(normalized.values()) != {"negative", "neutral", "positive"}:
        normalized = default

    return normalized


def run_batch(model, tokenizer, batch_texts, max_len, use_fp16):
    """
    Run one sentiment batch.

    Returns:
      pred_ids: list[int]   predicted class index per input text
      scores:   list[float] confidence score of the predicted class
    """
    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
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
        pred = torch.argmax(probs, dim=-1)
        conf = probs.gather(1, pred.view(-1, 1)).squeeze(1)

    return pred.detach().cpu().tolist(), conf.detach().cpu().tolist()


# -------------------
# MAIN
# -------------------
def main():
    print("30 (SENTIMENT) | optimized | uses text_en if present")

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

    if not os.path.exists(INPUT_FILE):
        print(f"Missing input: {INPUT_FILE}")
        return

    print("Load tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_DIR, local_files_only=True)
    model.eval().to("cuda")

    id2sent = map_sentiment_id2label(model)
    print("Sentiment id->label:", id2sent)

    df = pd.read_csv(INPUT_FILE)
    text_col = pick_text_col(df)
    print(f"Using text column: {text_col}")

    df[text_col] = df[text_col].fillna("").astype(str)

    # Define which rows are valid for inference
    mask_valid = df[text_col].str.len() > 0
    valid_indices = df.index[mask_valid].to_list()
    valid_texts = df.loc[mask_valid, text_col].to_list()
    n = len(valid_texts)
    print(f"Texts to analyze: {n}")

    # Resume from checkpoint if exists
    sentiment_label_out = ["neutral"] * len(df)
    sentiment_score_out = [0.0] * len(df)

    done_mask = set()
    if os.path.exists(CHECKPOINT_FILE):
        try:
            ck = pd.read_csv(CHECKPOINT_FILE)
            if len(ck) == len(df) and "sentiment_label" in ck.columns and "sentiment_score" in ck.columns:
                sentiment_label_out = ck["sentiment_label"].fillna("neutral").astype(str).tolist()
                sentiment_score_out = ck["sentiment_score"].fillna(0.0).astype(float).tolist()

                # Mark done where we have a non-zero score or label != neutral
                done_mask = set(
                    i for i in valid_indices
                    if sentiment_score_out[i] != 0.0 or sentiment_label_out[i] != "neutral"
                )
                print(f"Resume: {len(done_mask)} rows already filled from checkpoint.")
        except Exception:
            pass

    # Bucketing by token length to reduce padding overhead (optional but useful)
    print("Bucketing by length...")
    lengths = [quick_token_len(tokenizer, t[:2000]) for t in tqdm(valid_texts, total=n)]
    buckets = bucketize_lengths(lengths)

    # Process smaller items first to reduce padding waste
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

    # Diagnostics tracking
    sent_counter = Counter()
    score_sum = 0.0
    seen_samples = 0

    # Warmup pass (helps reduce first-batch overhead)
    warm = ordered_texts[:min(8, n2)]
    if warm:
        _ = run_batch(model, tokenizer, warm, MAX_LEN, USE_FP16)
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
                pred_ids, confs = run_batch(model, tokenizer, batch_texts, MAX_LEN, USE_FP16)
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

        # Write outputs
        for df_i, pid, sc in zip(batch_df_idx, pred_ids, confs):
            lbl = id2sent.get(int(pid), "neutral")
            sentiment_label_out[df_i] = lbl
            sentiment_score_out[df_i] = float(sc)

        # Update diagnostics
        batch_lbls = [id2sent.get(int(pid), "neutral") for pid in pred_ids]
        sent_counter.update(batch_lbls)
        score_sum += sum(confs)
        seen_samples += len(batch_texts)

        processed = start + len(batch_texts)
        dt = time.time() - t0
        speed = processed / dt if dt > 0 else 0.0
        eta_min = ((n2 - processed) / speed) / 60.0 if speed > 0 else float("inf")
        pbar.set_description(f"GPU sentiment | bs={batch_size} | {speed:.1f} txt/s | ETA {eta_min:.1f} min")

        # Periodic diagnostics
        if bi % PRINT_EVERY == 0:
            total = sum(sent_counter.values()) or 1
            avg_score = (score_sum / seen_samples) if seen_samples else 0.0
            print("\n=== INTERIM DISTRIBUTION (SENTIMENT) ===")
            for lbl, cnt in sent_counter.most_common():
                print(f"  {lbl:10s} {cnt/total:6.2%}")
            print(f"Avg confidence score: {avg_score:.3f}")
            print("=======================================\n")

        if bi % EMPTY_CACHE_EVERY == 0:
            torch.cuda.empty_cache()

        if bi % CHECKPOINT_EVERY_BATCHES == 0:
            ck = df.copy()
            ck["sentiment_label"] = sentiment_label_out
            ck["sentiment_score"] = sentiment_score_out
            ck.to_csv(CHECKPOINT_FILE, index=False)

    # Save final
    df["sentiment_label"] = sentiment_label_out
    df["sentiment_score"] = sentiment_score_out
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
