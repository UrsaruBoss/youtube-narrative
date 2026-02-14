# 20_translate_data.py
# --------------------
# PURPOSE
# -------
# Translation stage for the YouTube pipeline.
#
# Responsibilities:
#   - Load the preprocessed master dataset (comments + video metadata).
#   - Detect/guess the source language per row using:
#       1) Region-to-language mapping (if available),
#       2) Lightweight regex heuristics (Cyrillic, diacritics, common words).
#   - Translate non-English text into English using an NLLB seq2seq model.
#   - Skip translation for rows guessed as English (copy original text).
#   - Cache translations to speed up duplicates and resume runs.
#   - Periodically checkpoint output CSV so the process can resume after interruption.
#
# Output:
#   - data/processed/master_dataset_translated.csv
#
# PERFORMANCE / SAFETY NOTES
# --------------------------
# - NLLB seq2seq is GPU-heavy. Batch size must match VRAM and input length.
# - This implementation buckets rows by detected language to reduce tokenizer state changes.
# - CommentThreads text can be short/noisy; truncation is applied via MAX_SRC_LEN.
# - This script resumes by loading OUTPUT_FILE and filling text_en/src_lang if possible.

import os
import re
import time
import math
import hashlib
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

# Input dataset produced by preprocessing stage
INPUT_FILE = "data/processed/master_dataset.csv"

# Output dataset with English translations in `text_en`
OUTPUT_FILE = "data/processed/master_dataset_translated.csv"

# Column holding the raw text we want to translate
TEXT_COL = "text"             # Change if your column is named differently

# Optional metadata column used to improve language guessing
REGION_COL = "region"         # Set to None if you do not have region column

# NLLB model to use; this variant is smaller than the full 3.3B model
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Local on-disk model cache folder (avoids re-downloading every run)
LOCAL_DIR = "./model_nllb_600m_local"

# Translation parameters
BATCH_SIZE = 256              # Must be tuned to your GPU VRAM; seq2seq can OOM easily
MAX_SRC_LEN = 128             # Max input tokens (truncate long comments)
MAX_NEW_TOKENS = 64           # Max output tokens generated
USE_FP16 = True               # FP16 improves throughput on supported GPUs

# Checkpointing and memory cache
SAVE_EVERY_ROWS = 2000        # Write OUTPUT_FILE periodically for resume safety
PRINT_EVERY_BATCHES = 50      # Occasional CUDA cache cleanup cadence
CACHE_MAX = 250_000           # Max in-memory cache entries (translation de-dup)

# If guessed English, skip model translation and copy `text` into `text_en`
SKIP_IF_ENGLISH = True

# ------------------------------------------------------------
# REGION -> NLLB language code mapping
# ------------------------------------------------------------
# Used as a primary hint when region buckets are known and mostly single-language.
# For mixed regions, store None and rely on heuristics.
REGION_TO_LANG = {
    "ROMANIA": "ron_Latn",
    "MOLDOVA": "ron_Latn",
    "POLAND": "pol_Latn",
    "HUNGARY": "hun_Latn",
    "BULGARIA": "bul_Cyrl",
    "UKRAINE": "ukr_Cyrl",
    "CZECH_SLOVAKIA": None,          # guess ces/slk
    "BALTICS_FINLAND": None,         # guess fin/est/lav/lit
    "SCANDINAVIA_ARCTIC": None,      # guess swe/nob
    "OSINT_TACTICAL": "eng_Latn",
    "GLOBAL_NATO_NEWS": "eng_Latn",
}

# ------------------------------------------------------------
# FAST HEURISTICS (regex)
# ------------------------------------------------------------
# These are intentionally lightweight and approximate. They aim to reduce translation errors
# by selecting a plausible NLLB source language code for the bucket.
RE_CYR = re.compile(r"[\u0400-\u04FF]")                  # Cyrillic block
RE_PL = re.compile(r"[ąćęłńóśźż]", re.I)                # Polish diacritics
RE_RO = re.compile(r"[ăâîșț]", re.I)                    # Romanian diacritics
RE_HU = re.compile(r"[őű]", re.I)                       # Hungarian diacritics
RE_CZ = re.compile(r"[ěščřžýáíéúůďťň]", re.I)           # Czech diacritics
RE_SK = re.compile(r"[ôľŕäčďľňšťžýáíéú]", re.I)          # Slovak-ish set (rough)
RE_FI = re.compile(r"[äö]", re.I)                       # Finnish/Swedish/Norwegian overlap

# Baltics: rough word-based signals (not exhaustive)
RE_ET = re.compile(r"\b(ja|ei|see|ole|ning)\b", re.I)    # Estonian frequent tokens
RE_LV = re.compile(r"\b(un|arī|kā|tas|šis)\b", re.I)     # Latvian frequent tokens
RE_LT = re.compile(r"\b(ir|kad|tai|šis|nes)\b", re.I)    # Lithuanian frequent tokens

# Scandinavian: rough word-based signals
RE_SV = re.compile(r"\b(och|att|det|inte|är)\b", re.I)   # Swedish frequent tokens
RE_NO = re.compile(r"\b(og|ikke|det|er|jeg)\b", re.I)    # Norwegian Bokmål frequent tokens


def sha1(s: str) -> str:
    """
    Stable hash for caching translations.
    - Used as a lightweight key for deduplication and resume.
    """
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def ensure_model():
    """
    Ensure tokenizer and model weights exist locally.
    If missing, download from Hugging Face and save to LOCAL_DIR.

    This avoids repeated downloads and supports offline runs after initial fetch.
    """
    if os.path.exists(LOCAL_DIR) and len(os.listdir(LOCAL_DIR)) > 0:
        return

    print(f"Downloading NLLB model: {MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.save_pretrained(LOCAL_DIR)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    mdl.save_pretrained(LOCAL_DIR)


def guess_lang_nllb(region: str, text: str) -> str:
    """
    Best-effort language guess returning an NLLB language code.

    Strategy:
      1) If REGION_TO_LANG provides a direct mapping, use it.
      2) If Cyrillic is detected, choose based on region if possible (bul/ukr), else fallback.
      3) If Latin diacritics indicate a specific language, return that.
      4) If region is a mixed bucket, use word hints or fallback defaults.
      5) Default to eng_Latn to avoid harmful "wrong source language" translation attempts.

    Note:
      - This is heuristic and may misclassify short/noisy text.
      - Misclassification generally hurts translation quality more than skipping translation.
    """
    t = (text or "").strip()
    r = (region or "").upper().strip()

    direct = REGION_TO_LANG.get(r, None)
    if direct:
        return direct

    # Cyrillic scripts: region-specific if known, otherwise fallback
    if RE_CYR.search(t):
        if r == "BULGARIA":
            return "bul_Cyrl"
        if r == "UKRAINE":
            return "ukr_Cyrl"
        return "ukr_Cyrl"

    # Latin scripts with distinctive diacritics
    if RE_PL.search(t):
        return "pol_Latn"
    if RE_RO.search(t):
        return "ron_Latn"
    if RE_HU.search(t):
        return "hun_Latn"
    if RE_CZ.search(t):
        return "ces_Latn"
    if RE_SK.search(t):
        return "slk_Latn"

    # Finnish/Scandinavian overlap for ä/ö
    if RE_FI.search(t):
        if r == "BALTICS_FINLAND":
            return "fin_Latn"
        if r == "SCANDINAVIA_ARCTIC":
            if RE_SV.search(t):
                return "swe_Latn"
            if RE_NO.search(t):
                return "nob_Latn"
            return "swe_Latn"
        return "fin_Latn"

    # Mixed region buckets: use basic word hints then fall back
    if r == "CZECH_SLOVAKIA":
        if RE_SK.search(t):
            return "slk_Latn"
        return "ces_Latn"

    if r == "SCANDINAVIA_ARCTIC":
        if RE_NO.search(t):
            return "nob_Latn"
        if RE_SV.search(t):
            return "swe_Latn"
        return "swe_Latn"

    if r == "BALTICS_FINLAND":
        if RE_ET.search(t):
            return "est_Latn"
        if RE_LV.search(t):
            return "lav_Latn"
        if RE_LT.search(t):
            return "lit_Latn"
        return "fin_Latn"

    # Default: treat as English to avoid low-quality wrong-source translations
    return "eng_Latn"


def translate_batch(model, tokenizer, texts, src_lang, tgt_lang="eng_Latn"):
    """
    Translate a list of texts from src_lang -> tgt_lang using NLLB.

    Implementation details:
      - tokenizer.src_lang sets the source language for NLLB.
      - forced_bos_token_id forces the target language token at generation start.
      - truncation/padding keeps batch tensors uniform and bounded.
      - num_beams=1 is chosen for speed and determinism.
    """
    tokenizer.src_lang = src_lang

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SRC_LEN,
    ).to("cuda")

    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.inference_mode():
        if USE_FP16:
            # Autocast improves throughput on GPUs with good FP16 support
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model.generate(
                    **enc,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=1,
                    do_sample=False,
                )
        else:
            out = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=1,
                do_sample=False,
            )

    return tokenizer.batch_decode(out, skip_special_tokens=True)


def main():
    # ------------------------------------------------------------
    # BASIC INPUT CHECKS
    # ------------------------------------------------------------
    if not os.path.exists(INPUT_FILE):
        print(f"Missing input: {INPUT_FILE}")
        return

    if not torch.cuda.is_available():
        print("CUDA not available. Install a CUDA-enabled torch build or check drivers.")
        return

    # Enable faster matmul on modern NVIDIA GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("GPU:", torch.cuda.get_device_name(0))

    # Ensure local model exists, then load from disk
    ensure_model()

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_DIR, local_files_only=True).to("cuda").eval()

    # Load dataset and normalize text column
    df = pd.read_csv(INPUT_FILE)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    # Region column is optional
    if REGION_COL is None or REGION_COL not in df.columns:
        regions = pd.Series([""] * len(df))
        has_region = False
    else:
        regions = df[REGION_COL].fillna("").astype(str)
        has_region = True

    # ------------------------------------------------------------
    # RESUME SUPPORT
    # ------------------------------------------------------------
    # If OUTPUT_FILE already exists and matches length, reuse previously translated rows.
    if os.path.exists(OUTPUT_FILE):
        print("Resume enabled: existing output found, continuing...")
        out = pd.read_csv(OUTPUT_FILE)

        if len(out) == len(df) and "text_en" in out.columns:
            df["text_en"] = out["text_en"].fillna("").astype(str)
            df["src_lang"] = out.get("src_lang", "").fillna("").astype(str)
        else:
            df["text_en"] = ""
            df["src_lang"] = ""
    else:
        df["text_en"] = ""
        df["src_lang"] = ""

    # Build cache from already translated rows (helps duplicates + resume)
    cache = {}
    filled = df[df["text_en"].str.len() > 0][[TEXT_COL, "text_en", "src_lang"]]
    for _, row in filled.head(CACHE_MAX).iterrows():
        cache[sha1(row[TEXT_COL])] = (row["text_en"], row["src_lang"])

    # Rows needing translation or direct copy
    mask_need = (df[TEXT_COL].str.len() > 3) & (df["text_en"].str.len() == 0)
    idxs = df.index[mask_need].tolist()
    print(f"Rows needing processing: {len(idxs)} / {len(df)}")

    # ------------------------------------------------------------
    # LANGUAGE BUCKETING
    # ------------------------------------------------------------
    # Determine src_lang and bucket rows by language for efficient batch translation.
    buckets = defaultdict(list)
    english_idxs = []

    for i in idxs:
        lang = df.at[i, "src_lang"]
        if not lang:
            lang = guess_lang_nllb(regions.iloc[i] if has_region else "", df.at[i, TEXT_COL])
            df.at[i, "src_lang"] = lang

        if SKIP_IF_ENGLISH and lang == "eng_Latn":
            english_idxs.append(i)
        else:
            buckets[lang].append(i)

    # Copy English rows directly (no model call)
    for i in english_idxs:
        df.at[i, "text_en"] = df.at[i, TEXT_COL]

    # ------------------------------------------------------------
    # TRANSLATION LOOP
    # ------------------------------------------------------------
    start_time = time.time()
    total_target = len(df)

    # Process largest language buckets first (better for overall progress perception)
    bucket_items = sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True)

    for lang, lang_idxs in bucket_items:
        if lang == "eng_Latn":
            continue

        print(f"Translating {lang} | {len(lang_idxs)} rows")
        num_batches = math.ceil(len(lang_idxs) / BATCH_SIZE)
        pbar = tqdm(range(0, len(lang_idxs), BATCH_SIZE), total=num_batches)

        for b, start in enumerate(pbar, start=1):
            batch_ids = lang_idxs[start:start + BATCH_SIZE]

            # Split batch into cached vs uncached texts
            uncached_texts = []
            uncached_hashes = []
            uncached_rowids = []

            for rid in batch_ids:
                txt = df.at[rid, TEXT_COL]
                h = sha1(txt)

                # If translation already cached, reuse it
                if h in cache:
                    df.at[rid, "text_en"] = cache[h][0]
                    df.at[rid, "src_lang"] = cache[h][1]
                else:
                    uncached_texts.append(txt)
                    uncached_hashes.append(h)
                    uncached_rowids.append(rid)

            # Translate only uncached texts
            if uncached_texts:
                try:
                    outs = translate_batch(model, tokenizer, uncached_texts, src_lang=lang, tgt_lang="eng_Latn")
                except RuntimeError as e:
                    # OOM recovery: retry smaller chunks
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        outs = []
                        half = max(1, len(uncached_texts) // 2)
                        for s2 in range(0, len(uncached_texts), half):
                            outs.extend(
                                translate_batch(
                                    model,
                                    tokenizer,
                                    uncached_texts[s2:s2 + half],
                                    src_lang=lang,
                                    tgt_lang="eng_Latn",
                                )
                            )
                    else:
                        raise

                # Write back outputs and populate cache
                for rid, h, en in zip(uncached_rowids, uncached_hashes, outs):
                    df.at[rid, "text_en"] = en
                    cache[h] = (en, lang)

                    # Prevent runaway cache growth
                    if len(cache) > CACHE_MAX:
                        cache.pop(next(iter(cache)))

            # Progress metrics (rows/s and ETA)
            total_done = int((df["text_en"].str.len() > 0).sum())
            elapsed = time.time() - start_time
            speed = (total_done / elapsed) if elapsed > 0 else 0.0
            eta_min = ((total_target - total_done) / speed) / 60.0 if speed > 0 else float("inf")

            pbar.set_description(f"{lang} | done {total_done}/{total_target} | {speed:.1f} rows/s | ETA {eta_min:.1f}m")

            # Periodic checkpoint to support resume
            if total_done % SAVE_EVERY_ROWS < BATCH_SIZE:
                df.to_csv(OUTPUT_FILE, index=False)

            # Periodic CUDA cache cleanup to reduce fragmentation over long runs
            if (b % PRINT_EVERY_BATCHES) == 0:
                torch.cuda.empty_cache()

        # End-of-language checkpoint
        torch.cuda.empty_cache()
        df.to_csv(OUTPUT_FILE, index=False)

    # Final write
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done. Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
