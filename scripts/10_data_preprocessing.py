# 10_data_preprocessing.py
# ------------------------
# PURPOSE
# -------
# Preprocessing stage for the YouTube ingestion pipeline.
#
# Responsibilities:
#   1) Load the video registry (video metadata: region, handle, title)
#   2) Discover and load all raw comment batch CSVs (comments_batch_*.csv)
#   3) Concatenate all batches into a single dataframe
#   4) Clean and normalize text and timestamps
#   5) Deduplicate comments (exact match strategy)
#   6) Join comments with video metadata
#   7) Export the final master dataset
#
# Output:
#   data/processed/master_dataset.csv

import pandas as pd
import glob
import os
import sys

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

INPUT_DIR = "data/youtube_batch"
OUTPUT_DIR = "data/processed"
REGISTRY_FILE = "video_registry.csv"


def clean_text(text):
    """
    Basic text normalization:
        - Handle NaN safely
        - Remove newlines and carriage returns
        - Trim leading/trailing whitespace
    """
    if pd.isna(text):
        return ""
    return str(text).replace('\n', ' ').replace('\r', '').strip()


def main():
    # ------------------------------------------------------------
    # 1) OUTPUT DIRECTORY SETUP
    # ------------------------------------------------------------
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # ------------------------------------------------------------
    # 2) LOCATE VIDEO REGISTRY
    # ------------------------------------------------------------
    registry_path = REGISTRY_FILE

    if not os.path.exists(registry_path):
        registry_path = os.path.join(INPUT_DIR, REGISTRY_FILE)
        if not os.path.exists(registry_path):
            registry_path = os.path.join("data/raw", REGISTRY_FILE)

    if not os.path.exists(registry_path):
        print(f"ERROR: Could not find {REGISTRY_FILE}.")
        sys.exit(1)

    print(f"Video registry found at: {registry_path}")

    # ------------------------------------------------------------
    # 3) LOAD VIDEO REGISTRY
    # ------------------------------------------------------------
    df_videos = pd.read_csv(registry_path)

    # Ensure consistent ID type for merging
    df_videos['id'] = df_videos['id'].astype(str)

    # Remove duplicate video IDs
    df_videos = df_videos.drop_duplicates(subset=['id'])

    print(f"Unique videos indexed: {len(df_videos)}")

    # ------------------------------------------------------------
    # 4) DISCOVER COMMENT BATCH FILES
    # ------------------------------------------------------------
    comment_files = []

    possible_paths = [
        "*.csv",
        "data/raw/*.csv",
        "data/youtube_batch/*.csv",
        "data/youtube_raw/*.csv"
    ]

    for path in possible_paths:
        found = glob.glob(path)
        comment_files.extend([f for f in found if "comments_batch" in f])

    comment_files = list(set(comment_files))

    if not comment_files:
        print("ERROR: No comment batch files found.")
        sys.exit(1)

    print(f"Comment batch files discovered: {len(comment_files)}")

    # ------------------------------------------------------------
    # 5) LOAD AND CONCATENATE COMMENT FILES
    # ------------------------------------------------------------
    df_list = []
    for f in comment_files:
        try:
            df_temp = pd.read_csv(f)
            df_list.append(df_temp)
        except Exception as e:
            print(f"Warning: Failed reading {f}: {e}")

    if not df_list:
        print("ERROR: No comment data loaded.")
        sys.exit(1)

    df_comments = pd.concat(df_list, ignore_index=True)
    initial_count = len(df_comments)

    print(f"Total raw comments: {initial_count}")

    # ------------------------------------------------------------
    # 6) DEDUPLICATION AND CLEANING
    # ------------------------------------------------------------
    df_comments = df_comments.drop_duplicates(
        subset=['video_id', 'author', 'text']
    )

    print(f"Duplicates removed: {initial_count - len(df_comments)}")

    df_comments['text'] = df_comments['text'].apply(clean_text)

    df_comments['date'] = pd.to_datetime(
        df_comments['date'],
        errors='coerce'
    )

    # ------------------------------------------------------------
    # 7) MERGE WITH VIDEO METADATA
    # ------------------------------------------------------------
    df_master = pd.merge(
        df_comments,
        df_videos[['id', 'region', 'handle', 'title']],
        left_on='video_id',
        right_on='id',
        how='left'
    )

    # ------------------------------------------------------------
    # 8) SAVE OUTPUT
    # ------------------------------------------------------------
    output_file = os.path.join(OUTPUT_DIR, "master_dataset.csv")
    df_master.to_csv(output_file, index=False)

    # ------------------------------------------------------------
    # 9) SUMMARY REPORT
    # ------------------------------------------------------------
    print("Preprocessing completed.")
    print(f"Output file: {output_file}")
    print(f"Final row count: {len(df_master)}")
    print(f"Columns: {list(df_master.columns)}")

    if 'region' in df_master.columns:
        print("Top regions by comment volume:")
        print(df_master['region'].value_counts().head())


if __name__ == "__main__":
    main()
