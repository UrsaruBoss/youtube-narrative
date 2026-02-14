# 00_gather_data.py
# -----------------
# PURPOSE
# -------
# Massive YouTube data ingestion script.
#
# Responsibilities:
#   - Fetch latest N videos from configured YouTube channel handles
#   - Extract top-level comments for each new video (NOW WITH PAGINATION)
#   - Avoid re-processing already scanned videos (registry-based deduplication)
#   - Respect daily YouTube Data API quota limits
#   - Persist raw data for downstream ETL stages
#
# This script represents Stage 00 (Ingestion) of the pipeline.

import os
import sys
import json
import pandas as pd
import logging
from datetime import datetime
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ------------------------------------------------------------
# LOGGING CONFIGURATION
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
CONFIG_FILE = "config/config_targets.json"
OUTPUT_DIR = "data/raw"


class YouTubeHeavyScraper:
    """
    Heavy YouTube scraper with quota awareness and registry-based deduplication.

    Main API endpoints used:
        - channels().list()            -> Resolve uploads playlist
        - playlistItems().list()       -> Fetch channel videos (paginated)
        - commentThreads().list()      -> Fetch top-level comments (paginated)

    Design Goals:
        - Minimize API quota waste
        - Avoid re-scraping known videos
        - Stop safely before hitting daily quota hard limit (~10k units)
    """

    def __init__(self, api_key):
        self.youtube = build("youtube", "v3", developerKey=api_key)

        self.quota_used = 0
        self.max_quota = 9800  # safety margin under 10k daily limit

        self.registry_path = f"{OUTPUT_DIR}/video_registry.csv"
        self.scraped_ids = self._load_registry()

    def _load_registry(self):
        """Load previously scraped video IDs from registry."""
        if os.path.exists(self.registry_path):
            try:
                df = pd.read_csv(self.registry_path)
                return set(df["id"].astype(str).unique())
            except Exception:
                return set()
        return set()

    def has_quota(self, cost: int) -> bool:
        """Check whether sufficient quota remains before making a request."""
        if self.quota_used + cost > self.max_quota:
            logging.warning("API quota nearly exhausted. Stopping ingestion for today.")
            return False
        return True

    def get_channel_uploads(self, handle: str, limit: int = 100):
        """
        Retrieve up to `limit` videos from a channel using its uploads playlist.

        Quota cost:
            ~1 unit for channels().list()
            ~1 unit per playlistItems page (max 50 videos per page)
        """
        if not self.has_quota(2):
            return []

        try:
            ch_resp = (
                self.youtube.channels()
                .list(part="contentDetails", forHandle=handle)
                .execute()
            )
            self.quota_used += 1

            if not ch_resp.get("items"):
                logging.warning(f"Invalid or unresolved handle: {handle}")
                return []

            playlist_id = ch_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

            all_items = []
            next_page_token = None

            while len(all_items) < limit:
                if not self.has_quota(1):
                    break

                v_resp = (
                    self.youtube.playlistItems()
                    .list(
                        part="snippet,contentDetails",
                        playlistId=playlist_id,
                        maxResults=min(50, limit - len(all_items)),
                        pageToken=next_page_token,
                    )
                    .execute()
                )
                self.quota_used += 1

                items = v_resp.get("items", [])
                all_items.extend(items)

                next_page_token = v_resp.get("nextPageToken")
                if not next_page_token:
                    break

            return all_items

        except Exception as e:
            logging.error(f"Pagination error for {handle}: {e}")
            return []

    def get_comments(self, video_id: str, limit: int):
        """
        Extract up to `limit` top-level comments for a given video (PAGINATED).

        Important:
          - commentThreads().list maxResults is 100 per page.
          - We paginate using pageToken until:
              * we collected `limit`, or
              * no nextPageToken, or
              * quota is about to run out.

        Quota cost:
          - commentThreads().list is typically 1 unit per call.
        """
        if limit <= 0:
            return []

        comments = []
        next_page_token = None

        while len(comments) < limit:
            if not self.has_quota(1):
                break

            try:
                c_resp = (
                    self.youtube.commentThreads()
                    .list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=min(100, limit - len(comments)),
                        pageToken=next_page_token,
                        textFormat="plainText",
                        order="time",  # newest first; remove if you prefer relevance
                    )
                    .execute()
                )
                self.quota_used += 1

                items = c_resp.get("items", [])
                for item in items:
                    try:
                        sn = item["snippet"]["topLevelComment"]["snippet"]
                        comments.append(sn)
                    except Exception:
                        # skip malformed items
                        continue

                next_page_token = c_resp.get("nextPageToken")
                if not next_page_token:
                    break

            except HttpError as e:
                # Common cases:
                # - 403 commentsDisabled
                # - 404 video not found / removed
                # - quotaExceeded
                msg = str(e)
                if "commentsDisabled" in msg or "disabled comments" in msg:
                    return []
                if "quotaExceeded" in msg:
                    logging.warning("Quota exceeded according to API response. Stopping.")
                    self.quota_used = self.max_quota
                    break
                logging.warning(f"Comments fetch failed for {video_id}: {e}")
                break
            except Exception:
                break

        return comments


def load_config():
    """Load ingestion targets and settings from JSON config file."""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


async def main():
    """
    Main ingestion loop.

    Iterates through:
        - Regions
        - Channel handles per region
        - Videos per channel
        - Comments per video

    Stops early if quota limit is reached.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    config = load_config()
    scraper = YouTubeHeavyScraper(API_KEY)

    video_batch = []
    comment_batch = []

    logging.info(f"Starting large-scale monitoring. Known video IDs: {len(scraper.scraped_ids)}")

    for target in config["targets"]:
        region = target["region"]
        logging.info(f"Region: {region}")

        for handle in target["handles"]:
            logging.info(f"Scanning handle: {handle}")

            videos = scraper.get_channel_uploads(
                handle,
                config["settings"]["videos_per_channel"],
            )

            for v in videos:
                vid_id = v["contentDetails"]["videoId"]

                if vid_id in scraper.scraped_ids:
                    continue

                v_snippet = v["snippet"]
                video_batch.append(
                    {
                        "id": vid_id,
                        "region": region,
                        "handle": handle,
                        "title": v_snippet["title"],
                        "date": v_snippet["publishedAt"],
                    }
                )

                logging.info(f"Fetching comments for video: {vid_id}")

                comments = scraper.get_comments(
                    vid_id,
                    config["settings"]["comments_per_video"],
                )

                for c in comments:
                    comment_batch.append(
                        {
                            "video_id": vid_id,
                            "author": c.get("authorDisplayName", ""),
                            "text": c.get("textDisplay", ""),
                            "date": c.get("publishedAt", ""),
                        }
                    )

                scraper.scraped_ids.add(vid_id)

                if scraper.quota_used >= scraper.max_quota:
                    break

            if scraper.quota_used >= scraper.max_quota:
                break

    # ------------------------------------------------------------
    # DATA PERSISTENCE
    # ------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if video_batch:
        df_v = pd.DataFrame(video_batch)
        df_v.to_csv(
            scraper.registry_path,
            mode="a",
            header=not os.path.exists(scraper.registry_path),
            index=False,
        )
        logging.info(f"Added {len(df_v)} new videos to registry.")

    if comment_batch:
        df_c = pd.DataFrame(comment_batch)
        df_c.to_csv(
            f"{OUTPUT_DIR}/comments_batch_{timestamp}.csv",
            index=False,
        )
        logging.info(f"Extracted {len(df_c)} new comments.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
