"""Flickr30k adapter: merge tags from flickr30k.csv + captions from results.csv.

Data sources:
- flickr30k.csv: file,gt,pred — tags in 'gt' column separated by ' | '
- results.csv: image_name| comment_number| comment — ~5 captions per image

Usage:
    python -m batch_image.preprocess \
        --adapter flickr30k \
        --tags-csv data_processing/flickr30k.csv \
        --captions-csv batch_image/archive/flickr30k_images/results.csv \
        --image-root batch_image/archive/flickr30k_images/flickr30k_images \
        --output-dir datasets/flickr30k
"""

import csv
from collections import defaultdict
from typing import Dict, List, Optional

from .base_adapter import BaseAdapter, RawEntry


def _parse_tags_csv(tags_csv_path: str) -> Dict[str, List[str]]:
    """Parse flickr30k.csv → {filename: [tag1, tag2, ...]}."""
    result = {}
    with open(tags_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["file"].strip()
            gt_str = row.get("gt", "").strip()
            if gt_str:
                tags = [t.strip().lower() for t in gt_str.split("|") if t.strip()]
            else:
                tags = []
            result[filename] = tags
    return result


def _parse_captions_csv(captions_csv_path: str) -> Dict[str, List[str]]:
    """Parse results.csv → {filename: [caption1, caption2, ...]}."""
    grouped = defaultdict(list)
    with open(captions_csv_path, "r", encoding="utf-8") as f:
        # Skip header
        next(f)
        for line in f:
            # Format: "image_name| comment_number| comment"
            parts = line.strip().split("| ")
            if len(parts) < 3:
                continue
            filename = parts[0].strip()
            caption = parts[2].strip()
            if filename and caption:
                grouped[filename].append(caption)
    return dict(grouped)


class Flickr30kAdapter(BaseAdapter):
    """Merge Flickr30k tags CSV + captions CSV into RawEntry list."""

    def __init__(
        self,
        tags_csv: str,
        captions_csv: Optional[str] = None,
    ):
        self.tags_csv = tags_csv
        self.captions_csv = captions_csv

    def load_entries(self) -> List[RawEntry]:
        # Load tags (required)
        tags_map = _parse_tags_csv(self.tags_csv)

        # Load captions (optional — if not provided, auto-generate from tags)
        captions_map: Dict[str, List[str]] = {}
        if self.captions_csv:
            captions_map = _parse_captions_csv(self.captions_csv)

        entries = []
        for filename, tags in tags_map.items():
            captions = captions_map.get(filename, [])
            entries.append(RawEntry(
                image_path=filename,
                tags=tags,
                captions=captions,
            ))

        return entries
