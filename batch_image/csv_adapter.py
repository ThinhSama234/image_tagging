"""CSV adapter: load dataset from CSV file.

Expected CSV format (no header):
    image_path,tag1|tag2|tag3,caption text here

- Column 1: image path (relative or absolute)
- Column 2: tags separated by pipe |
- Column 3: caption (optional, can be empty)

Multiple captions per image: duplicate rows with same image_path, different captions.
"""

import csv
from collections import defaultdict
from typing import List

from .base_adapter import BaseAdapter, RawEntry


class CsvAdapter(BaseAdapter):
    """Load entries from a CSV file."""

    def __init__(self, csv_path: str, delimiter: str = ",", tag_sep: str = "|"):
        self.csv_path = csv_path
        self.delimiter = delimiter
        self.tag_sep = tag_sep

    def load_entries(self) -> List[RawEntry]:
        # Group by image_path to merge multiple captions
        grouped = defaultdict(lambda: {"tags": set(), "captions": []})

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            for row in reader:
                if not row or not row[0].strip():
                    continue

                img_path = row[0].strip()
                tags_str = row[1].strip() if len(row) > 1 else ""
                caption = row[2].strip() if len(row) > 2 else ""

                tags = [t.strip().lower() for t in tags_str.split(self.tag_sep) if t.strip()]
                grouped[img_path]["tags"].update(tags)
                if caption:
                    grouped[img_path]["captions"].append(caption)

        entries = []
        for img_path, data in grouped.items():
            entries.append(RawEntry(
                image_path=img_path,
                tags=sorted(data["tags"]),
                captions=data["captions"],
            ))
        return entries
