"""Folder adapter: load dataset from directory structure.

Supports two layouts:

Layout A — flat folder with a labels file:
    images/
        img001.jpg
        img002.jpg
    labels.txt  (format: filename,tag1|tag2|tag3)

Layout B — class-name subfolders (each subfolder = one tag):
    images/
        dog/
            img001.jpg
        cat/
            img002.jpg
"""

import os
from collections import defaultdict
from typing import List, Optional

from .base_adapter import BaseAdapter, RawEntry
from .config import IMAGE_EXTENSIONS


class FolderAdapter(BaseAdapter):
    """Load entries from a folder of images."""

    def __init__(
        self,
        image_dir: str,
        labels_file: Optional[str] = None,
        tag_sep: str = "|",
    ):
        """
        Args:
            image_dir: root directory containing images
            labels_file: optional txt/csv mapping filename→tags.
                         If None, uses subfolder names as tags (Layout B).
            tag_sep: separator for tags in labels_file
        """
        self.image_dir = image_dir
        self.labels_file = labels_file
        self.tag_sep = tag_sep

    def load_entries(self) -> List[RawEntry]:
        if self.labels_file:
            return self._load_from_labels_file()
        return self._load_from_subfolders()

    def _load_from_labels_file(self) -> List[RawEntry]:
        """Layout A: flat folder + labels.txt"""
        grouped = defaultdict(lambda: {"tags": set(), "captions": []})

        with open(self.labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 2)
                filename = parts[0].strip()
                tags_str = parts[1].strip() if len(parts) > 1 else ""
                caption = parts[2].strip() if len(parts) > 2 else ""

                img_path = os.path.join(self.image_dir, filename)
                tags = [t.strip().lower() for t in tags_str.split(self.tag_sep) if t.strip()]
                grouped[img_path]["tags"].update(tags)
                if caption:
                    grouped[img_path]["captions"].append(caption)

        entries = []
        for img_path, data in grouped.items():
            if os.path.isfile(img_path):
                entries.append(RawEntry(
                    image_path=img_path,
                    tags=sorted(data["tags"]),
                    captions=data["captions"],
                ))
        return entries

    def _load_from_subfolders(self) -> List[RawEntry]:
        """Layout B: subfolder names = tag names"""
        img_tags = defaultdict(set)

        for class_name in sorted(os.listdir(self.image_dir)):
            class_dir = os.path.join(self.image_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            tag = class_name.lower().replace("_", " ")
            for fname in os.listdir(class_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTENSIONS:
                    continue
                img_path = os.path.join(class_dir, fname)
                img_tags[img_path].add(tag)

        entries = []
        for img_path, tags in img_tags.items():
            entries.append(RawEntry(
                image_path=img_path,
                tags=sorted(tags),
                captions=[],
            ))
        return entries
