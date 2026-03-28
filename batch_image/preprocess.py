"""Main preprocessing pipeline: RawEntry → finetune JSON annotation.

Responsibilities:
- Load tag list → build tag-to-index mapping
- Map raw tag names → indices (with fuzzy fallback)
- Auto-generate captions from tags if missing
- Validate image paths exist
- Train/val split
- Export JSON for finetune_dataset

Usage as CLI:
    python -m batch_image.preprocess \
        --adapter csv --csv-path data/labels.csv \
        --output-dir datasets/custom \
        --train-ratio 0.9

    python -m batch_image.preprocess \
        --adapter folder --image-dir data/images/ \
        --output-dir datasets/custom

    python -m batch_image.preprocess \
        --adapter folder --image-dir data/images/ --labels-file data/labels.txt \
        --output-dir datasets/custom
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

from .base_adapter import BaseAdapter, RawEntry
from .config import DEFAULT_TAG_LIST, DEFAULT_TRAIN_RATIO, NUM_CLASSES


# ---------------------------------------------------------------------------
# Tag mapping
# ---------------------------------------------------------------------------
def load_tag_index(tag_list_path: str) -> Dict[str, int]:
    """Load ram_tag_list.txt → {tag_name: index}."""
    tag_to_idx = {}
    with open(tag_list_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            tag = line.strip().lower()
            if tag:
                tag_to_idx[tag] = idx
    return tag_to_idx


def map_tags_to_indices(
    tags: List[str], tag_to_idx: Dict[str, int]
) -> List[int]:
    """Map tag names → RAM tag indices. Skip unknown tags."""
    indices = []
    for tag in tags:
        tag_lower = tag.strip().lower()
        if tag_lower in tag_to_idx:
            indices.append(tag_to_idx[tag_lower])
    return sorted(set(indices))


# ---------------------------------------------------------------------------
# Caption generation (fallback when no caption provided)
# ---------------------------------------------------------------------------
def generate_caption_from_tags(tags: List[str]) -> str:
    """Simple caption: 'a photo of tag1, tag2 and tag3'."""
    if not tags:
        return "a photo"
    if len(tags) == 1:
        return f"a photo of {tags[0]}"
    return f"a photo of {', '.join(tags[:-1])} and {tags[-1]}"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def preprocess_dataset(
    adapter: BaseAdapter,
    tag_list_path: str = DEFAULT_TAG_LIST,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = 42,
    image_root: str = "",
) -> Tuple[List[dict], List[dict]]:
    """Run full pipeline: adapter → validated annotations → train/val split.

    Returns:
        (train_annotations, val_annotations) — each is list of dicts ready for JSON export.
    """
    tag_to_idx = load_tag_index(tag_list_path)
    raw_entries = adapter.load_entries()

    annotations = []
    skipped_no_tags = 0
    skipped_no_image = 0

    for entry in raw_entries:
        # Validate image exists
        img_path = entry.image_path
        full_path = os.path.join(image_root, img_path) if image_root else img_path
        if not os.path.isfile(full_path):
            skipped_no_image += 1
            continue

        # Map tags → indices
        tag_indices = map_tags_to_indices(entry.tags, tag_to_idx)
        if not tag_indices:
            skipped_no_tags += 1
            continue

        # Captions: use provided or auto-generate
        captions = entry.captions if entry.captions else [generate_caption_from_tags(entry.tags)]

        # parse_label_id: per-caption tag indices
        # For simplicity, each caption gets the full union tags.
        # Override in adapter if you need per-caption parsing.
        parse_label_ids = [tag_indices for _ in captions]

        annotations.append({
            "image_path": img_path,
            "caption": captions,
            "union_label_id": tag_indices,
            "parse_label_id": parse_label_ids,
        })

    # Stats
    print(f"Loaded: {len(raw_entries)} raw entries")
    print(f"Valid:  {len(annotations)} annotations")
    print(f"Skipped (no matching tags): {skipped_no_tags}")
    print(f"Skipped (image not found):  {skipped_no_image}")

    # Train/val split
    random.seed(seed)
    shuffled = annotations[:]
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    train_ann = shuffled[:split_idx]
    val_ann = shuffled[split_idx:]

    print(f"Train: {len(train_ann)} | Val: {len(val_ann)}")
    return train_ann, val_ann


def export_json(annotations: List[dict], output_path: str):
    """Write annotations list to JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path} ({len(annotations)} entries)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_adapter(args) -> BaseAdapter:
    """Build adapter from CLI args."""
    if args.adapter == "csv":
        from .csv_adapter import CsvAdapter
        return CsvAdapter(
            csv_path=args.csv_path,
            delimiter=args.delimiter,
            tag_sep=args.tag_sep,
        )
    elif args.adapter == "folder":
        from .folder_adapter import FolderAdapter
        return FolderAdapter(
            image_dir=args.image_dir,
            labels_file=args.labels_file,
            tag_sep=args.tag_sep,
        )
    elif args.adapter == "flickr30k":
        from .flickr30k_adapter import Flickr30kAdapter
        return Flickr30kAdapter(
            tags_csv=args.tags_csv,
            captions_csv=args.captions_csv,
        )
    elif args.adapter == "voc":
        from .voc_adapter import VocAdapter
        return VocAdapter(
            root=args.voc_root or "data/",
            year=args.voc_year or "2012",
            image_set=args.voc_split or "val",
            download=not args.no_download,
        )
    else:
        raise ValueError(f"Unknown adapter: {args.adapter}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess images+tags → RAM finetune JSON"
    )
    parser.add_argument(
        "--adapter", required=True, choices=["csv", "folder", "flickr30k", "voc"],
        help="Data source adapter type"
    )
    parser.add_argument("--csv-path", help="CSV file path (for csv adapter)")
    parser.add_argument("--tags-csv", help="Tags CSV path (for flickr30k adapter)")
    parser.add_argument("--captions-csv", help="Captions CSV path (for flickr30k adapter, optional)")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter")
    parser.add_argument("--image-dir", help="Image directory (for folder adapter)")
    parser.add_argument("--labels-file", help="Labels file (for folder adapter, optional)")
    parser.add_argument("--tag-sep", default="|", help="Tag separator in source data")
    # VOC adapter args
    parser.add_argument("--voc-root", default="data/", help="VOC dataset root (for voc adapter)")
    parser.add_argument("--voc-year", default="2012", help="VOC year (for voc adapter)")
    parser.add_argument("--voc-split", default="val", choices=["train", "val", "trainval"],
                        help="VOC split (for voc adapter)")
    parser.add_argument("--no-download", action="store_true", help="Skip dataset download")
    parser.add_argument(
        "--tag-list", default=DEFAULT_TAG_LIST,
        help="Path to ram_tag_list.txt"
    )
    parser.add_argument(
        "--output-dir", default="datasets/custom",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--image-root", default="",
        help="Root prefix for image paths (for validation)"
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-split", action="store_true",
        help="Don't split, output single train.json"
    )

    args = parser.parse_args()
    adapter = build_adapter(args)

    train_ann, val_ann = preprocess_dataset(
        adapter=adapter,
        tag_list_path=args.tag_list,
        train_ratio=1.0 if args.no_split else args.train_ratio,
        seed=args.seed,
        image_root=args.image_root,
    )

    export_json(train_ann, os.path.join(args.output_dir, "train.json"))
    if val_ann and not args.no_split:
        export_json(val_ann, os.path.join(args.output_dir, "val.json"))

    # Also generate finetune config pointing to the output
    config_path = os.path.join(args.output_dir, "finetune_config.yaml")
    train_json = os.path.join(args.output_dir, "train.json")
    _write_finetune_config(config_path, train_json)


def _write_finetune_config(config_path: str, train_json: str):
    """Generate a starter finetune YAML config."""
    config = f"""train_file:
  - '{train_json}'

image_path_root: ""

vit: 'swin_l'
vit_grad_ckpt: True
vit_ckpt_layer: 0

image_size: 384
batch_size: 8

weight_decay: 0.05
init_lr: 5e-06
min_lr: 0
max_epoch: 5
warmup_steps: 500

class_num: {NUM_CLASSES}
"""
    with open(config_path, "w") as f:
        f.write(config)
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
