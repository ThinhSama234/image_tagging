"""Evaluate RAM/RAM++ on Pascal VOC 2012 val set.

Downloads VOC 2012 automatically, runs inference, computes per-class AP and mAP.

Usage:
    # Evaluate pretrained RAM++
    python evaluate_voc.py \
        --checkpoint pretrained/ram_plus_swin_large_14m.pth

    # Compare pretrained vs finetuned
    python evaluate_voc.py \
        --checkpoint pretrained/ram_plus_swin_large_14m.pth \
        --checkpoint-b output/finetune/checkpoint_best.pth

    # Custom VOC root (skip download if already exists)
    python evaluate_voc.py \
        --checkpoint pretrained/ram_plus_swin_large_14m.pth \
        --voc-root /kaggle/working/data \
        --no-download
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VOCDetection
from tqdm import tqdm

from ram import get_transform
from ram.models import ram, ram_plus

# VOC 2012 has 20 object classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Map VOC class → set of RAM tag strings that count as correct prediction
VOC_TO_RAM_SYNONYMS = {
    "aeroplane": {"airplane", "aeroplane", "aircraft", "plane", "airliner"},
    "bicycle": {"bicycle", "bike"},
    "bird": {"bird"},
    "boat": {"boat", "ship", "watercraft"},
    "bottle": {"bottle"},
    "bus": {"bus"},
    "car": {"car", "automobile"},
    "cat": {"cat"},
    "chair": {"chair"},
    "cow": {"cow", "cattle"},
    "diningtable": {"table", "dining table"},
    "dog": {"dog"},
    "horse": {"horse"},
    "motorbike": {"motorcycle", "motorbike"},
    "person": {"person", "people", "man", "woman", "boy", "girl", "child"},
    "pottedplant": {"plant", "houseplant", "potted plant", "flower pot"},
    "sheep": {"sheep", "lamb"},
    "sofa": {"couch", "sofa"},
    "train": {"train", "locomotive"},
    "tvmonitor": {"television", "monitor", "tv", "screen"},
}


def load_model(checkpoint_path, model_type, image_size, device):
    """Load RAM/RAM++ model from checkpoint."""
    if model_type == "ram_plus":
        model = ram_plus(pretrained=checkpoint_path, image_size=image_size, vit="swin_l")
    elif model_type == "ram":
        model = ram(pretrained=checkpoint_path, image_size=image_size, vit="swin_l")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Handle finetuned checkpoint format
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded finetuned checkpoint (epoch {ckpt.get('epoch', '?')})")

    model.eval()
    return model.to(device)


def predict_tags(model, image_tensor):
    """Run inference → set of predicted tag strings."""
    with torch.no_grad():
        tags, _ = model.generate_tag(image_tensor)
    return set(t.strip().lower() for t in tags[0].split("|") if t.strip())


def extract_voc_classes(annotation):
    """Extract set of VOC class names from annotation dict."""
    objects = annotation["annotation"].get("object", [])
    if not isinstance(objects, list):
        objects = [objects]
    return set(obj["name"].strip().lower() for obj in objects if obj.get("name"))


def compute_per_class_ap(gt_labels, pred_scores):
    """Compute Average Precision for a single class.

    Args:
        gt_labels: list of 0/1 per image (ground truth)
        pred_scores: list of 0/1 per image (prediction)

    Returns:
        AP value (float)
    """
    gt = np.array(gt_labels)
    pred = np.array(pred_scores)

    # True positives and false positives
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def evaluate_on_voc(model, dataset, transform, device, max_samples=0):
    """Run evaluation on VOC dataset, return per-class metrics."""
    n = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)

    # Per-class ground truth and predictions (binary per image)
    gt_per_class = {c: [] for c in VOC_CLASSES}
    pred_per_class = {c: [] for c in VOC_CLASSES}

    for i in tqdm(range(n), desc="Evaluating on VOC"):
        _, annotation = dataset[i]
        img_path = dataset.images[i]

        try:
            image = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Skip {img_path}: {e}")
            continue

        # Ground truth VOC classes for this image
        gt_classes = extract_voc_classes(annotation)

        # RAM predicted tags
        pred_tags = predict_tags(model, image)

        # Per-class binary evaluation
        for voc_class in VOC_CLASSES:
            gt_present = 1 if voc_class in gt_classes else 0
            # Check if any synonym of this VOC class appears in predictions
            synonyms = VOC_TO_RAM_SYNONYMS[voc_class]
            pred_present = 1 if pred_tags & synonyms else 0

            gt_per_class[voc_class].append(gt_present)
            pred_per_class[voc_class].append(pred_present)

    return gt_per_class, pred_per_class


def print_results(gt_per_class, pred_per_class, label="Model"):
    """Compute and print per-class metrics + mAP."""
    print(f"\n{'='*70}")
    print(f"  {label} — VOC 2012 Val Results")
    print(f"{'='*70}")
    print(f"  {'Class':<15} {'Prec':>7} {'Recall':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5} {'GT':>5}")
    print(f"  {'-'*62}")

    all_metrics = {}
    f1_scores = []

    for voc_class in VOC_CLASSES:
        metrics = compute_per_class_ap(gt_per_class[voc_class], pred_per_class[voc_class])
        all_metrics[voc_class] = metrics
        f1_scores.append(metrics["f1"])
        gt_count = sum(gt_per_class[voc_class])

        print(
            f"  {voc_class:<15} {metrics['precision']:>7.3f} {metrics['recall']:>7.3f} "
            f"{metrics['f1']:>7.3f} {metrics['tp']:>5} {metrics['fp']:>5} {metrics['fn']:>5} {gt_count:>5}"
        )

    mean_f1 = np.mean(f1_scores)
    mean_prec = np.mean([m["precision"] for m in all_metrics.values()])
    mean_rec = np.mean([m["recall"] for m in all_metrics.values()])

    print(f"  {'-'*62}")
    print(f"  {'MEAN':<15} {mean_prec:>7.3f} {mean_rec:>7.3f} {mean_f1:>7.3f}")
    print(f"{'='*70}")

    return {"per_class": all_metrics, "mean_precision": mean_prec, "mean_recall": mean_rec, "mean_f1": mean_f1}


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAM/RAM++ on VOC 2012 val")
    parser.add_argument("--checkpoint", required=True, help="Model A checkpoint")
    parser.add_argument("--checkpoint-b", default="", help="Model B checkpoint (comparison)")
    parser.add_argument("--model-type", default="ram_plus", choices=["ram_plus", "ram"])
    parser.add_argument("--image-size", default=384, type=int)
    parser.add_argument("--voc-root", default="data/", help="VOC dataset root directory")
    parser.add_argument("--voc-year", default="2012", help="VOC year")
    parser.add_argument("--voc-split", default="val", choices=["train", "val", "trainval"])
    parser.add_argument("--no-download", action="store_true", help="Skip VOC download")
    parser.add_argument("--max-samples", default=0, type=int, help="Limit eval samples (0=all)")
    parser.add_argument("--output", default="", help="Save results JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load VOC dataset
    print(f"Loading VOC {args.voc_year} {args.voc_split}...")
    dataset = VOCDetection(
        root=args.voc_root,
        year=args.voc_year,
        image_set=args.voc_split,
        download=not args.no_download,
    )
    print(f"VOC samples: {len(dataset)}")

    transform = get_transform(image_size=args.image_size)

    # Evaluate Model A
    print(f"\nLoading Model A: {args.checkpoint}")
    model_a = load_model(args.checkpoint, args.model_type, args.image_size, device)
    gt_a, pred_a = evaluate_on_voc(model_a, dataset, transform, device, args.max_samples)
    results_a = print_results(gt_a, pred_a, label=f"Model A ({os.path.basename(args.checkpoint)})")
    del model_a
    torch.cuda.empty_cache()

    # Evaluate Model B (optional)
    results_b = None
    if args.checkpoint_b:
        print(f"\nLoading Model B: {args.checkpoint_b}")
        model_b = load_model(args.checkpoint_b, args.model_type, args.image_size, device)
        gt_b, pred_b = evaluate_on_voc(model_b, dataset, transform, device, args.max_samples)
        results_b = print_results(gt_b, pred_b, label=f"Model B ({os.path.basename(args.checkpoint_b)})")
        del model_b
        torch.cuda.empty_cache()

        # Comparison
        print(f"\n{'='*70}")
        print(f"  COMPARISON (B vs A)")
        print(f"{'='*70}")
        for metric_name in ["mean_precision", "mean_recall", "mean_f1"]:
            diff = results_b[metric_name] - results_a[metric_name]
            arrow = "+" if diff > 0 else ""
            print(f"  {metric_name:<18}: {arrow}{diff:.4f}")
        print(f"{'='*70}")

    # Save results
    if args.output:
        output_data = {"model_a": results_a}
        if results_b:
            output_data["model_b"] = results_b
        # Convert numpy values to native Python for JSON
        def to_native(obj):
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            return obj

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(to_native(output_data), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
