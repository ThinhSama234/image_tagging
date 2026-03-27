"""Evaluate RAM/RAM++ model on a val set (JSON format from batch_image pipeline).

Computes: Precision, Recall, F1, mAP per-image.
Compares pretrained vs finetuned if two checkpoints given.

Usage:
    # Single model evaluation
    python evaluate.py \
        --checkpoint output/flickr30k_finetune/checkpoint_best.pth \
        --val-json datasets/flickr30k/val.json \
        --image-root batch_image/archive/flickr30k_images/flickr30k_images \
        --model-type ram_plus

    # Compare pretrained vs finetuned
    python evaluate.py \
        --checkpoint pretrained/ram_plus_swin_large_14m.pth \
        --checkpoint-b output/flickr30k_finetune/checkpoint_best.pth \
        --val-json datasets/flickr30k/val.json \
        --image-root batch_image/archive/flickr30k_images/flickr30k_images \
        --model-type ram_plus
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ram.models import ram_plus, ram
from ram import get_transform


def load_model(checkpoint_path, model_type, image_size, device):
    """Load model from checkpoint (finetune format or pretrained)."""
    if model_type == 'ram_plus':
        model = ram_plus(pretrained=checkpoint_path, image_size=image_size, vit='swin_l')
    elif model_type == 'ram':
        model = ram(pretrained=checkpoint_path, image_size=image_size, vit='swin_l')
    else:
        raise ValueError(f"Unsupported model_type for eval: {model_type}")

    # If checkpoint is a finetune save (has 'model' key), load state_dict
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
        print(f"Loaded finetuned checkpoint (epoch {ckpt.get('epoch', '?')})")

    model.eval()
    model = model.to(device)
    return model


def predict_tags(model, image_tensor):
    """Run inference, return set of predicted tag strings."""
    with torch.no_grad():
        tags, _ = model.generate_tag(image_tensor)
    # tags[0] is a string like "tag1 | tag2 | tag3"
    tag_set = set(t.strip().lower() for t in tags[0].split("|") if t.strip())
    return tag_set


def compute_metrics(gt_tags, pred_tags):
    """Compute P, R, F1 for a single image."""
    if not gt_tags and not pred_tags:
        return 1.0, 1.0, 1.0
    if not gt_tags or not pred_tags:
        return 0.0, 0.0, 0.0

    tp = len(gt_tags & pred_tags)
    precision = tp / len(pred_tags) if pred_tags else 0.0
    recall = tp / len(gt_tags) if gt_tags else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_tag_list(tag_list_path):
    """Load RAM tag list: index → tag name."""
    tags = []
    with open(tag_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            tag = line.strip().lower()
            if tag:
                tags.append(tag)
    return tags


def evaluate_model(model, val_data, image_root, transform, tag_list, device):
    """Evaluate model on val set. Returns per-image metrics."""
    results = []

    for entry in tqdm(val_data, desc="Evaluating"):
        img_path = os.path.join(image_root, entry['image_path'])
        if not os.path.isfile(img_path):
            continue

        try:
            image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        except Exception as e:
            print(f"Skip {img_path}: {e}")
            continue

        pred_tags = predict_tags(model, image)

        # GT tags from union_label_id → tag names
        gt_tag_ids = entry.get('union_label_id', [])
        gt_tags = set(tag_list[i] for i in gt_tag_ids if i < len(tag_list))

        precision, recall, f1 = compute_metrics(gt_tags, pred_tags)
        results.append({
            'image': entry['image_path'],
            'gt_count': len(gt_tags),
            'pred_count': len(pred_tags),
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

    return results


def print_summary(results, label="Model"):
    """Print aggregate metrics."""
    if not results:
        print(f"  {label}: No results")
        return {}

    avg_p = np.mean([r['precision'] for r in results])
    avg_r = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_gt = np.mean([r['gt_count'] for r in results])
    avg_pred = np.mean([r['pred_count'] for r in results])

    print(f"\n{'='*60}")
    print(f"  {label} — {len(results)} images")
    print(f"{'='*60}")
    print(f"  Precision:  {avg_p:.4f}")
    print(f"  Recall:     {avg_r:.4f}")
    print(f"  F1 Score:   {avg_f1:.4f}")
    print(f"  Avg GT tags:   {avg_gt:.1f}")
    print(f"  Avg Pred tags: {avg_pred:.1f}")
    print(f"{'='*60}")

    return {'precision': avg_p, 'recall': avg_r, 'f1': avg_f1}


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAM/RAM++ on val set')
    parser.add_argument('--checkpoint', required=True, help='Model A checkpoint path')
    parser.add_argument('--checkpoint-b', default='', help='Model B checkpoint path (for comparison)')
    parser.add_argument('--val-json', required=True, help='Validation JSON from batch_image pipeline')
    parser.add_argument('--image-root', default='', help='Root path for images')
    parser.add_argument('--model-type', default='ram_plus', choices=['ram_plus', 'ram'])
    parser.add_argument('--image-size', default=384, type=int)
    parser.add_argument('--tag-list', default='ram/data/ram_tag_list.txt')
    parser.add_argument('--max-samples', default=0, type=int, help='Limit eval samples (0=all)')
    parser.add_argument('--output', default='', help='Save results JSON to this path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load val data
    with open(args.val_json, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    if args.max_samples > 0:
        val_data = val_data[:args.max_samples]
    print(f"Val samples: {len(val_data)}")

    tag_list = load_tag_list(args.tag_list)
    transform = get_transform(image_size=args.image_size)

    # Evaluate Model A
    print(f"\nLoading Model A: {args.checkpoint}")
    model_a = load_model(args.checkpoint, args.model_type, args.image_size, device)
    results_a = evaluate_model(model_a, val_data, args.image_root, transform, tag_list, device)
    metrics_a = print_summary(results_a, label=f"Model A ({os.path.basename(args.checkpoint)})")
    del model_a
    torch.cuda.empty_cache()

    # Evaluate Model B (optional comparison)
    metrics_b = {}
    if args.checkpoint_b:
        print(f"\nLoading Model B: {args.checkpoint_b}")
        model_b = load_model(args.checkpoint_b, args.model_type, args.image_size, device)
        results_b = evaluate_model(model_b, val_data, args.image_root, transform, tag_list, device)
        metrics_b = print_summary(results_b, label=f"Model B ({os.path.basename(args.checkpoint_b)})")
        del model_b
        torch.cuda.empty_cache()

        # Comparison
        print(f"\n{'='*60}")
        print(f"  COMPARISON (B vs A)")
        print(f"{'='*60}")
        for m in ['precision', 'recall', 'f1']:
            diff = metrics_b[m] - metrics_a[m]
            arrow = "+" if diff > 0 else ""
            print(f"  {m:12s}: {arrow}{diff:.4f}")
        print(f"{'='*60}")

    # Save results
    if args.output:
        output_data = {
            'model_a': {'checkpoint': args.checkpoint, 'metrics': metrics_a},
        }
        if metrics_b:
            output_data['model_b'] = {'checkpoint': args.checkpoint_b, 'metrics': metrics_b}
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
