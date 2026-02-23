"""
Recognize-Anything Web Demo
============================
Gradio-based web UI for RAM++, RAM, and Tag2Text inference.

Usage:
    python app.py
    python app.py --checkpoint ../pretrained/ram_plus_swin_large_14m.pth
    python app.py --model-type tag2text --checkpoint ../pretrained/tag2text_swin_14m.pth
"""

import argparse
import sys
import os

# Add parent directory to path so we can import ram package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
import torch
from PIL import Image

from ram.models import ram_plus, ram, tag2text
from ram import inference_ram, inference_ram_openset, inference_tag2text, get_transform


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
MODEL = None
MODEL_TYPE = None
TRANSFORM = None
DEVICE = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_type: str, checkpoint: str, image_size: int, device: torch.device):
    """Load the specified model and return it in eval mode."""
    if model_type == "ram_plus":
        vit = "swin_l"
        model = ram_plus(pretrained=checkpoint, image_size=image_size, vit=vit)
    elif model_type == "ram":
        vit = "swin_l"
        model = ram(pretrained=checkpoint, image_size=image_size, vit=vit)
    elif model_type == "tag2text":
        vit = "swin_b"
        delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]
        model = tag2text(
            pretrained=checkpoint,
            image_size=image_size,
            vit=vit,
            delete_tag_index=delete_tag_index,
        )
        model.threshold = 0.68
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------
def predict_tags(input_image, specified_tags="None"):
    """Run inference on a single image and return results."""
    global MODEL, MODEL_TYPE, TRANSFORM, DEVICE

    if MODEL is None:
        return "Model not loaded. Please check your checkpoint path.", "", ""

    if input_image is None:
        return "Please upload an image.", "", ""

    # Convert to PIL if needed (Gradio may pass numpy array)
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)

    # Ensure RGB
    input_image = input_image.convert("RGB")

    # Transform and move to device
    image_tensor = TRANSFORM(input_image).unsqueeze(0).to(DEVICE)

    # Run inference based on model type
    if MODEL_TYPE in ("ram_plus", "ram"):
        tags_en, tags_cn = inference_ram(image_tensor, MODEL)
        return tags_en, tags_cn, ""

    elif MODEL_TYPE == "tag2text":
        tag_input = specified_tags if specified_tags and specified_tags.strip() else "None"
        model_tags, user_tags, caption = inference_tag2text(
            image_tensor, MODEL, tag_input
        )
        user_tags_str = user_tags if user_tags else "N/A"
        return model_tags, user_tags_str, caption

    return "Unknown model type", "", ""


def predict_with_confidence(input_image):
    """Run inference and return tags with confidence scores (RAM/RAM++ only)."""
    global MODEL, MODEL_TYPE, TRANSFORM, DEVICE

    if MODEL is None or input_image is None:
        return "Model not loaded or no image provided."

    if MODEL_TYPE not in ("ram_plus", "ram"):
        return "Confidence scores only available for RAM/RAM++."

    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)

    input_image = input_image.convert("RGB")
    image_tensor = TRANSFORM(input_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_embeds = MODEL.image_proj(MODEL.visual_encoder(image_tensor))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(DEVICE)

        if MODEL_TYPE == "ram_plus":
            # RAM++ with description reweighting
            import torch.nn.functional as F

            image_cls_embeds = image_embeds[:, 0, :]
            image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(
                dim=-1, keepdim=True
            )

            des_per_class = int(MODEL.label_embed.shape[0] / MODEL.num_class)
            reweight_scale = MODEL.reweight_scale.exp()
            logits_per_image = reweight_scale * image_cls_embeds @ MODEL.label_embed.t()
            logits_per_image = logits_per_image.view(1, -1, des_per_class)

            weight_normalized = F.softmax(logits_per_image, dim=2)
            reshaped_value = MODEL.label_embed.view(-1, des_per_class, 512)
            product = weight_normalized[0].unsqueeze(-1) * reshaped_value
            label_embed_reweight = product.sum(dim=1).unsqueeze(0)

            label_embed = torch.nn.functional.relu(
                MODEL.wordvec_proj(label_embed_reweight)
            )
        else:
            # RAM
            label_embed = torch.nn.functional.relu(
                MODEL.wordvec_proj(MODEL.label_embed)
            ).unsqueeze(0)

        tagging_embed, _ = MODEL.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode="tagging",
        )

        logits = torch.sigmoid(MODEL.fc(tagging_embed).squeeze(-1))

    # Get scores and tags
    scores = logits[0].cpu().numpy()
    tag_list = MODEL.tag_list

    # Build results: tag → score, sorted by score descending
    results = []
    for i, score in enumerate(scores):
        threshold = (
            MODEL.class_threshold[i].item()
            if hasattr(MODEL, "class_threshold")
            else 0.5
        )
        if score >= threshold:
            results.append((tag_list[i], float(score)))

    results.sort(key=lambda x: x[1], reverse=True)

    # Format as table
    lines = []
    for tag, score in results[:50]:  # top 50
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        lines.append(f"{tag:<30} {bar} {score:.3f}")

    return "\n".join(lines) if lines else "No tags detected."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    """Build the Gradio interface."""
    global MODEL_TYPE

    with gr.Blocks(
        title="Recognize Anything Demo",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 1.5em; }
        .output-box { font-family: monospace; font-size: 14px; }
        """,
    ) as demo:

        gr.HTML(
            """
            <h1 class="main-title">🏷️ Recognize Anything Demo</h1>
            <p class="subtitle">
                Upload an image to get automatic tags and captions using
                <strong>RAM++ / RAM / Tag2Text</strong> models.
            </p>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=400,
                )

                if MODEL_TYPE == "tag2text":
                    specified_tags = gr.Textbox(
                        label="Specified Tags (Tag2Text only)",
                        placeholder="e.g. dog, grass (leave empty for auto)",
                        value="None",
                    )
                else:
                    specified_tags = gr.Textbox(
                        visible=False,
                        value="None",
                    )

                run_btn = gr.Button(
                    "🔍 Recognize",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                tags_en = gr.Textbox(
                    label="🏷️ Image Tags (English)",
                    lines=4,
                    interactive=False,
                )
                tags_cn = gr.Textbox(
                    label="🏷️ Image Tags (Chinese)" if MODEL_TYPE != "tag2text" else "🏷️ User Specified Tags",
                    lines=2,
                    interactive=False,
                )

                if MODEL_TYPE == "tag2text":
                    caption_output = gr.Textbox(
                        label="📝 Image Caption",
                        lines=3,
                        interactive=False,
                    )
                else:
                    caption_output = gr.Textbox(
                        visible=False,
                    )

        # Confidence scores section (RAM/RAM++ only)
        if MODEL_TYPE in ("ram_plus", "ram"):
            gr.HTML("<hr><h3 style='text-align:center;'>📊 Tag Confidence Scores</h3>")
            with gr.Row():
                confidence_btn = gr.Button(
                    "📊 Show Confidence Scores",
                    variant="secondary",
                )
            confidence_output = gr.Textbox(
                label="Confidence Scores (top 50 tags)",
                lines=15,
                interactive=False,
                elem_classes=["output-box"],
            )
            confidence_btn.click(
                fn=predict_with_confidence,
                inputs=[input_image],
                outputs=[confidence_output],
            )

        # Wire up main button
        run_btn.click(
            fn=predict_tags,
            inputs=[input_image, specified_tags],
            outputs=[tags_en, tags_cn, caption_output],
        )

        # Examples
        example_dir = os.path.join(os.path.dirname(__file__), "..", "images")
        example_images = []
        if os.path.isdir(example_dir):
            for fname in sorted(os.listdir(example_dir)):
                fpath = os.path.join(example_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp")
                ):
                    example_images.append(fpath)
            # Also check demo subfolder
            demo_dir = os.path.join(example_dir, "demo")
            if os.path.isdir(demo_dir):
                for fname in sorted(os.listdir(demo_dir)):
                    fpath = os.path.join(demo_dir, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".webp")
                    ):
                        example_images.append(fpath)

        if example_images:
            gr.HTML("<hr><h3 style='text-align:center;'>🖼️ Example Images</h3>")
            gr.Examples(
                examples=[[img] for img in example_images[:8]],
                inputs=[input_image],
                label="Click an example to try",
            )

        # Footer
        gr.HTML(
            """
            <hr>
            <div style="text-align: center; color: #888; font-size: 12px;">
                <p>
                    <strong>Models:</strong> RAM++ | RAM | Tag2Text &nbsp;|&nbsp;
                    <strong>Backbone:</strong> Swin Transformer &nbsp;|&nbsp;
                    <strong>Tags:</strong> 4585 categories
                </p>
                <p>Powered by Recognize-Anything &amp; Gradio</p>
            </div>
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Recognize-Anything Web Demo")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ram_plus", "ram", "tag2text"],
        default="ram_plus",
        help="Model type (default: ram_plus)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. If not provided, will look in ../pretrained/",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Input image size (default: 384)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    return parser.parse_args()


def auto_find_checkpoint(model_type: str) -> str:
    """Try to find a checkpoint in the default pretrained directory."""
    pretrained_dir = os.path.join(os.path.dirname(__file__), "..", "pretrained")
    candidates = {
        "ram_plus": ["ram_plus_swin_large_14m.pth"],
        "ram": ["ram_swin_large_14m.pth"],
        "tag2text": ["tag2text_swin_14m.pth"],
    }
    for name in candidates.get(model_type, []):
        path = os.path.join(pretrained_dir, name)
        if os.path.isfile(path):
            return path
    return None


if __name__ == "__main__":
    args = parse_args()

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # Find checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = auto_find_checkpoint(args.model_type)
        if checkpoint is None:
            print(
                f"ERROR: No checkpoint found for {args.model_type}.\n"
                f"Please provide --checkpoint or place weights in ../pretrained/"
            )
            sys.exit(1)
    print(f"Checkpoint: {checkpoint}")

    # Load model
    MODEL_TYPE = args.model_type
    print(f"Loading {MODEL_TYPE} model...")
    MODEL = load_model(MODEL_TYPE, checkpoint, args.image_size, DEVICE)
    TRANSFORM = get_transform(image_size=args.image_size)
    print(f"Model loaded successfully!")

    # Build and launch UI
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
