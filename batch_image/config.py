"""Default configuration for data preprocessing pipeline."""

import os

# Path to RAM tag list (4585 tags)
DEFAULT_TAG_LIST = os.path.join(
    os.path.dirname(__file__), "..", "ram", "data", "ram_tag_list.txt"
)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Default train/val split ratio
DEFAULT_TRAIN_RATIO = 0.9

# Max words in caption (matches ram/data/utils.py pre_caption)
MAX_CAPTION_WORDS = 30

# Total tag classes in RAM
NUM_CLASSES = 4585
