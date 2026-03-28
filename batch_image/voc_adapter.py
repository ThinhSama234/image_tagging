"""Pascal VOC adapter: load VOC 2012 annotations into RawEntry format.

Uses torchvision.datasets.VOCDetection to handle download + XML parsing.
Maps 20 VOC object classes to RAM-compatible tag names.

Usage:
    python -m batch_image.preprocess \
        --adapter voc \
        --voc-root data/ --voc-year 2012 --voc-split val \
        --output-dir datasets/pascal_voc_2012 \
        --no-split
"""

from typing import List

from .base_adapter import BaseAdapter, RawEntry

# VOC class name → RAM tag name(s) for matching
# Some VOC classes use different names in RAM's 4585 tag list
VOC_TO_RAM_TAGS = {
    "aeroplane": ["airplane"],
    "bicycle": ["bicycle"],
    "bird": ["bird"],
    "boat": ["boat"],
    "bottle": ["bottle"],
    "bus": ["bus"],
    "car": ["car"],
    "cat": ["cat"],
    "chair": ["chair"],
    "cow": ["cow"],
    "diningtable": ["table"],
    "dog": ["dog"],
    "horse": ["horse"],
    "motorbike": ["motorcycle"],
    "person": ["person"],
    "pottedplant": ["houseplant", "plant"],
    "sheep": ["sheep"],
    "sofa": ["couch"],
    "train": ["train"],
    "tvmonitor": ["television", "monitor"],
}


class VocAdapter(BaseAdapter):
    """Load Pascal VOC detection dataset as RawEntry list for RAM pipeline."""

    def __init__(
        self,
        root: str = "data/",
        year: str = "2012",
        image_set: str = "val",
        download: bool = True,
    ):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.download = download

    def load_entries(self) -> List[RawEntry]:
        from torchvision.datasets import VOCDetection

        dataset = VOCDetection(
            root=self.root,
            year=self.year,
            image_set=self.image_set,
            download=self.download,
        )

        entries = []
        for i in range(len(dataset)):
            img, annotation = dataset[i]
            img_path = dataset.images[i]

            # Extract unique object classes from XML annotation
            objects = annotation["annotation"].get("object", [])
            if not isinstance(objects, list):
                objects = [objects]

            voc_classes = set()
            for obj in objects:
                name = obj.get("name", "").strip().lower()
                if name:
                    voc_classes.add(name)

            # Map VOC classes → RAM tag names
            ram_tags = []
            for voc_class in voc_classes:
                mapped = VOC_TO_RAM_TAGS.get(voc_class, [voc_class])
                ram_tags.extend(mapped)

            if ram_tags:
                entries.append(RawEntry(
                    image_path=img_path,
                    tags=ram_tags,
                    captions=[],
                ))

        return entries
