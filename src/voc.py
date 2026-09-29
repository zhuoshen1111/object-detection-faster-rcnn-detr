"""Pascal VOC 2012 parsing helpers shared by both detector pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def validate_voc_layout(voc_root: str | Path) -> Path:
    """Validate the directories and official train/validation split files."""

    root = Path(voc_root)
    required = (
        root / "Annotations",
        root / "JPEGImages",
        root / "ImageSets" / "Main" / "train.txt",
        root / "ImageSets" / "Main" / "val.txt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Pascal VOC paths:\n- " + "\n- ".join(missing))
    return root


def read_split_ids(voc_root: str | Path, split: str) -> list[str]:
    """Read image identifiers from an official Pascal VOC split file."""

    root = validate_voc_layout(voc_root)
    split_path = root / "ImageSets" / "Main" / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]


def parse_voc_annotation(
    xml_path: str | Path,
    *,
    include_difficult: bool = False,
) -> dict[str, Any]:
    """Parse one VOC XML annotation into zero-based absolute ``xyxy`` boxes.

    Pascal VOC stores 1-based inclusive coordinates. Following Detectron2's
    Pascal VOC loader, ``xmin`` and ``ymin`` are shifted down by one while
    ``xmax`` and ``ymax`` are kept as the exclusive upper bounds.
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    width = int(root.findtext("./size/width", default="0"))
    height = int(root.findtext("./size/height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        class_name = obj.findtext("name")
        if class_name not in VOC_CLASSES:
            continue

        difficult = int(obj.findtext("difficult", default="0"))
        if difficult and not include_difficult:
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = max(0.0, float(bbox.findtext("xmin", default="1")) - 1.0)
        ymin = max(0.0, float(bbox.findtext("ymin", default="1")) - 1.0)
        xmax = min(float(width), float(bbox.findtext("xmax", default="0")))
        ymax = min(float(height), float(bbox.findtext("ymax", default="0")))
        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append(
            {
                "class_name": class_name,
                "category_id": VOC_CLASSES.index(class_name),
                "bbox_xyxy": [xmin, ymin, xmax, ymax],
                "difficult": difficult,
            }
        )

    return {"width": width, "height": height, "objects": objects}


def build_voc_records(
    voc_root: str | Path,
    split: str,
    *,
    include_difficult: bool = False,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Build framework-neutral records for training and evaluation."""

    root = validate_voc_layout(voc_root)
    image_ids = read_split_ids(root, split)
    if max_samples is not None:
        image_ids = image_ids[:max_samples]

    records: list[dict[str, Any]] = []
    for numeric_id, voc_id in enumerate(image_ids, start=1):
        image_path = root / "JPEGImages" / f"{voc_id}.jpg"
        xml_path = root / "Annotations" / f"{voc_id}.xml"
        if not image_path.exists() or not xml_path.exists():
            raise FileNotFoundError(f"Missing image or annotation for VOC ID {voc_id}")

        parsed = parse_voc_annotation(xml_path, include_difficult=include_difficult)
        boxes = [obj["bbox_xyxy"] for obj in parsed["objects"]]
        labels = [obj["category_id"] for obj in parsed["objects"]]

        records.append(
            {
                "file_name": str(image_path),
                "image_id": numeric_id,
                "voc_id": voc_id,
                "height": parsed["height"],
                "width": parsed["width"],
                "boxes": boxes,
                "labels": labels,
            }
        )
    return records


def to_coco_annotations(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one neutral record to the COCO-style input expected by DETR."""

    annotations = []
    for annotation_id, (box, label) in enumerate(
        zip(record["boxes"], record["labels"]), start=1
    ):
        xmin, ymin, xmax, ymax = box
        box_width = xmax - xmin
        box_height = ymax - ymin
        annotations.append(
            {
                "id": annotation_id,
                "image_id": record["image_id"],
                "category_id": label,
                "bbox": [xmin, ymin, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            }
        )
    return {"image_id": record["image_id"], "annotations": annotations}
