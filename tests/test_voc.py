from pathlib import Path

from src.voc import parse_voc_annotation, to_coco_annotations


VOC_XML = """\
<annotation>
  <size><width>100</width><height>80</height><depth>3</depth></size>
  <object>
    <name>cat</name><difficult>0</difficult>
    <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox>
  </object>
  <object>
    <name>dog</name><difficult>1</difficult>
    <bndbox><xmin>50</xmin><ymin>40</ymin><xmax>100</xmax><ymax>80</ymax></bndbox>
  </object>
</annotation>
"""


def _write_xml(tmp_path: Path) -> Path:
    path = tmp_path / "sample.xml"
    path.write_text(VOC_XML)
    return path


def test_parse_voc_annotation_converts_coordinates(tmp_path):
    parsed = parse_voc_annotation(_write_xml(tmp_path))

    assert parsed["width"] == 100
    assert parsed["height"] == 80
    assert len(parsed["objects"]) == 1
    assert parsed["objects"][0]["class_name"] == "cat"
    assert parsed["objects"][0]["bbox_xyxy"] == [0.0, 1.0, 10.0, 20.0]


def test_parse_voc_annotation_can_include_difficult_objects(tmp_path):
    parsed = parse_voc_annotation(_write_xml(tmp_path), include_difficult=True)

    assert [obj["class_name"] for obj in parsed["objects"]] == ["cat", "dog"]
    assert parsed["objects"][1]["bbox_xyxy"] == [49.0, 39.0, 100.0, 80.0]


def test_coco_conversion_uses_xywh_and_zero_based_label(tmp_path):
    parsed = parse_voc_annotation(_write_xml(tmp_path))
    obj = parsed["objects"][0]
    record = {
        "image_id": 7,
        "boxes": [obj["bbox_xyxy"]],
        "labels": [obj["category_id"]],
    }

    converted = to_coco_annotations(record)
    annotation = converted["annotations"][0]

    assert converted["image_id"] == 7
    assert annotation["bbox"] == [0.0, 1.0, 10.0, 19.0]
    assert annotation["category_id"] == 7  # cat is index 7 in Pascal VOC
    assert annotation["area"] == 190.0
