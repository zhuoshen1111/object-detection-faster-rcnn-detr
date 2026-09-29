"""Utilities for the Pascal VOC object-detection comparison project."""

from .voc import VOC_CLASSES, build_voc_records, parse_voc_annotation, validate_voc_layout

__all__ = [
    "VOC_CLASSES",
    "build_voc_records",
    "parse_voc_annotation",
    "validate_voc_layout",
]
