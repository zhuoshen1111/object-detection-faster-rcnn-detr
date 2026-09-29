# Faster R-CNN vs. DETR on Pascal VOC 2012

This repository turns a CSCI 677 homework experiment into a reproducible, Colab-first object-detection project. It fine-tunes and compares two different detector families on the same Pascal VOC 2012 train/validation split:

- **Faster R-CNN with a ResNet-50 FPN** using Detectron2
- **DETR with a ResNet-50 backbone** using Hugging Face Transformers

The refactor focuses on a fair comparison: the same 20 classes, the official VOC 2012 split, the same validation images, and the same TorchMetrics `mAP@0.50` implementation for both models.

## Why this refactor was necessary

The original homework notebooks recorded useful experiments, but they also contained repeated setup cells, stale outputs from different Colab sessions, incorrect argument ordering in `register_pascal_voc`, and a DETR target-box pipeline that passed normalized `xyxy` boxes where the model expected processor-generated DETR labels. The cleaned notebook fixes those issues and separates historical results from metrics produced by the corrected pipeline.

## Current result status

| Model | Metric | Value | Status |
|---|---:|---:|---|
| Faster R-CNN | Historical Detectron2 AP50 | 72.73% | Present in the original notebook output |
| DETR | Standardized TorchMetrics mAP@0.50 | Pending rerun | The previously reported 56.3% has no matching DETR evaluation log |

The notebook writes a new `comparison_metrics.json` after both corrected evaluations finish. That file should replace the pending entry above before the repository is presented as a final benchmark.

## Key corrections

- Uses the official VOC 2012 `train.txt` and `val.txt` instead of silently creating a 70/30 split.
- Parses VOC's 1-based inclusive boxes into 0-based `xyxy` coordinates correctly.
- Registers Detectron2 datasets with explicit records and consistent class IDs.
- Sends BGR images to Detectron2 inference and RGB images to visualizers.
- Lets `DetrImageProcessor` create the normalized DETR targets.
- Pads variable-size DETR batches and passes `pixel_mask` to the model.
- Evaluates both models with one metric implementation and one target representation.
- Stores datasets, checkpoints, and generated outputs in Google Drive rather than GitHub.

## Repository structure

```text
.
├── notebooks/
│   └── object_detection_comparison_colab.ipynb
├── results/
│   ├── README.md
│   └── legacy_metrics.json
├── src/
│   ├── __init__.py
│   └── voc.py
├── tests/
│   └── test_voc.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Run in Google Colab

1. Open `notebooks/object_detection_comparison_colab.ipynb` in Colab.
2. Enable a GPU runtime.
3. Run the install cell, then restart the runtime only if Colab requests it.
4. Mount Google Drive when prompted.
5. Start with `QUICK_RUN = True` to validate the pipeline.
6. Set `QUICK_RUN = False` for the full training and evaluation run.

The default persistent directory is:

```text
/content/drive/MyDrive/object-detection-faster-rcnn-detr/
```

Large datasets, checkpoints, and generated figures are excluded from Git.

## Reproducibility notes

- Dataset: Pascal VOC 2012 train/validation archive
- Classes: 20 Pascal VOC object categories
- Random seed: 42
- Primary comparison metric: TorchMetrics `MeanAveragePrecision` with IoU threshold 0.50
- Faster R-CNN initialization: Detectron2 COCO-pretrained `R50-FPN 3x`
- DETR initialization: `facebook/detr-resnet-50` with a new 20-class prediction head

Object detection results depend on the Colab GPU, package versions, and training budget. Commit the final notebook with outputs plus the generated metrics JSON after the full run.

## Tests

The lightweight tests cover VOC XML parsing, difficult-object filtering, and bounding-box conversion:

```bash
python -m pytest -q
```

## Limitations

- A short Colab run is intended as a pipeline check, not a competitive DETR benchmark.
- DETR generally needs a larger training budget than Faster R-CNN to converge well.
- The historical Faster R-CNN number was produced by the homework workflow; the corrected notebook's standardized metric is the value to use for the final comparison.
