# SUIM to YOLO Conversion Guide

## Quick Start

### Step 1: Convert SUIM Masks to YOLO Bounding Boxes

From the project directory:

```bash
cd nauticai-underwater-detection

python convert_suim_to_yolo.py \
  --images_dir ../SUIM_FULL/train_val/images \
  --masks_dir ../SUIM_FULL/train_val/masks \
  --output_dir ./raw_data
```

**What it does:**
- Reads SUIM segmentation masks
- Finds object contours using OpenCV
- Converts to bounding boxes
- Normalizes to YOLO format (0-1)
- Saves as `.txt` label files

**Class Mapping:**
- SUIM PF (Plants) + RI (Reefs) → Class 1 (marine_growth)
- SUIM WR (Wrecks) → Class 0 (corrosion)
- SUIM SR (Sea-floor) → Class 3 (healthy_surface)
- Ignores: Background, Humans, Robots, Fish

### Step 2: Prepare Dataset (80/20 Split)

```bash
python dataset_prepare.py \
  --source_images ./raw_data/images \
  --source_labels ./raw_data/labels
```

### Step 3: Validate Dataset

```bash
python dataset_prepare.py --validate_only
```

### Step 4: Train YOLOv8

```bash
yolo detect train \
  data=data.yaml \
  model=yolov8s.pt \
  epochs=150 \
  imgsz=640 \
  batch=16
```

### Step 5: Deploy Model

```bash
copy runs\train\exp\weights\best.pt models\best.pt
streamlit run app.py
```

## Customization

### Adjust Minimum Box Size

To filter out very small detections:

```bash
python convert_suim_to_yolo.py \
  --images_dir ../SUIM_FULL/train_val/images \
  --masks_dir ../SUIM_FULL/train_val/masks \
  --output_dir ./raw_data \
  --min_area 200 \
  --min_width 15 \
  --min_height 15
```

## Output Format

**YOLO Label Format:**
```
class_id x_center y_center width height
```

All values normalized to [0, 1].

**Example:**
```
0 0.500000 0.300000 0.150000 0.200000
1 0.750000 0.600000 0.100000 0.120000
```

## Troubleshooting

**Issue:** "Images directory not found"
- Make sure you're in the `nauticai-underwater-detection` directory
- Check the path to your SUIM dataset

**Issue:** "No mask for image.jpg"
- Ensure mask files have same name as images
- Check if masks are `.bmp` or `.png`

**Issue:** "No objects detected"
- Lower `--min_area`, `--min_width`, `--min_height`
- Check mask pixel values (should be 0-7 for SUIM)
