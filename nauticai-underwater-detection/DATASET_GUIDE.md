# Dataset Preparation Guide - Phase 2

## 📁 Dataset Structure

Your dataset is now organized as follows:

```
nauticai-underwater-detection/
├── dataset/
│   ├── images/
│   │   ├── train/          # Training images (80%)
│   │   └── val/            # Validation images (20%)
│   └── labels/
│       ├── train/          # Training labels (YOLO format)
│       └── val/            # Validation labels (YOLO format)
├── dataset_prepare.py      # Dataset preparation script
├── data.yaml              # YOLOv8 dataset configuration
└── augmentation_config.yaml  # Data augmentation settings
```

## 🎯 Classes Defined

The dataset uses 4 classes for underwater hull inspection:

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | corrosion | Rust or corrosion on hull/infrastructure |
| 1 | marine_growth | Biofouling (barnacles, algae, marine organisms) |
| 2 | debris | Floating debris or attached foreign objects |
| 3 | healthy_surface | Clean, undamaged surface |

## 🚀 Using the Dataset Preparation Script

### Prerequisites

Ensure you have the SUIM dataset downloaded and extracted. Your raw data should be in this format:

```
raw_data/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── labels/
    ├── image001.txt
    ├── image002.txt
    └── ...
```

### Option 1: Basic Usage (80/20 Split)

```bash
python dataset_prepare.py --source_images ./raw_data/images --source_labels ./raw_data/labels
```

### Option 2: Custom Split Ratio (e.g., 85/15)

```bash
python dataset_prepare.py --source_images ./raw_data/images --source_labels ./raw_data/labels --split_ratio 0.85
```

### Option 3: Validate Existing Dataset

```bash
python dataset_prepare.py --validate_only
```

### Option 4: Show Detailed Issues

```bash
python dataset_prepare.py --validate_only --show_issues
```

### Option 5: Generate Augmentation Config Only

```bash
python dataset_prepare.py --augmentation_config_only
```

## 📊 What the Script Does

### 1. Dataset Splitting (80/20)
- Randomly shuffles all images
- Splits into training (80%) and validation (20%) sets
- Copies images to respective directories
- Moves corresponding label files with images
- Uses random seed for reproducibility

### 2. YOLO Format Validation
Checks every annotation file for:
- ✅ Each image has a corresponding label file
- ✅ Bounding box values are normalized (0-1)
- ✅ Each line has exactly 5 values: `class_id x_center y_center width height`
- ✅ All values are valid numbers
- ✅ No empty label files

**YOLO Label Format:**
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalized to [0, 1]:
- `x_center`: Center X coordinate / image width
- `y_center`: Center Y coordinate / image height
- `width`: Box width / image width
- `height`: Box height / image height

**Example:**
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

### 3. Dataset Statistics
Provides detailed statistics:
- Total number of images
- Number of training/validation images
- Total bounding boxes
- Class distribution (per class)
- Issues found (missing labels, invalid boxes, etc.)

### 4. Data Augmentation Configuration

The script generates `augmentation_config.yaml` with underwater-specific augmentations:

#### Geometric Augmentations
- **Horizontal Flip (50%)**: Simulates different viewing angles
- **Rotation (±10°)**: Minor rotations for orientation variance
- **Translation (10%)**: Simulates camera movement
- **Scaling (±50%)**: Different distances from objects

#### Color Augmentations (Underwater Conditions)
- **HSV Adjustment**: Simulates varying underwater lighting
  - Hue shift: 1.5%
  - Saturation: 70%
  - Value: 40%
- **Mosaic (100%)**: Combines 4 images for multi-object learning
- **MixUp (10%)**: Blends images for better generalization

## 🏋️ Training with Augmentation

Use the generated augmentation settings when training:

```bash
yolo detect train \
  data=data.yaml \
  model=yolov8s.pt \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  fliplr=0.5 \
  mosaic=1.0 \
  mixup=0.1
```

### Training Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `data` | data.yaml | Dataset configuration file |
| `model` | yolov8s.pt | Pre-trained model (n/s/m/l/x available) |
| `epochs` | 150 | Number of training iterations |
| `imgsz` | 640 | Input image size (640 recommended) |
| `batch` | 16 | Batch size (adjust based on GPU memory) |
| `hsv_h` | 0.015 | HSV hue augmentation |
| `hsv_s` | 0.7 | HSV saturation augmentation |
| `hsv_v` | 0.4 | HSV value augmentation |
| `fliplr` | 0.5 | Left-right flip probability |
| `mosaic` | 1.0 | Mosaic augmentation probability |
| `mixup` | 0.1 | MixUp augmentation probability |

## 🔧 Advanced Options

### Change Dataset Location

```bash
python dataset_prepare.py \
  --source_images ./raw_data/images \
  --source_labels ./raw_data/labels \
  --dataset_root ./custom_dataset
```

### Set Custom Random Seed

```bash
python dataset_prepare.py \
  --source_images ./raw_data/images \
  --source_labels ./raw_data/labels \
  --seed 123
```

## ✅ Validation Checklist

Before training, ensure:

- [ ] All images have corresponding label files
- [ ] No empty label files
- [ ] All bounding box values are between 0 and 1
- [ ] Class IDs are valid (0-3)
- [ ] Dataset split is balanced (roughly 80/20)
- [ ] Class distribution is reasonable (no extreme imbalance)

Run validation to check:
```bash
python dataset_prepare.py --validate_only
```

## 🐛 Common Issues & Solutions

### Issue: "No label file found for image.jpg"
**Solution:** Ensure label files have the same name as images but with `.txt` extension

### Issue: "Box values not normalized (0-1)"
**Solution:** Convert absolute coordinates to normalized format:
```python
x_center_norm = x_center / image_width
y_center_norm = y_center / image_height
width_norm = box_width / image_width
height_norm = box_height / image_height
```

### Issue: "Parse error: could not convert string to float"
**Solution:** Check label files for non-numeric values or extra spaces

### Issue: Class imbalance
**Solution:** 
- Use weighted loss during training
- Apply class-specific augmentation
- Collect more data for underrepresented classes

## 📈 Next Steps After Dataset Preparation

1. **Validate Dataset**
   ```bash
   python dataset_prepare.py --validate_only
   ```

2. **Review Statistics**
   - Check class distribution
   - Ensure no critical issues

3. **Train Model**
   ```bash
   yolo detect train data=data.yaml model=yolov8s.pt epochs=150 imgsz=640
   ```

4. **Monitor Training**
   - Watch loss curves
   - Check mAP (mean Average Precision)
   - Validate on test images

5. **Deploy Model**
   - Copy `runs/train/exp/weights/best.pt` to `models/best.pt`
   - Test in Streamlit app

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Format Specification](https://docs.ultralytics.com/datasets/detect/)
- [Data Augmentation Guide](https://docs.ultralytics.com/modes/train/#augmentation)
- [SUIM Dataset Paper](https://arxiv.org/abs/1911.11954)

## 💡 Tips for Better Results

1. **Data Quality > Quantity**
   - Ensure accurate annotations
   - Remove blurry/corrupted images
   - Verify class labels

2. **Augmentation Strategy**
   - More augmentation for smaller datasets
   - Reduce augmentation if model struggles to converge

3. **Class Balance**
   - Aim for relatively balanced classes
   - Use class weights if severe imbalance exists

4. **Validation Monitoring**
   - Don't rely solely on training metrics
   - Monitor validation mAP closely
   - Stop training if validation plateaus

---

**Ready for Phase 3: Model Training!** 🚀
