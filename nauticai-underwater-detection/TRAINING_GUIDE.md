# YOLOv8 Training Guide - Phase 3

## 🎯 Training Pipeline

This guide covers training, evaluation, and inference for the underwater anomaly detection model.

---

## 📋 Prerequisites

Before training, ensure:
- ✅ Dataset converted from SUIM to YOLO format
- ✅ Dataset split into train/val (80/20)
- ✅ `data.yaml` configured
- ✅ Dataset validated (no errors)

Check dataset status:
```bash
python dataset_prepare.py --validate_only
```

---

## 🚀 Training the Model

### Method 1: Using Python Script (Recommended)

```bash
python train.py
```

**With custom parameters:**
```bash
python train.py --epochs 100 --batch 32 --model yolov8s.pt
```

**Available parameters:**
- `--data`: Path to data.yaml (default: data.yaml)
- `--model`: Pretrained model (n/s/m/l/x, default: yolov8n.pt)
- `--epochs`: Number of epochs (default: 50)
- `--imgsz`: Image size (default: 640)
- `--batch`: Batch size (default: 16, use -1 for auto)
- `--device`: GPU device (0, 1, cpu, default: 0)
- `--name`: Experiment name (default: underwater_detection)

### Method 2: Using CLI Command

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

**Simplified:**
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

---

## 📊 What Happens During Training

### Transfer Learning Process

1. **Load Pretrained Weights**: YOLOv8n trained on COCO (80 classes)
2. **Adapt Network**: Replace detection head for our 4 classes
3. **Fine-tune**: Train entire network on underwater dataset
4. **Learn**: Model learns underwater-specific features (turbidity, lighting, marine textures)

### Augmentations Applied

**Geometric:**
- Horizontal flip: 50% probability
- Rotation: ±10 degrees
- Translation: ±10%
- Scaling: ±50%

**Color (Underwater-specific):**
- HSV Hue: 1.5% shift (lighting variations)
- HSV Saturation: 70% (color intensity)
- HSV Value: 40% (brightness)

**Advanced:**
- Mosaic: 100% (combines 4 images)
- MixUp: 10% (blends images)

### Metrics Logged

- **Precision**: % of correct detections
- **Recall**: % of ground truth objects detected
- **mAP@0.5**: Mean Average Precision at 50% IoU
- **mAP@0.5:0.95**: COCO metric (averaged across IoU thresholds)
- **Loss curves**: Box loss, class loss, DFL loss

### Output Files

```
runs/train/underwater_detection/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Last epoch checkpoint
├── results.csv          # Training metrics per epoch
├── confusion_matrix.png # Class confusion matrix
├── results.png          # Loss and metric curves
├── F1_curve.png         # F1-Confidence curve
├── P_curve.png          # Precision-Confidence curve
└── R_curve.png          # Recall-Confidence curve
```

---

## 📈 Evaluating the Model

```bash
python evaluate.py
```

**With custom model:**
```bash
python evaluate.py --model runs/train/underwater_detection/weights/best.pt
```

**Output:**
```
📊 Overall Performance:
  Precision:     0.8532
  Recall:        0.7891
  mAP@0.5:       0.8245
  mAP@0.5:0.95:  0.6123

📊 Per-Class Metrics:
  corrosion:      0.8541
  marine_growth:  0.8832
  debris:         0.7123
  healthy_surface: 0.8483
```

---

## 🔍 Testing Inference

### Single Image

```bash
python predict.py --source test_images/underwater001.jpg
```

### Directory of Images

```bash
python predict.py --source test_images/
```

### Video

```bash
python predict.py --source underwater_inspection.mp4
```

### Adjust Confidence Threshold

```bash
python predict.py --source test.jpg --conf 0.15
```

Lower confidence = more detections  
Higher confidence = more precise detections

### Output

```
Total Detections: 12

Detections by Class:
  corrosion:      3
  marine_growth:  6
  debris:         1
  healthy_surface: 2

📁 Results saved to: runs/predict/underwater_test/
```

---

## 🚢 Deploying to Production

### Step 1: Copy Best Model

```bash
# Windows
copy runs\train\underwater_detection\weights\best.pt models\best.pt

# Linux/Mac
cp runs/train/underwater_detection/weights/best.pt models/best.pt
```

### Step 2: Run Streamlit App

```bash
streamlit run app.py
```

### Step 3: Test in Browser

1. Upload underwater image
2. Adjust confidence threshold
3. Click "Run Detection"
4. Review results
5. Download PDF report

---

## 💡 Training Tips

### For Better Accuracy

1. **More Data**: Aim for 1000+ images per class
2. **Balanced Classes**: Similar number of samples per class
3. **Quality Labels**: Precise bounding boxes
4. **Longer Training**: 100-200 epochs for complex datasets
5. **Larger Model**: Try yolov8s.pt or yolov8m.pt

### For Faster Training

1. **Smaller Model**: Use yolov8n.pt
2. **Larger Batch**: Increase batch size (requires more GPU memory)
3. **Lower Resolution**: Try imgsz=416 or imgsz=512
4. **Fewer Epochs**: Start with 50, increase if needed

### For Production Deployment

1. **Export to ONNX**: For faster inference
   ```bash
   yolo export model=models/best.pt format=onnx
   ```
2. **Optimize for Edge**: Use yolov8n for embedded devices
3. **Quantization**: INT8 quantization for mobile deployment

---

## 🐛 Troubleshooting

### "CUDA out of memory"
**Solution**: Reduce batch size
```bash
python train.py --batch 8
```

### "No images found in dataset"
**Solution**: Run dataset preparation first
```bash
python convert_suim_to_yolo.py ...
python dataset_prepare.py ...
```

### "Model not converging"
**Solution**: 
- Check learning rate (try --lr0 0.001)
- Increase epochs
- Verify labels are correct

### Low mAP scores
**Solution**:
- Collect more data
- Improve label quality
- Use larger model (yolov8s/m)
- Train longer (100+ epochs)

---

## 📚 Model Comparison

| Model | Size | Speed (FPS) | mAP | Use Case |
|-------|------|-------------|-----|----------|
| yolov8n | 3.2 MB | ~45 | Good | Fast inference, edge devices |
| yolov8s | 11.2 MB | ~35 | Better | Balanced speed/accuracy |
| yolov8m | 25.9 MB | ~25 | Best | High accuracy required |
| yolov8l | 43.7 MB | ~20 | Excellent | Research, offline processing |
| yolov8x | 68.2 MB | ~15 | Superior | Maximum accuracy |

---

## 🎉 Complete Workflow

```bash
# 1. Convert SUIM to YOLO
python convert_suim_to_yolo.py --images_dir ../SUIM_FULL/train_val/images --masks_dir ../SUIM_FULL/train_val/masks --output_dir ./raw_data

# 2. Prepare dataset
python dataset_prepare.py --source_images ./raw_data/images --source_labels ./raw_data/labels

# 3. Validate
python dataset_prepare.py --validate_only

# 4. Train
python train.py --epochs 50 --batch 16

# 5. Evaluate
python evaluate.py

# 6. Test inference
python predict.py --source test_images/sample.jpg

# 7. Deploy
copy runs\train\underwater_detection\weights\best.pt models\best.pt
streamlit run app.py
```

---

**Ready for Phase 3 training!** 🚀
