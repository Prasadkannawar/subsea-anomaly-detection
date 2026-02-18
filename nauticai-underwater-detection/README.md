# 🌊 NauticAI - Underwater Anomaly Detection

> AI-powered underwater hull and subsea infrastructure inspection using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com/)

## 📋 Overview

NauticAI is a production-ready web application for underwater anomaly detection in hull and subsea infrastructure inspections. It leverages YOLOv8 object detection to identify corrosion, cracks, biofouling, and other structural anomalies.

### ✨ Features

- 🎯 **YOLOv8-Powered Detection**: State-of-the-art object detection for underwater anomalies
- 🖼️ **Image & Video Support**: Process both images and short video clips
- 📊 **Real-time Visualization**: View annotated results with bounding boxes and confidence scores
- 📄 **PDF Report Generation**: Automatic professional inspection reports
- 🎨 **Modern Web UI**: Clean, responsive interface built with Streamlit
- 🌊 **Underwater Dehazing**: Integrated CLAHE filter for clarity in murky waters
- ⚙️ **Configurable Thresholds**: Adjust confidence and IoU thresholds

## 🏗️ Project Structure

```
nauticai-underwater-detection/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data.yaml                 # YOLO dataset configuration
├── .gitignore               # Git ignore rules
│
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── model_loader.py       # Model loading and caching
│   ├── inference.py          # Detection engine
│   └── pdf_generator.py      # PDF report generation
│
├── models/                   # Trained model weights
│   └── best.pt              # Your trained YOLOv8 model (place here)
│
├── data/                     # Training dataset
│   ├── images/
│   │   ├── train/           # Training images
│   │   └── val/             # Validation images
│   └── labels/
│       ├── train/           # Training labels (YOLO format)
│       └── val/             # Validation labels
│
├── uploads/                  # Temporary uploaded files
└── outputs/                  # Detection results and reports
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster inference

### Installation

1. **Clone the repository**
   ```bash
   cd nauticai-underwater-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎓 Training Your Model

### 1. Prepare Your Dataset

Organize your underwater images and labels in YOLO format:

```
data/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        └── ...
```

**Label Format** (YOLO): Each `.txt` file contains one bounding box per line:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized to [0, 1].

### 2. Configure data.yaml

Update `data.yaml` with your class names:

```yaml
path: ./data
train: images/train
val: images/val

nc: 5  # number of classes
names:
  0: corrosion
  1: biofouling
  2: crack
  3: dent
  4: anomaly
```

### 3. Train the Model

Using Ultralytics CLI:

```bash
# Train YOLOv8 nano model (fastest)
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640

# Train YOLOv8 small model (balanced)
yolo detect train data=data.yaml model=yolov8s.pt epochs=100 imgsz=640

# Train YOLOv8 medium model (more accurate)
yolo detect train data=data.yaml model=yolov8m.pt epochs=150 imgsz=640
```

**Training Parameters:**
- `epochs`: Number of training epochs (50-200 recommended)
- `imgsz`: Image size (640 standard, 1280 for high-res)
- `batch`: Batch size (adjust based on GPU memory)
- `device`: GPU device (0, 1, etc.) or 'cpu'

**Advanced Training:**
```bash
yolo detect train \
  data=data.yaml \
  model=yolov8s.pt \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=0 \
  patience=50 \
  save=True \
  project=runs/train \
  name=underwater_v1
```

### 4. Use Your Trained Model

After training, copy the best weights to the models directory:

```bash
# Windows
copy runs\train\underwater_v1\weights\best.pt models\best.pt

# Linux/Mac
cp runs/train/underwater_v1/weights/best.pt models/best.pt
```

The app will automatically load `models/best.pt` when available.

## 📊 Using the Application

1. **Upload Media**: Use the sidebar to upload an underwater image or video
2. **Adjust Settings**: Set confidence threshold (0.1-1.0)
3. **Run Detection**: Click "Run Detection" button
4. **Review Results**: View annotated images and detection statistics
5. **Download Report**: Generate and download PDF inspection report

## ☁️ Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Add Model Files**
   - Upload `models/best.pt` using Git LFS or cloud storage
   - Update `model_loader.py` to download from cloud if needed

### Alternative Deployment Options

- **Docker**: Create a Dockerfile for containerized deployment
- **AWS/GCP/Azure**: Deploy on cloud platforms with GPU support
- **Local Server**: Run on local server with `streamlit run app.py --server.port 8501`

## 🔧 Configuration

### Model Configuration

Edit `utils/model_loader.py` to change default model path or add custom models.

### Detection Parameters

Adjust in Streamlit sidebar or modify defaults in `app.py`:
- Confidence threshold: 0.25 (default)
- IoU threshold: 0.45 (default)

### PDF Report Customization

Edit `utils/pdf_generator.py` to customize:
- Company name/logo
- Report styling
- Additional sections

## 📝 Dataset Sources & Annotations

### Recommended Annotation Tools

- [LabelImg](https://github.com/heartexlabs/labelImg): Simple bounding box annotation
- [Roboflow](https://roboflow.com/): Online annotation with auto-labeling
- [CVAT](https://github.com/opencv/cvat): Advanced computer vision annotation tool
- [Label Studio](https://labelstud.io/): ML-powered annotation platform

### Public Underwater Datasets (for reference)

- **Underwater Trash Detection**: Search Kaggle/Roboflow
- **Marine Life Detection**: NOAA datasets
- **Custom**: Collect from ROV/AUV inspection footage

## 🧪 Testing

Run inference on test images:

```bash
python -c "from utils.model_loader import load_default_model; from utils.inference import InferenceEngine; model = load_default_model(); engine = InferenceEngine(model); results = engine.predict_image('path/to/test.jpg'); print(results)"
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📧 Contact & Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Email: [your-email@example.com]
- Documentation: [link-to-docs]

## 🙏 Acknowledgments

- **Ultralytics YOLOv8**: State-of-the-art object detection framework
- **Streamlit**: Rapid web app development
- **OpenCV**: Computer vision processing
- **ReportLab**: PDF generation

---

**Built with ❤️ for safer underwater inspections**
