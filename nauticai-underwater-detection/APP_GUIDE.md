# NautiCAI Web Application Guide

## 🌊 Professional Streamlit Interface

Investor-demo-ready web application for underwater anomaly detection.

---

## 🚀 Quick Start

```bash
cd nauticai-underwater-detection
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ✨ Features

### 1. Modern Maritime Design
- **Blue/Teal Color Palette**: Professional maritime theme
- **Custom CSS Styling**: Gradient headers, card layouts
- **Logo Placeholder**: 🌊 Icon with clean typography
- **Responsive Layout**: Wide layout with column grids

### 2. File Upload
- **Supported Formats**: JPG, PNG, MP4
- **Preview**: Instant image/video preview
- **File Info**: Size, type, filename display

### 3. AI Inspection
- **YOLOv8 Detection**: Real-time inference
- **Progress Indicators**: Loading spinner + progress bar
- **Model Caching**: `@st.cache_resource` for performance
- **Annotated Output**: Bounding boxes with confidence scores

### 4. Detection Dashboard
**Metrics Displayed:**
- 🔍 Total Detections
- 🔴 Corrosion Count
- 🟢 Marine Growth Count
- 🟡 Debris Count
- 🔵 Healthy Surface Count
- 💯 Average Confidence

### 5. Risk Scoring System

**Logic:**
```python
if corrosion_count > 5:
    risk = "🔴 HIGH RISK"
elif corrosion_count >= 2:
    risk = "🟠 MEDIUM RISK"
else:
    risk = "🟢 LOW RISK"
```

**Visual Indicators:**
- 🟢 **Low Risk**: Green gradient banner
- 🟠 **Medium Risk**: Orange gradient banner
- 🔴 **High Risk**: Red gradient banner

### 6. Recommendations Engine

**Auto-generated based on detections:**
- High corrosion (>5): "Immediate Action Required"
- Medium corrosion (2-5): "Maintenance Recommended within 30 days"
- Low corrosion (<2): "Structure Condition Good"
- High marine growth (>10): "Consider cleaning schedule"

### 7. PDF Report Generation

**Includes:**
- ✅ Unique Inspection ID: `NTI-YYYYMMDD-XXXXXXXX`
- ✅ Inspection Date & Time
- ✅ File Information
- ✅ Risk Level Assessment
- ✅ Detection Metrics Table
- ✅ Annotated Image
- ✅ Class Distribution
- ✅ Recommendations

**Download:**
- One-click PDF download
- Named: `NautiCAI_Inspection_[ID].pdf`

### 8. Expandable Sections

**Deployment Notes:**
- NVIDIA Jetson compatibility
- Edge deployment specs
- Export instructions for ONNX

**Detection Details:**
- Per-detection breakdown
- Confidence scores
- Bounding box coordinates

---

## 🎨 UI Layout

```
┌─────────────────────────────────────────────┐
│  🌊 Logo                                    │
│  ╔═══════════════════════════════════════╗ │
│  ║  NautiCAI                             ║ │
│  ║  Real-Time Anomaly Detection          ║ │
│  ╚═══════════════════════════════════════╝ │
├─────────────────────────────────────────────┤
│  📁 Upload Media        │  📊 Classes       │
├─────────────────────────────────────────────┤
│  👁️ Preview                                 │
│  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Image     │  │  File Info          │  │
│  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────┤
│  [🔍 Run AI Inspection]                     │
├─────────────────────────────────────────────┤
│  🖼️ Annotated       │  📈 Metrics          │
│  ┌─────────────┐   │  ┌─────┬─────┐       │
│  │ Detection   │   │  │ 🔍  │ 🔴  │       │
│  │ Output      │   │  ├─────┼─────┤       │
│  └─────────────┘   │  │ 🟢  │ 🟡  │       │
│                    │  └─────┴─────┘       │
├─────────────────────────────────────────────┤
│  ⚠️ Risk Assessment                          │
│  ┌───────────────────────────────────────┐  │
│  │  🔴 HIGH RISK / 🟠 MEDIUM / 🟢 LOW    │  │
│  └───────────────────────────────────────┘  │
├─────────────────────────────────────────────┤
│  📋 Recommendations                         │
│  • Action items based on risk               │
├─────────────────────────────────────────────┤
│  🔍 Detection Details (Expandable)          │
├─────────────────────────────────────────────┤
│  [📥 Download PDF Report]                   │
├─────────────────────────────────────────────┤
│  🚀 Deployment Notes (Expandable)           │
├─────────────────────────────────────────────┤
│  Footer: NautiCAI - AI Engineer Assessment  │
└─────────────────────────────────────────────┘
```

---

## 💡 Usage Tips

### For Demo/Presentation

1. **Prepare Sample Images**: Use underwater hull inspection images
2. **Model Ready**: Ensure `models/best.pt` exists
3. **Clear Outputs**: Delete old runs for clean demo

### Performance Optimization

- **Model Caching**: Model loads once, cached across reruns
- **Progress Simulation**: Quick 1-second load for UX
- **Lazy Loading**: Model loads on first app start

### Customization

**Change Risk Thresholds:**
```python
# In app.py, function calculate_risk_level()
if corrosion_count > 3:  # Changed from 5
    return "🔴 HIGH RISK", "risk-high"
```

**Adjust Colors:**
```css
/* In st.markdown CSS section */
--maritime-blue: #YOUR_COLOR;
--teal: #YOUR_COLOR;
```

---

## 🐛 Troubleshooting

### "Model not found"
**Solution:**
```bash
# Train model first or copy existing model
python train.py
# Or copy trained model
copy runs\train\underwater_detection\weights\best.pt models\best.pt
```

### "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
```

### Video not processing
**Note:** Current version processes first frame of video for demo purposes.
Full video processing can be added via frame-by-frame loop.

### PDF not generating
**Check:** Ensure `outputs/` directory exists
```bash
mkdir outputs
```

---

## 🚢 Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy `app.py`
4. Add `models/best.pt` to repository (or use Git LFS)

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

### Production Considerations
- Use environment variables for configs
- Add authentication
- Implement rate limiting
- Set up logging
- Use production database for reports

---

## 📊 Feature Comparison

| Feature | Basic App | NautiCAI Pro |
|---------|-----------|--------------|
| File Upload | ✅ | ✅ |
| Detection | ✅ | ✅ |
| Risk Scoring | ❌ | ✅ |
| PDF Reports | Basic | Enhanced |
| Custom Theme | ❌ | ✅ |
| Metrics Dashboard | Basic | Advanced |
| Recommendations | ❌ | ✅ |
| Deployment Notes | ❌ | ✅ |

---

**Ready for investor demo!** 🎯
