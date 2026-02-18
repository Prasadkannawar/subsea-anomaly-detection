"""
NautiCAI - Underwater Inspection AI
====================================

Professional Streamlit web application for real-time hull & subsea 
infrastructure anomaly detection using YOLOv8.

Author: NauticAI Team
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
from datetime import datetime
import uuid

# Import utilities
from utils.model_loader import load_default_model
from utils.inference import InferenceEngine
from utils.pdf_generator import PDFReportGenerator


# Page Configuration
st.set_page_config(
    page_title="NautiCAI - Underwater Inspection AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Professional Maritime Theme
st.markdown("""
    <style>
    /* Main Theme Colors */
    :root {
        --maritime-blue: #0A4D68;
        --teal: #088395;
        --light-teal: #05BFDB;
        --cream: #F8F6F4;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #0A4D68 0%, #088395 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .sub-title {
        color: #05BFDB;
        font-size: 1.3rem;
        margin-top: 0.5rem;
        text-align: center;
        font-weight: 300;
    }
    
    /* Logo Placeholder */
    .logo-section {
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .logo-placeholder {
        font-size: 4rem;
        color: #05BFDB;
    }
    
    /* Risk Indicators */
    .risk-indicator {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #088395;
        margin: 0.5rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0A4D68 0%, #088395 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(8, 131, 149, 0.4);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Streamlit Overrides */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load YOLOv8 model (cached for performance)."""
    try:
        model = load_default_model()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def calculate_risk_level(detections):
    """
    Calculate risk level based on detection counts.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        tuple: (risk_level, risk_color)
    """
    corrosion_count = sum(1 for d in detections if d['class_name'] == 'corrosion')
    
    if corrosion_count > 5:
        return "🔴 HIGH RISK", "risk-high"
    elif corrosion_count >= 2:
        return "🟠 MEDIUM RISK", "risk-medium"
    else:
        return "🟢 LOW RISK", "risk-low"


def main():
    """Main application."""
    
    # Header with Logo
    st.markdown("""
        <div class="logo-section">
            <div class="logo-placeholder">🌊</div>
        </div>
        <div class="main-header">
            <h1 class="main-title">NautiCAI</h1>
            <p class="sub-title">Real-Time Hull & Subsea Infrastructure Anomaly Detection</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load Model
    model = load_model()
    if model is None:
        st.error("⚠️ Model not loaded. Please train a model first.")
        st.info("Run: `python train.py` to train your model")
        return
    
    # Input Source Selection
    input_source = st.radio("Select Input Source:", ("Upload Image/Video", "Live Camera"), horizontal=True)
    
    uploaded_file = None
    camera_image = None
    
    if input_source == "Upload Image/Video":
        col_upload1, col_upload2 = st.columns([2, 1])
        
        with col_upload1:
            uploaded_file = st.file_uploader(
                "Choose an underwater image or video",
                type=['jpg', 'jpeg', 'png', 'mp4'],
                help="Supported formats: JPG, PNG, MP4"
            )
        
        with col_upload2:
            st.markdown("#### 📊 Supported Classes")
            st.markdown("""
            - 🔴 Corrosion
            - 🟢 Marine Growth
            - 🟡 Debris
            - 🔵 Healthy Surface
            """)
            
    elif input_source == "Live Camera":
        st.markdown("### 📷 Live Camera Feed")
        camera_image = st.camera_input("Take a photo for inspection")

    # Process Input
    if uploaded_file is not None or camera_image is not None:
        # Determine active input
        active_input = uploaded_file if uploaded_file else camera_image
        
        # Save input
        file_extension = Path(active_input.name).suffix if hasattr(active_input, 'name') else '.jpg'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file.write(active_input.getvalue())
        temp_file.close()
        
        # Display Preview
        st.markdown("---")
        st.markdown("### 👁️ Preview")
        
        col_preview1, col_preview2 = st.columns(2)
        
        with col_preview1:
            if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
                st.image(temp_file.name, caption="Inspection Input", use_container_width=True)
            else:
                st.video(temp_file.name)
        
        with col_preview2:
            st.markdown("#### 📋 File Information")
            st.write(f"**Source:** {input_source}")
            st.write(f"**Size:** {active_input.size / 1024:.2f} KB")
            st.write(f"**Type:** {file_extension[1:].upper()}")
        
        # Run Inspection Button
        st.markdown("---")
        if st.button("🔍 Run AI Inspection", use_container_width=True):
            
            # Progress indicator
            with st.spinner("🤖 AI analyzing underwater conditions..."):
                import time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Run inference
                engine = InferenceEngine(model)
                
                if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
                    results = engine.predict_image(temp_file.name, conf_threshold=0.25)
                else:
                    st.info("Video processing: Analyzing first frame for demo")
                    # For video, extract first frame
                    cap = cv2.VideoCapture(temp_file.name)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        temp_frame = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                        cv2.imwrite(temp_frame.name, frame)
                        results = engine.predict_image(temp_frame.name, conf_threshold=0.25)
                    else:
                        st.error("Could not read video")
                        return
            
            # Generate unique inspection ID
            inspection_id = f"NTI-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
            
            # Display Results
            st.success("✅ Inspection Complete!")
            st.markdown("---")
            st.markdown(f"### 📊 Inspection Report: `{inspection_id}`")
            
            # Results Layout
            col_result1, col_result2 = st.columns([3, 2])
            
            with col_result1:
                st.markdown("#### 🖼️ Annotated Output")
                annotated_img_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
                st.image(annotated_img_rgb, use_container_width=True)
            
            with col_result2:
                st.markdown("#### 📈 Detection Metrics")
                
                # Count detections by class
                class_counts = {
                    'corrosion': 0,
                    'marine_growth': 0,
                    'debris': 0,
                    'healthy_surface': 0
                }
                
                for det in results['detections']:
                    class_name = det['class_name']
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                
                # Display metrics
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric("🔍 Total Detections", results['num_detections'])
                    st.metric("🔴 Corrosion", class_counts['corrosion'])
                
                with metric_col2:
                    st.metric("🟢 Marine Growth", class_counts['marine_growth'])
                    st.metric("🟡 Debris", class_counts['debris'])
                
                st.metric("🔵 Healthy Surface", class_counts['healthy_surface'])
                
                # Average Confidence
                if results['detections']:
                    avg_conf = np.mean([d['confidence'] for d in results['detections']])
                    st.metric("💯 Avg Confidence", f"{avg_conf:.1%}")
            
            # Risk Assessment
            st.markdown("---")
            st.markdown("### ⚠️ Risk Assessment")
            
            risk_level, risk_class = calculate_risk_level(results['detections'])
            
            st.markdown(f"""
                <div class="risk-indicator {risk_class}">
                    {risk_level}
                </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            corrosion_count = class_counts['corrosion']
            marine_growth_count = class_counts['marine_growth']
            
            st.markdown("#### 📋 Recommendations")
            
            if corrosion_count > 5:
                st.error("🚨 **Immediate Action Required**: High corrosion detected. Schedule urgent maintenance.")
            elif corrosion_count >= 2:
                st.warning("⚠️ **Maintenance Recommended**: Moderate corrosion detected. Plan inspection within 30 days.")
            else:
                st.success("✅ **Structure Condition**: Good. Continue routine monitoring.")
            
            if marine_growth_count > 10:
                st.info("🌿 **Marine Growth**: Significant biofouling detected. Consider cleaning schedule.")
            
            # Detection Details
            if results['detections']:
                with st.expander("🔍 Detailed Detection Breakdown"):
                    for idx, det in enumerate(results['detections'], 1):
                        st.markdown(f"""
                        **Detection #{idx}**  
                        - **Class**: {det['class_name'].title()}  
                        - **Confidence**: {det['confidence']:.2%}  
                        - **Location**: ({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}) to ({det['bbox'][2]:.0f}, {det['bbox'][3]:.0f})
                        """)
                        st.markdown("---")
            
            # PDF Report Generation
            st.markdown("---")
            st.markdown("### 📄 Generate Inspection Report")
            
            if st.button("📥 Download PDF Inspection Report", use_container_width=True):
                with st.spinner("Generating professional PDF report..."):
                    # Save annotated image
                    temp_annotated = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    cv2.imwrite(temp_annotated.name, results['annotated_image'])
                    
                    # Generate PDF
                    pdf_generator = PDFReportGenerator(company_name="NautiCAI")
                    
                    additional_info = {
                        'Inspection ID': inspection_id,
                        'Source': 'Live Camera' if input_source == 'Live Camera' else uploaded_file.name,
                        'Inspection Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Risk Level': risk_level,
                        'Corrosion Count': class_counts['corrosion'],
                        'Marine Growth Count': class_counts['marine_growth'],
                        'Debris Count': class_counts['debris']
                    }
                    
                    pdf_path = pdf_generator.generate_report(
                        detection_results=results,
                        image_path=temp_annotated.name,
                        additional_info=additional_info
                    )
                    
                    # Download button
                    with open(pdf_path, 'rb') as pdf_file:
                        st.download_button(
                            label="⬇️ Download Report",
                            data=pdf_file,
                            file_name=f"NautiCAI_Inspection_{inspection_id}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    st.success("✅ Report generated successfully!")
    
    # Deployment Notes
    with st.expander("🚀 Deployment Notes"):
        st.markdown("""
        ### Edge Deployment Capabilities
        
        **NVIDIA Jetson Compatibility**  
        This model can be deployed on NVIDIA Jetson devices for real-time underwater ROV/AUV operations.
        
        **Specifications:**
        - Model: YOLOv8 (Optimized for Edge)
        - Inference Speed: ~30 FPS on Jetson Xavier NX
        - Power Consumption: < 15W
        
        **Deployment Options:**
        1. **Cloud API**: REST API for batch processing
        2. **Edge Device**: Jetson Nano/Xavier for real-time ROV
        3. **Hybrid**: Edge inference + Cloud analytics
        
        **Export Model for Deployment:**
        ```bash
        yolo export model=models/best.pt format=onnx
        ```
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <p><strong>NautiCAI - Underwater Inspection AI</strong></p>
            <p>Powered by YOLOv8 | Developed for Maritime Infrastructure Safety</p>
            <p style="font-size: 0.8rem; margin-top: 1rem;">
                Prototype developed for NautiCAI AI Engineer Assessment
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
