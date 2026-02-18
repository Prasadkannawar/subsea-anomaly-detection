# 🏗️ NautiCAI System Architecture

This diagram illustrates the high-level architecture and data flow of the NautiCAI system.

```mermaid
graph TD
    %% User Layer
    subgraph "User Interface Layer"
        User((User/Inspector))
        UI["Streamlit Frontend (app.py)"]
        Input1[("Live Camera Stream\n(st.camera_input)")]
        Input2[("File Upload\n(st.file_uploader)")]
    end

    %% Application Core
    subgraph "Application Core (Python)"
        Controller["Main Controller Logic"]
        State["Session State Manager\n(st.session_state)"]
        Preproc["Image Preprocessing\n(OpenCV/PIL)"]
    end

    %% AI Engine
    subgraph "AI Inference Engine"
        ModelLoader["Model Loader\n(@st.cache_resource)"]
        YOLO["YOLOv8 Identification Model\n(runs on PyTorch/ONNX)"]
        PostProc["Post-Processing\n(NMS, Confidence Filtering)"]
    end

    %% Reporting Layer
    subgraph "Output & Reporting"
        RiskEngine["Risk Assessment Logic"]
        PDFGen["PDF Report Generator\n(ReportLab)"]
        Dashboard["Analytics Dashboard\n(st.metric, st.image)"]
    end

    %% Data Flow Connections
    User -->|Direct Interaction| UI
    UI -->|Select Source| Input1
    UI -->|Select Source| Input2
    
    Input1 -->|Raw Frames| Preproc
    Input2 -->|Image/Video| Preproc
    
    Preproc -->|Normalized Tensor| Controller
    Controller -->|Request Inference| ModelLoader
    ModelLoader -->|Load Weights| YOLO
    
    Controller -->|Image Batch| YOLO
    YOLO -->|Raw Detections| PostProc
    PostProc -->|Bounding Boxes + Classes| Controller
    
    Controller -->|save results| State
    
    %% Output Generation
    Controller -->|Detection Data| RiskEngine
    RiskEngine -->|Risk Score| Dashboard
    
    Controller -->|Render UI| Dashboard
    Dashboard -->|Visual Feedback| User
    
    User -->|Request Report| Controller
    State -->|Retrieve Data| PDFGen
    PDFGen -->|Generate Document| PDF_File[("Inspection Report.pdf")]
    PDF_File -->|Download| User

    %% Styling
    classDef ui fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef core fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef ai fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef report fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    
    class UI,Input1,Input2 ui;
    class Controller,State,Preproc core;
    class ModelLoader,YOLO,PostProc ai;
    class RiskEngine,PDFGen,Dashboard,PDF_File report;
```

## Key Components

1.  **Frontend (Streamlit)**: Handles user interaction, camera access, and visualizes results.
2.  **State Management**: Persists analysis data to allow report generation without re-running inference.
3.  **AI Engine**: YOLOv8 model optimized for underwater object detection.
4.  **Reporting Engine**: Custom PDF generator using ReportLab to create professional documentation.
