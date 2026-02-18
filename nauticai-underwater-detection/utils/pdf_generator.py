"""
PDF Report Generator Module

This module generates professional PDF inspection reports with:
- Detection summary statistics
- Annotated images with bounding boxes
- Confidence scores and class distributions
- Metadata (date, time, model version)
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
from pathlib import Path
import tempfile
import cv2
from typing import Dict, List


class PDFReportGenerator:
    """
    Generates PDF inspection reports for underwater anomaly detection.
    """
    
    def __init__(self, company_name: str = "NauticAI"):
        """
        Initialize PDF report generator.
        
        Args:
            company_name: Company/project name for report header
        """
        self.company_name = company_name
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5490'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))
    
    def generate_report(self, 
                       detection_results: Dict,
                       image_path: str,
                       output_path: str = None,
                       additional_info: Dict = None) -> str:
        """
        Generate a complete PDF inspection report.
        
        Args:
            detection_results: Detection results from InferenceEngine
            image_path: Path to annotated image
            output_path: Output PDF path (if None, auto-generated)
            additional_info: Additional metadata to include
            
        Returns:
            str: Path to generated PDF file
        """
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"outputs/inspection_report_{timestamp}.pdf"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build report content
        story = []
        
        # Header
        story.extend(self._create_header())
        
        # Summary section
        story.extend(self._create_summary_section(detection_results, additional_info))
        
        # Detection details
        story.extend(self._create_detection_details(detection_results))
        
        # Annotated image
        story.extend(self._create_image_section(image_path))
        
        # Footer
        story.extend(self._create_footer())
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def _create_header(self) -> List:
        """Create report header."""
        elements = []
        
        # Title
        title = Paragraph(
            f"{self.company_name}<br/>Underwater Inspection Report",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.3 * inch))
        
        # Date and time
        date_str = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        date_para = Paragraph(
            f"<b>Report Generated:</b> {date_str}",
            self.styles['Normal']
        )
        elements.append(date_para)
        elements.append(Spacer(1, 0.5 * inch))
        
        return elements
    
    def _create_summary_section(self, results: Dict, additional_info: Dict = None) -> List:
        """Create summary statistics section."""
        elements = []
        
        # Section heading
        heading = Paragraph("Executive Summary", self.styles['CustomHeading'])
        elements.append(heading)
        
        # Summary statistics
        num_detections = results.get('num_detections', 0)
        
        summary_data = [
            ['Total Anomalies Detected:', str(num_detections)],
            ['Detection Model:', 'YOLOv8 (Underwater Trained)'],
            ['Confidence Threshold:', '0.25'],
        ]
        
        if additional_info:
            for key, value in additional_info.items():
                summary_data.append([f"{key}:", str(value)])
        
        # Create table
        table = Table(summary_data, colWidths=[3 * inch, 3 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements
    
    def _create_detection_details(self, results: Dict) -> List:
        """Create detailed detection table."""
        elements = []
        
        detections = results.get('detections', [])
        
        if not detections:
            para = Paragraph(
                "<b>No anomalies detected in this inspection.</b>",
                self.styles['Normal']
            )
            elements.append(para)
            elements.append(Spacer(1, 0.3 * inch))
            return elements
        
        # Section heading
        heading = Paragraph("Detected Anomalies", self.styles['CustomHeading'])
        elements.append(heading)
        
        # Table header
        table_data = [['#', 'Anomaly Type', 'Confidence', 'Location (x1, y1, x2, y2)']]
        
        # Add detection rows
        for idx, det in enumerate(detections, 1):
            bbox = det['bbox']
            bbox_str = f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})"
            table_data.append([
                str(idx),
                det['class_name'].title(),
                f"{det['confidence']:.2%}",
                bbox_str
            ])
        
        # Create table
        table = Table(table_data, colWidths=[0.5 * inch, 2 * inch, 1.5 * inch, 2.5 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5490')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.5 * inch))
        
        return elements
    
    def _create_image_section(self, image_path: str) -> List:
        """Add annotated image to report."""
        elements = []
        
        # Section heading
        heading = Paragraph("Annotated Inspection Image", self.styles['CustomHeading'])
        elements.append(heading)
        
        # Add image (resize to fit page)
        try:
            img = RLImage(image_path, width=6 * inch, height=4.5 * inch)
            elements.append(img)
        except Exception as e:
            error_para = Paragraph(
                f"<i>Error loading image: {str(e)}</i>",
                self.styles['Normal']
            )
            elements.append(error_para)
        
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements
    
    def _create_footer(self) -> List:
        """Create report footer."""
        elements = []
        
        elements.append(Spacer(1, 0.5 * inch))
        
        footer_text = Paragraph(
            "<i>This report was automatically generated by NauticAI underwater inspection system. "
            "For questions or concerns, please contact your inspection specialist.</i>",
            self.styles['Normal']
        )
        elements.append(footer_text)
        
        return elements


# Example usage:
if __name__ == "__main__":
    # Create sample report
    generator = PDFReportGenerator()
    
    sample_results = {
        'num_detections': 3,
        'detections': [
            {'class_name': 'corrosion', 'confidence': 0.89, 'bbox': [100, 150, 250, 300]},
            {'class_name': 'crack', 'confidence': 0.76, 'bbox': [300, 200, 450, 350]},
            {'class_name': 'biofouling', 'confidence': 0.92, 'bbox': [500, 100, 650, 250]},
        ]
    }
    
    # pdf_path = generator.generate_report(sample_results, 'path/to/image.jpg')
    # print(f"Report generated: {pdf_path}")
