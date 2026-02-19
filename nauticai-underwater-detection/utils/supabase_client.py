import os
import streamlit as st
from supabase import create_client, Client

@st.cache_resource
def init_supabase() -> Client:
    """
    Initialize Supabase client using Streamlit secrets.
    Returns:
        Client: Supabase client instance or None if secrets are missing.
    """
    try:
        # Try to get secrets from Streamlit secrets management
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        
        return create_client(url, key)
    except Exception as e:
        # Return None if not configured, so the app doesn't crash
        return None

def upload_inspection(client: Client, image_file, inspection_data: dict):
    """
    Upload inspection image and data to Supabase.
    
    Args:
        client: Supabase client
        image_file: Bytes of the image
        inspection_data: Dictionary containing risk_score, timestamp, etc.
    """
    if not client:
        return False, "Supabase not configured"

    try:
        # 1. Upload Image to 'inspections' bucket
        # Note: In a real app, you'd handle file naming and conflict resolution
        file_path = f"inspections/{inspection_data['inspection_id']}.jpg"
        
        # Determine mime type
        mime_type = "image/jpeg"
        
        client.storage.from_("images").upload(
            file_path, 
            image_file, 
            {"content-type": mime_type}
        )
        
        # Get public URL
        public_url = client.storage.from_("images").get_public_url(file_path)
        
        # 2. Insert record into 'inspection_logs' table
        record = {
            "inspection_id": inspection_data['inspection_id'],
            "risk_score": float(inspection_data['risk_score']),
            "risk_level": inspection_data['risk_level'],
            "detections_count": int(inspection_data['detections_count']),
            "image_url": public_url,
            "created_at": inspection_data['timestamp']
        }
        
        data, count = client.table("inspection_logs").insert(record).execute()
        return True, "Upload successful!"
        
    except Exception as e:
        return False, str(e)
