"""
S3 Utilities
Handles downloading PDFs from AWS S3 bucket on app startup.
"""

import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError

# AWS Configuration
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'research-paper-bucket2')
S3_REGION = os.getenv('AWS_REGION', 'us-east-1')
PDF_LOCAL_FOLDER = 'data/pdfs'

# Initialize S3 client
s3_client = None

def init_s3_client():
    """
    Initialize the S3 client.
    Works with IAM roles on EB or local AWS credentials.
    """
    global s3_client
    try:
        s3_client = boto3.client('s3', region_name=S3_REGION)
        print(f"✓ S3 client initialized for region: {S3_REGION}")
        return True
    except Exception as e:
        print(f"✗ Error initializing S3 client: {e}")
        return False


def download_all_pdfs():
    """
    Download all PDFs from S3 bucket to local folder.
    Called when the app starts.
    
    Returns:
        dict: {
            'success': bool,
            'downloaded_count': int,
            'error': str or None
        }
    """
    if not s3_client:
        return {
            'success': False,
            'downloaded_count': 0,
            'error': 'S3 client not initialized'
        }
    
    try:
        # Create local folder if it doesn't exist
        os.makedirs(PDF_LOCAL_FOLDER, exist_ok=True)
        
        # List all objects in S3 bucket
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME)
        
        if 'Contents' not in response:
            print(f"ℹ No files found in S3 bucket: {S3_BUCKET_NAME}")
            return {
                'success': True,
                'downloaded_count': 0,
                'error': None
            }
        
        downloaded_count = 0
        errors = []
        
        # Download each PDF
        for obj in response.get('Contents', []):
            key = obj['Key']
            
            # Only download PDFs
            if key.endswith('.pdf'):
                try:
                    # Extract filename from S3 key
                    filename = key.split('/')[-1]
                    local_path = os.path.join(PDF_LOCAL_FOLDER, filename)
                    
                    # Skip if already exists (to save bandwidth)
                    if os.path.exists(local_path):
                        print(f"✓ Already exists: {filename}")
                        continue
                    
                    # Download from S3
                    print(f"⬇️  Downloading: {filename}...")
                    s3_client.download_file(S3_BUCKET_NAME, key, local_path)
                    downloaded_count += 1
                    print(f"✓ Downloaded: {filename}")
                    
                except ClientError as e:
                    error_msg = f"Failed to download {filename}: {str(e)}"
                    print(f"✗ {error_msg}")
                    errors.append(error_msg)
        
        print(f"\n✓ S3 sync complete! Downloaded {downloaded_count} PDFs")
        
        return {
            'success': len(errors) == 0,
            'downloaded_count': downloaded_count,
            'error': ' | '.join(errors) if errors else None
        }
    
    except NoCredentialsError:
        error_msg = "AWS credentials not found. Make sure IAM role is configured on EB or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY are set."
        print(f"✗ {error_msg}")
        return {
            'success': False,
            'downloaded_count': 0,
            'error': error_msg
        }
    
    except ClientError as e:
        error_msg = f"S3 client error: {str(e)}"
        print(f"✗ {error_msg}")
        return {
            'success': False,
            'downloaded_count': 0,
            'error': error_msg
        }
    
    except Exception as e:
        error_msg = f"Unexpected error downloading PDFs: {str(e)}"
        print(f"✗ {error_msg}")
        return {
            'success': False,
            'downloaded_count': 0,
            'error': error_msg
        }


def get_local_pdfs():
    """
    Get list of all locally available PDFs.
    
    Returns:
        list: List of PDF filenames
    """
    if not os.path.exists(PDF_LOCAL_FOLDER):
        return []
    
    pdfs = []
    for file in os.listdir(PDF_LOCAL_FOLDER):
        if file.endswith('.pdf'):
            pdfs.append(file)
    
    return pdfs
