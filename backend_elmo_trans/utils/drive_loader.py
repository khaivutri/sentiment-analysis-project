import os
import io
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class DriveModelLoader:
    """
    Google Drive model loader with caching support
    """
    def __init__(self):
        # Load credentials from environment variables
        refresh_token = os.getenv('REFRESH_TOKEN')
        client_id = os.getenv('CLIENT_ID')
        client_secret = os.getenv('CLIENT_SECRET')
        
        if not all([refresh_token, client_id, client_secret]):
            raise ValueError(
                "Missing required environment variables: "
                "REFRESH_TOKEN, CLIENT_ID, or CLIENT_SECRET"
            )
        
        self.credentials = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            client_id=client_id,
            client_secret=client_secret,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        self.cache_dir = Path("./model_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_drive_service(self):
        """Get authenticated Google Drive service"""
        if self.credentials.expired:
            self.credentials.refresh(Request())
        return build('drive', 'v3', credentials=self.credentials)
    
    def get_file_id_by_name(self, folder_id, filename):
        """Get file ID from folder ID and filename"""
        service = self.get_drive_service()
        results = service.files().list(
            q=f"'{folder_id}' in parents and name='{filename}'",
            fields="files(id,name)"
        ).execute()
        files = results.get('files', [])
        
        if not files:
            raise FileNotFoundError(f"File '{filename}' not found in folder")
        
        return files[0]['id']
    
    def download_model_if_needed(self, folder_id, filename):
        """Download model file with caching"""
        cache_path = self.cache_dir / filename
        
        try:
            file_id = self.get_file_id_by_name(folder_id, filename)
            
            if self._should_download(file_id, cache_path):
                print(f"📥 Downloading {filename} from Google Drive...")
                self._download_file(file_id, cache_path)
                print(f"✅ Downloaded to {cache_path}")
            else:
                print(f"📂 Using cached model: {cache_path}")
                
            return str(cache_path)
            
        except Exception as e:
            print(f"❌ Google Drive download failed: {e}")
            if cache_path.exists():
                print(f"📂 Using existing cached file: {cache_path}")
                return str(cache_path)
            raise FileNotFoundError(f"Cannot load model: Drive failed and no cache exists")
    
    def _should_download(self, file_id, cache_path):
        """Check if file should be downloaded (simple check for now)"""
        return not cache_path.exists()
    
    def _download_file(self, file_id, output_path):
        """Download file from Google Drive"""
        service = self.get_drive_service()
        request = service.files().get_media(fileId=file_id)
        
        with open(output_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    print(f"📊 Download progress: {progress}%")
    
    
    def download_elmo_transformer_model(self):
        """
        Download the ELMo Transformer model from Google Drive
        
        Returns:
            str: Path to the downloaded model file
        """
        folder_id = "10fyKtCIofs00kncvM89pwlD0xskiLKtG"
        filename = "elmo_transformer.pth"
        
        return self.download_model_if_needed(folder_id, filename)
    
  
