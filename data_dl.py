import os
from tqdm import tqdm
import objaverse.xl as oxl

DOWNLOAD_DIR = "objaverse_xl_data"

def download_all_objects():
    # Ensure download directory exists
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Get all annotations
    annotations = oxl.get_annotations(download_dir=DOWNLOAD_DIR)  

    try:
        oxl.download_objects(annotations, DOWNLOAD_DIR)
    except Exception as e:
        print("Error downloading objects")

if __name__ == "__main__":
    download_all_objects()
