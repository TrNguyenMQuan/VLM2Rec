import io
import os
import csv
import requests
from PIL import Image
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session():
    session = requests.Session()
    retries = Retry(
        total=5,                
        backoff_factor=1,       
        status_forcelist=[500, 502, 503, 504],  
        allowed_methods=["GET"] 
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def fix_url(url):
    if url.startswith("http://ecx.images-amazon.com"):
        url = url.replace("http://ecx.images-amazon.com", "https://m.media-amazon.com")
    if url.startswith("https://ecx.images-amazon.com"):
        url = url.replace("https://ecx.images-amazon.com", "https://m.media-amazon.com")
    return url

def download_images_from_csv(csv_path, image_dir):
    os.makedirs(image_dir, exist_ok=True)
    image_paths = []
    session = create_session()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Generate Image Pool"):
            item_id = row["sid"]
            imgurl = fix_url(row["imgurl"])
            save_path = os.path.join(image_dir, f"{item_id}.jpg")

            if os.path.exists(save_path):
                print(f"Skip (already exists): {save_path}")
                image_paths.append(save_path)
                continue

            try:
                resp = session.get(imgurl, timeout=10)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img.save(save_path)
                image_paths.append(save_path)
            except Exception as e:
                print(f"Failed to download {imgurl}: {e}")
                continue

    return image_paths

if __name__ == "__main__":
    import sys
    DATA_NAME = sys.argv[1]
    CSV_PATH = f"./processed/{DATA_NAME}_item_meta.csv"
    IMAGE_DIR = f"./processed/{DATA_NAME}_image_pool"

    download_images_from_csv(CSV_PATH, IMAGE_DIR)
