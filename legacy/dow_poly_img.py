import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# List all three of your files here
JSON_FILES = [
    'polyvore/train_no_dup.json', 
    'polyvore/test_no_dup.json', 
    'polyvore/valid_no_dup.json'
]
OUTPUT_DIR = 'polyvore_images'

os.makedirs(OUTPUT_DIR, exist_ok=True)
download_tasks = []

# --- 2. GATHER ALL URLS ---
print("Scanning JSON files for image URLs...")
for file_path in JSON_FILES:
    try:
        with open(file_path, 'r') as f:
            outfits = json.load(f)
            
        for outfit in outfits:
            set_id = outfit['set_id']
            for item in outfit['items']:
                image_url = item.get('image')
                item_index = item.get('index')
                
                if image_url:
                    save_folder = os.path.join(OUTPUT_DIR, str(set_id))
                    save_path = os.path.join(save_folder, f"{item_index}.jpg")
                    
                    download_tasks.append({
                        'url': image_url,
                        'folder': save_folder,
                        'save_path': save_path
                    })
        print(f" -> Processed {file_path}")
    except FileNotFoundError:
        print(f" -> WARNING: Could not find {file_path}. Skipping.")

print(f"\nTotal unique images queued for download: {len(download_tasks)}")

# --- 3. THE DOWNLOAD ENGINE ---
def fetch_image(task):
    """Downloads a single image if it doesn't already exist."""
    if os.path.exists(task['save_path']):
        return "Exists"
        
    try:
        os.makedirs(task['folder'], exist_ok=True)
        # 5-second timeout so the script doesn't freeze on broken links
        response = requests.get(task['url'], stream=True, timeout=5)
        
        if response.status_code == 200:
            with open(task['save_path'], 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return "Success"
        else:
            return f"Failed"
    except Exception:
        return "Error"

# --- 4. EXECUTE IN PARALLEL ---
print("Starting high-speed multi-threaded download...")
# 15 workers means your CPU is downloading 15 images at the exact same time
with ThreadPoolExecutor(max_workers=15) as executor:
    results = list(tqdm(executor.map(fetch_image, download_tasks), total=len(download_tasks)))

print("\n--- Download Complete ---")