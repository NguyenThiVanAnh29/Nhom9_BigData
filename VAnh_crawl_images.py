import os
import requests
from ddgs import DDGS
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

# ================= CONFIG =================
SAVE_DIR = "dataset/raw_input"
TARGET_SIZE_GB = 10
TARGET_SIZE_BYTES = TARGET_SIZE_GB * 1024 * 1024 * 1024

KEYWORDS = {
    "people": [
        "portrait person",
        "human face",
        "young man portrait",
        "young woman portrait",
        "old man portrait",
        "old woman portrait",
        "asian face portrait",
        "european face portrait",
        "smiling person",
        "serious face",
        "person indoor",
        "person outdoor",
        "people candid photo",
        "profile face photo",
        "front face photo",
        "side face portrait",
        "person wearing glasses",
        "person without glasses"
    ],

    "street": [
        "street photography",
        "urban street",
        "city road",
        "city traffic",
        "busy street",
        "empty street",
        "night street",
        "street daytime",
        "crosswalk street",
        "city sidewalk",
        "urban intersection",
        "street rain",
        "street sunny",
        "street crowd",
        "street building view",
        "street market",
        "street alley"
    ]
}

MAX_IMAGES_PER_KEYWORD = 3000
TIMEOUT = 10
SLEEP_TIME = 0.5
# ==========================================


def get_folder_size(folder):
    total = 0
    for root, _, files in os.walk(folder):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def download_image(url, save_path):
    try:
        r = requests.get(url, timeout=TIMEOUT)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(save_path, "JPEG", quality=90)
        return True
    except:
        return False


def crawl():
    os.makedirs(SAVE_DIR, exist_ok=True)

    with DDGS() as ddgs:
        for category, queries in KEYWORDS.items():
            cat_dir = os.path.join(SAVE_DIR, category)
            os.makedirs(cat_dir, exist_ok=True)

            for query in queries:
                print(f"\nCrawling keyword: {query}")
                results = ddgs.images(query, max_results=MAX_IMAGES_PER_KEYWORD)

                for idx, r in enumerate(tqdm(results)):
                    current_size = get_folder_size(SAVE_DIR)

                    if current_size >= TARGET_SIZE_BYTES:
                        print("\nReached 10GB. STOP.")
                        return

                    url = r.get("image")
                    if not url:
                        continue

                    filename = f"{category}_{query.replace(' ','_')}_{idx}.jpg"
                    save_path = os.path.join(cat_dir, filename)

                    if os.path.exists(save_path):
                        continue

                    if download_image(url, save_path):
                        time.sleep(SLEEP_TIME)

    print("Crawling finished.")


if __name__ == "__main__":
    crawl()
