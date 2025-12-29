import os
import requests
from ddgs import DDGS
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

# ================= CONFIG =================
SAVE_DIR = "dataset/raw_input"
TARGET_SIZE_GB = 30
TARGET_SIZE_BYTES = TARGET_SIZE_GB * 1024 * 1024 * 1024

# Thêm nhiều category và từ khóa khác nhau
KEYWORDS = {
    "nature": [
        "forest",
        "mountain",
        "river",
        "lake reflection",
        "desert landscape",
        "sunset",
        "waterfall",
        "snow mountain",
        "beach",
        "volcano"
    ],
    "animals": [
        "dog",
        "cat",
        "lion",
        "tiger",
        "elephant",
        "bird flying",
        "horse",
        "bear",
        "giraffe",
        "wolf"
    ],
    "food": [
        "pizza",
        "burger",
        "sushi",
        "dessert",
        "fruits",
        "cake",
        "coffee",
        "bread",
        "ice cream",
        "salad"
    ],
    "vehicles": [
        "car",
        "motorbike",
        "bus",
        "train",
        "airplane",
        "boat",
        "truck",
        "bicycle",
        "scooter",
        "helicopter"
    ]
}

MAX_IMAGES_PER_KEYWORD = 5000
TIMEOUT = 10
SLEEP_TIME = 0.1  # giảm thời gian sleep cho nhanh
MAX_WORKERS = 10  # số thread tải song song
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


def download_worker(r, category, query, idx):
    url = r.get("image")
    if not url:
        return
    cat_dir = os.path.join(SAVE_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)
    filename = f"{category}_{query.replace(' ','_')}_{idx}.jpg"
    save_path = os.path.join(cat_dir, filename)
    if os.path.exists(save_path):
        return
    if download_image(url, save_path):
        time.sleep(SLEEP_TIME)


def crawl():
    os.makedirs(SAVE_DIR, exist_ok=True)

    with DDGS() as ddgs:
        for category, queries in KEYWORDS.items():
            cat_dir = os.path.join(SAVE_DIR, category)
            os.makedirs(cat_dir, exist_ok=True)

            for query in queries:
                print(f"\nCrawling keyword: {query}")
                results = ddgs.images(query, max_results=MAX_IMAGES_PER_KEYWORD)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    for idx, r in enumerate(tqdm(results)):
                        current_size = get_folder_size(SAVE_DIR)

                        if current_size >= TARGET_SIZE_BYTES:
                            print("\nReached 30GB. STOP.")
                            return

                        executor.submit(download_worker, r, category, query, idx)

    print("Crawling finished.")


if __name__ == "__main__":
    crawl()
