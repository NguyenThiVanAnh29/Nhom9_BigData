import json
import random
from pathlib import Path

# ==============================
# CONFIG (PROJECT-RELATIVE)
# ==============================

# Thư mục gốc project (nơi đặt generate_metadata.py)
PROJECT_ROOT = Path(__file__).resolve().parent

# Cấu trúc dataset chuẩn cho ELECT
RAW_IMAGE_DIR = PROJECT_ROOT / "dataset" / "raw_input"
INSTRUCTION_FILE = PROJECT_ROOT / "instructions.txt"
OUTPUT_METADATA = PROJECT_ROOT / "dataset" / "metadata.json"

# Seed config cho ELECT
SEEDS_PER_IMAGE = 5          # số seed cho mỗi ảnh
SEED_MIN = 0
SEED_MAX = 100000

random.seed(42)

# ==============================
# LOAD INSTRUCTIONS
# ==============================

if not INSTRUCTION_FILE.exists():
    raise FileNotFoundError(f"Instruction file not found: {INSTRUCTION_FILE}")

with open(INSTRUCTION_FILE, "r", encoding="utf-8") as f:
    instructions = [line.strip() for line in f if line.strip()]

if len(instructions) == 0:
    raise ValueError("Instruction list is empty!")

print(f"Loaded {len(instructions)} instructions")

# ==============================
# LOAD RAW IMAGES (RECURSIVE)
# ==============================

if not RAW_IMAGE_DIR.exists():
    raise FileNotFoundError(f"Raw image directory not found: {RAW_IMAGE_DIR}")

image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

image_paths = [
    p for p in RAW_IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in image_extensions
]

if len(image_paths) == 0:
    raise ValueError("No images found in raw_input!")

print(f"Found {len(image_paths)} raw images")

# ==============================
# GENERATE METADATA
# ==============================

metadata = []

for img_path in image_paths:
    instruction = random.choice(instructions)

    seeds = random.sample(
        range(SEED_MIN, SEED_MAX),
        SEEDS_PER_IMAGE
    )

    # đường dẫn tương đối (giữ cấu trúc folder con)
    relative_path = img_path.relative_to(RAW_IMAGE_DIR)

    item = {
        "raw_image": f"raw_input/{relative_path.as_posix()}",
        "instruction": instruction,
        "seeds": seeds
    }

    metadata.append(item)

# ==============================
# SAVE METADATA
# ==============================

OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Metadata generated successfully")
print(f"Total images       : {len(metadata)}")
print(f"Seeds per image    : {SEEDS_PER_IMAGE}")
print(f"Metadata saved to  : {OUTPUT_METADATA}")
