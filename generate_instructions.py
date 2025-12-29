import random
import json
from pathlib import Path

# ==============================
# CONFIG
# ==============================
NUM_INSTRUCTIONS = 500   # 200–500 là đủ cho 10GB ảnh
OUTPUT_TXT = "instructions.txt"
OUTPUT_JSON = "instructions.json"

random.seed(42)

# ==============================
# INSTRUCTION POOLS (CHUẨN IP2P)
# ==============================

LIGHTING_COLOR = [
    "make the image brighter",
    "make the image darker",
    "increase contrast",
    "reduce contrast",
    "add warm lighting",
    "add cold lighting",
    "enhance color balance",
    "reduce saturation",
    "increase saturation",
    "improve overall lighting"
]

STYLE = [
    "make it realistic",
    "make it cinematic style",
    "convert to watercolor painting",
    "convert to oil painting",
    "convert to pencil sketch",
    "make it look professional",
    "make it artistic",
    "apply soft visual style",
    "apply dramatic style"
]

DETAIL_QUALITY = [
    "enhance image quality",
    "sharpen the main subject",
    "improve visual clarity",
    "reduce noise",
    "add fine details",
    "make the image cleaner",
    "improve texture appearance"
]

SCENE_ATMOSPHERE = [
    "add fog",
    "add light mist",
    "make it look like sunset",
    "make it look like sunrise",
    "make it look like nighttime",
    "add dramatic atmosphere",
    "add soft background blur",
    "enhance background depth"
]

# ==============================
# GENERATOR
# ==============================

def generate_instruction():
    parts = [
        random.choice(LIGHTING_COLOR),
        random.choice(STYLE),
        random.choice(DETAIL_QUALITY),
        random.choice(SCENE_ATMOSPHERE)
    ]
    return ", ".join(parts)

# ==============================
# MAIN
# ==============================

def main():
    instructions = set()

    while len(instructions) < NUM_INSTRUCTIONS:
        ins = generate_instruction()
        instructions.add(ins)

    instructions = sorted(list(instructions))

    # Save TXT
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for ins in instructions:
            f.write(ins + "\n")

    # Save JSON (optional, useful for metadata)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(instructions, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(instructions)} instructions")
    print(f"Saved to {OUTPUT_TXT} and {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
