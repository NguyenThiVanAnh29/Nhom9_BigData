import torch
import numpy as np
from PIL import Image, ImageOps
import math
import sys
from omegaconf import OmegaConf

import cv2
import matplotlib.pyplot as plt
cmap = plt.colormaps["gray"]

from diffusers import EulerAncestralDiscreteScheduler

from pipelines.ensemble_pipeline import (
    StableDiffusionInstructPix2PixEnsemblePipeline
)
from pipelines.ensemble_sd3_pipeline import (
    StableDiffusion3InstructPix2PixEnsemblePipeline
)
from pipelines.instructdiffusion_pipeline import (
    InstructDiffusionEnsemblePipeline
)

# ======================================================
# Model paths
# ======================================================
MODEL_PATH = {
    "instructpix2pix": "timbrooks/instruct-pix2pix",
    "magicbrush": "vinesmsuic/magicbrush-jul7",
    "ultraedit": "BleachNick/SD3_UltraEdit_freeform",
    "instructdiffusion": "./checkpoints/v1-5-pruned-emaonly-adaption-task.ckpt",
    "mgie": "./checkpoints/mgie_7b/unet.pt",
}

# ======================================================
# Load model – CPU ONLY
# ======================================================
def load_model(args):
    device = "cpu"
    dtype = torch.float32   # ⚠️ CPU BẮT BUỘC float32

    print(f"[INFO] Loading model: {args.model} on CPU")

    if args.model == "instructpix2pix":
        pipe = StableDiffusionInstructPix2PixEnsemblePipeline.from_pretrained(
            MODEL_PATH["instructpix2pix"],
            torch_dtype=dtype,
            safety_checker=None
        )
        pipe = pipe.to(device)

    elif args.model == "magicbrush":
        pipe = StableDiffusionInstructPix2PixEnsemblePipeline.from_pretrained(
            MODEL_PATH["magicbrush"],
            torch_dtype=dtype,
            safety_checker=None
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        pipe = pipe.to(device)

    elif args.model == "ultraedit":
        pipe = StableDiffusion3InstructPix2PixEnsemblePipeline.from_pretrained(
            MODEL_PATH["ultraedit"],
            torch_dtype=dtype
        )
        pipe = pipe.to(device)

    elif args.model == "instructdiffusion":
        sys.path.append("./InstructDiffusion/stable_diffusion")
        from InstructDiffusion.stable_diffusion.ldm.util import instantiate_from_config

        config = OmegaConf.load(
            "./InstructDiffusion/configs/instruct_diffusion.yaml"
        )
        model = instantiate_from_config(config.model)

        state = torch.load(
            MODEL_PATH["instructdiffusion"],
            map_location="cpu"
        )["state_dict"]

        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()

        pipe = InstructDiffusionEnsemblePipeline(model)

    elif args.model == "mgie":
        pipe = StableDiffusionInstructPix2PixEnsemblePipeline.from_pretrained(
            MODEL_PATH["instructpix2pix"],
            torch_dtype=dtype,
            safety_checker=None
        )
        pipe = pipe.to(device)

        pipe.unet.load_state_dict(
            torch.load(MODEL_PATH["mgie"], map_location="cpu")
        )

        from mgie_module import MGIE_module
        mgie = MGIE_module(ckpt_dir="./checkpoints")
        pipe.generate_prompt = mgie.generate_prompt

    else:
        raise ValueError(f"Model {args.model} not supported")

    print("[INFO] Model loaded successfully")
    return pipe


# ======================================================
# Resize image
# ======================================================
def image_resize(image, resolution=512):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    w, h = image.size
    scale = resolution / max(w, h)

    new_w = int((w * scale) // 64) * 64
    new_h = int((h * scale) // 64) * 64

    new_w = max(new_w, 64)
    new_h = max(new_h, 64)

    image = ImageOps.fit(
        image,
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )
    return image


# ======================================================
# RLE decode
# ======================================================
def rle_decode(rle, shape=(512, 512)):
    try:
        s = list(map(int, rle))
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for i in range(0, len(s), 2):
            start = s[i]
            length = s[i + 1]
            mask[start:start + length] = 1
        return mask.reshape(shape)
    except Exception:
        return np.zeros(shape, dtype=np.uint8)


# ======================================================
# Heatmap visualization
# ======================================================
def heatmap_visualization(heatmap, size=(512, 512)):
    if not isinstance(heatmap, np.ndarray):
        heatmap = heatmap.cpu().detach().numpy()

    heatmap = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, size, cv2.INTER_CUBIC)
    return Image.fromarray(heatmap)
