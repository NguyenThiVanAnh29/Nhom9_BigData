import os
import json
from argparse import ArgumentParser
import torch
from PIL import Image
import random

from utils import load_model, image_resize


def generate(args, pipe, input_image, instruction, seed):
    res = {}

    generator = torch.Generator(device="cpu").manual_seed(seed)

    if args.model in ["instructpix2pix", "magicbrush"]:
        edited = pipe(
            image=input_image,
            prompt=[instruction],
            generator=generator,
            guidance_scale=args.text_cfg_scale,
            image_guidance_scale=args.image_cfg_scale,
            num_inference_steps=args.inference_steps,
        )
        res["output"] = edited.images[0]

    elif args.model == "instructdiffusion":
        res = pipe(
            args=args,
            prompt=instruction,
            input_image=input_image
        )

    elif args.model == "mgie":
        prompt_embeds, negative_prompt_embeds = pipe.generate_prompt(
            input_image, instruction
        )
        edited = pipe(
            image=input_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
        )
        res["output"] = edited.images[0]

    else:
        raise ValueError("Model not supported on CPU")

    return res


def main():
    parser = ArgumentParser()

    parser.add_argument("--run_type", default="run_single_image")
    parser.add_argument("--model", default="instructpix2pix")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--output_name", default="test")

    parser.add_argument("--input_path", required=True)
    parser.add_argument("--instruction", required=True)

    # ⚠️ CPU SAFE CONFIG
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--inference_steps", default=25, type=int)
    parser.add_argument("--text_cfg_scale", default=7.5, type=float)
    parser.add_argument("--image_cfg_scale", default=1.5, type=float)
    parser.add_argument("--seed", default=1, type=int)

    args = parser.parse_args()

    print(f"[INFO] Loading model {args.model} on CPU")
    pipe = load_model(args)

    out_dir = os.path.join(args.output_dir, f"{args.model}-{args.output_name}")
    os.makedirs(out_dir, exist_ok=True)

    image = Image.open(args.input_path).convert("RGB")
    image = image_resize(image, resolution=args.resolution)

    print("[INFO] Running inference...")
    res = generate(args, pipe, image, args.instruction, args.seed)

    image.save(os.path.join(out_dir, "input.png"))
    res["output"].save(os.path.join(out_dir, "output.png"))

    print("[DONE] Saved results to", out_dir)


if __name__ == "__main__":
    main()
