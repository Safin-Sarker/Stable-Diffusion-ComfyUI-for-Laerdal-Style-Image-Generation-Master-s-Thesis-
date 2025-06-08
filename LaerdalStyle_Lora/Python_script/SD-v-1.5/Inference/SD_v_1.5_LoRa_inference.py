import os
import json
import argparse
from datetime import datetime
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file 

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="SD v1.5 + Kohya LoRA inference")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt to suppress undesired features")
parser.add_argument("--base_model", type=str, required=True, help="Base model (e.g., runwayml/stable-diffusion-v1-5)")
parser.add_argument("--lora_path", type=str, required=True, help="Path to folder with LoRA .safetensors")
parser.add_argument("--lora_weight", type=str, required=True, help="LoRA weights filename (.safetensors)")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output")
parser.add_argument("--steps", type=int, default=30, help="Inference steps")
parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG guidance scale")
parser.add_argument("--width", type=int, default=512, help="Image width")
parser.add_argument("--height", type=int, default=512, help="Image height")
args = parser.parse_args()

# === OUTPUT PATHS ===
os.makedirs(args.output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_path = os.path.join(args.output_dir, f"image_{timestamp}.png")
metadata_path = os.path.join(args.output_dir, f"image_{timestamp}.json")

# === LOAD PIPELINE ===
print("Loading base SD v1.5 model...")
pipe = StableDiffusionPipeline.from_pretrained(
    args.base_model,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")



lora_file_path = os.path.join(args.lora_path, args.lora_weight)
state_dict = load_file(lora_file_path)

# Apply LoRA weights to the UNet
print(" Applying LoRA to UNet...")
compatible = {k: v for k, v in state_dict.items() if k in pipe.unet.state_dict()}
missing = [k for k in state_dict if k not in pipe.unet.state_dict()]
if missing:
    print(f"⚠️ Warning: {len(missing)} keys in LoRA not found in UNet. This is normal for text encoder LoRA.")
pipe.unet.load_state_dict(compatible, strict=False)

# === GENERATE ===
print(f"🎨 Generating image for prompt: {args.prompt}")
if args.negative_prompt:
    print(f" Using negative prompt: {args.negative_prompt}")

with torch.autocast("cuda"):
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height
    ).images[0]

# === SAVE OUTPUT ===
print(f"💾 Saving image to: {image_path}")
image.save(image_path)

metadata = {
    "prompt": args.prompt,
    "negative_prompt": args.negative_prompt,
    "base_model": args.base_model,
    "lora_path": args.lora_path,
    "lora_weights": args.lora_weight,
    "steps": args.steps,
    "guidance_scale": args.guidance_scale,
    "timestamp": timestamp,
    "output_image": image_path
}

print(f" Saving metadata to: {metadata_path}")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(" Done!")