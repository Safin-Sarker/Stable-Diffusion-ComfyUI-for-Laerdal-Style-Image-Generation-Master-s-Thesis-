import os
import json
import argparse
from datetime import datetime
import torch
from diffusers import StableDiffusionXLPipeline

# === Parse command-line arguments ===
parser = argparse.ArgumentParser(description="SDXL + LoRA inference with image and metadata saving")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
parser.add_argument("--base_model", type=str, required=True, help="Pretrained SDXL base model (e.g., stabilityai/stable-diffusion-xl-base-1.0)")
parser.add_argument("--lora_path", type=str, required=True, help="Path to folder containing LoRA .safetensors file")
parser.add_argument("--output_dir", type=str, required=True, help="Where to save the output image and metadata")
args = parser.parse_args()

# === Create output directory ===
os.makedirs(args.output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_filename = f"image_{timestamp}.png"
metadata_filename = f"image_{timestamp}.json"
image_path = os.path.join(args.output_dir, image_filename)
metadata_path = os.path.join(args.output_dir, metadata_filename)

print("🚀 Loading base SDXL model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    args.base_model,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

print("🔗 Loading LoRA weights...")
assert os.path.exists(os.path.join(args.lora_path, "SDXL_LoRa.safetensors")), "LoRA file not found!"
pipe.load_lora_weights(
    args.lora_path,
    weight_name="SDXL_LoRa.safetensors",  # Make sure the filename is correct
    adapter_name="laerdal-lora",
    local_files_only=True
)


print("🧬 Fusing LoRA...")
pipe.fuse_lora()

print(f"🎨 Generating image for prompt: {args.prompt}")
image = pipe(prompt=args.prompt).images[0]

print(f"💾 Saving image to: {image_path}")
image.save(image_path)

metadata = {
    "prompt": args.prompt,
    "base_model": args.base_model,
    "lora_path": args.lora_path,
    "lora_weights": "SDXL_LoRa.safetensors",
    "output_image": image_path,
    "timestamp": timestamp
}

print(f"📝 Saving metadata to: {metadata_path}")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ Done!")
