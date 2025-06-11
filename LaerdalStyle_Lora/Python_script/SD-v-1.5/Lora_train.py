import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.utils.import_utils import is_xformers_available


class LaerdalStyleDataset(Dataset):
    def __init__(self, data_dir, tokenizer, size=512):
        self.tokenizer = tokenizer
        self.size = size
        self.image_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(data_dir)
            for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption_path = os.path.splitext(image_path)[0] + ".txt"
        if os.path.exists(caption_path):
            with open(caption_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = "Laerdal-style minimalistic vector illustration"

        if random.random() < 0.1:
            prompt = ""

        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": encoded["input_ids"][0]
        }


def inject_lora_attn_processors(unet, rank):
    injected_layers = []
    for name, module in unet.named_modules():
        if hasattr(module, "set_attn_processor"):
            try:
                if hasattr(module, "to_q") and hasattr(module, "to_k"):
                    hidden_size = module.to_q.in_features
                    cross_attention_dim = module.to_k.in_features
                    lora = LoRAAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=rank
                    )
                    module.set_attn_processor(lora)
                    injected_layers.append(name)
            except Exception as e:
                print(f" Error injecting into {name}: {e}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet").to(device)

    inject_lora_attn_processors(unet, rank=args.lora_rank)

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    dataset = LaerdalStyleDataset(args.train_data_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    scale_factor = vae.config.scaling_factor

    global_step = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(dataloader):
            unet.train()
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * scale_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)

            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(input_ids)[0]

            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.float32):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            global_step += 1

            print(f"Epoch {epoch+1} | Step {step+1} | Loss: {loss.item():.6f}")

            if global_step % args.checkpointing_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                unet.save_attn_procs(checkpoint_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    unet.save_attn_procs(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Training complete. LoRA saved.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--lora_rank", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
