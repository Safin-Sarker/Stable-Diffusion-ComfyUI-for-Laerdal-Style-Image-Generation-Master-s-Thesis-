import argparse
import math
import os
import random
import shutil
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from safetensors.torch import save_file as save_safetensors

from torch.optim import AdamW
optimizer_class = AdamW

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,  subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=7)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)



    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args




def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )


        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(args):

  accelerator_project_config = ProjectConfiguration(
    project_dir=args.output_dir,
    logging_dir=os.path.join(args.output_dir, "logs")
)
  
  kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
  accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
  

  if args.seed is not None:
      set_seed(args.seed)


  if accelerator.is_main_process:
      if args.output_dir is not None:
          os.makedirs(args.output_dir, exist_ok=True)
  

  tokenizer_one = AutoTokenizer.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="tokenizer",
      use_fast=False,
  )
  tokenizer_two = AutoTokenizer.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="tokenizer_2",
      use_fast=False,
  )

  text_encoder_cls_one = import_model_class_from_model_name_or_path(
      args.pretrained_model_name_or_path
  )
  text_encoder_cls_two = import_model_class_from_model_name_or_path(
      args.pretrained_model_name_or_path, subfolder="text_encoder_2"
  )

  noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
  text_encoder_one = text_encoder_cls_one.from_pretrained(
      args.pretrained_model_name_or_path, subfolder="text_encoder"
  )
  text_encoder_two = text_encoder_cls_two.from_pretrained(
      args.pretrained_model_name_or_path, subfolder="text_encoder_2"
  )

  vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="vae",
  )

  unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", 
    )


  vae.requires_grad_(False)
  text_encoder_one.requires_grad_(False)
  text_encoder_two.requires_grad_(False)
  unet.requires_grad_(False)


  weight_dtype = torch.float16


  unet.to(accelerator.device, dtype=weight_dtype)

  vae.to(accelerator.device, dtype=torch.float32)


  unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

  unet.add_adapter(unet_lora_config)


  if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

  def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

  def save_model_hook(models, weights, output_dir):
      if accelerator.is_main_process:
          unet_lora_layers_to_save = None
          text_encoder_one_lora_layers_to_save = None
          text_encoder_two_lora_layers_to_save = None

          for model in models:
              if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                  unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
              elif isinstance(unwrap_model(model), type(unwrap_model(text_encoder_one))):
                  text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                      get_peft_model_state_dict(model)
                  )
              elif isinstance(unwrap_model(model), type(unwrap_model(text_encoder_two))):
                  text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                      get_peft_model_state_dict(model)
                  )
              else:
                  raise ValueError(f"unexpected save model: {model.__class__}")

              if weights:
                  weights.pop()

          state_dict = {}
          if unet_lora_layers_to_save:
              state_dict.update({f"unet.{k}": v for k, v in unet_lora_layers_to_save.items()})
          if text_encoder_one_lora_layers_to_save:
              state_dict.update({f"text_encoder.{k}": v for k, v in text_encoder_one_lora_layers_to_save.items()})
          if text_encoder_two_lora_layers_to_save:
              state_dict.update({f"text_encoder_2.{k}": v for k, v in text_encoder_two_lora_layers_to_save.items()})

          save_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
          save_safetensors(state_dict, save_path)


  def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:

            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
            )
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            cast_training_params(models, dtype=torch.float32)

  accelerator.register_save_state_pre_hook(save_model_hook)
  accelerator.register_load_state_pre_hook(load_model_hook)

  if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

  if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)

   # Optimizer creation
  params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
  if args.train_text_encoder:
      params_to_optimize = (
          params_to_optimize
          + list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
          + list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
      )
  optimizer = optimizer_class(
    params_to_optimize,
    lr=args.learning_rate,
  )  

  print("DEBUG: args.train_data_dir =", args.train_data_dir)

  dataset = load_dataset(
      "imagefolder",
      data_dir=args.train_data_dir
      )


  column_names = dataset["train"].column_names
  image_column = column_names[0]
  caption_column = column_names[1]

  def tokenize_captions(examples, is_train=True):
          captions = []
          for caption in examples[caption_column]:
              if isinstance(caption, str):
                  captions.append(caption)
              elif isinstance(caption, (list, np.ndarray)):
                  captions.append(random.choice(caption) if is_train else caption[0])
              else:
                  raise ValueError(
                      f"Caption column `{caption_column}` should contain either strings or lists of strings."
                  )
          tokens_one = tokenize_prompt(tokenizer_one, captions)
          tokens_two = tokenize_prompt(tokenizer_two, captions)
          return tokens_one, tokens_two

  interpolation = transforms.InterpolationMode.BILINEAR


  train_resize = transforms.Resize(args.resolution, interpolation=interpolation)  
  train_crop = transforms.CenterCrop(args.resolution)
  train_flip = transforms.RandomHorizontalFlip(p=1.0)
  train_transforms = transforms.Compose(
      [
          transforms.ToTensor(),
          transforms.Normalize([0.5], [0.5]),
      ]
  )
  def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)

            y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
            x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
            image = train_crop(image)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)

            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        tokens_one, tokens_two = tokenize_captions(examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
        return examples

  with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train, output_all_columns=True)
  
  def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
        input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
        result = {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
        }
        return result

  train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=4,
    )

  overrode_max_train_steps = False
  num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
  if args.max_train_steps is None:
      args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
      overrode_max_train_steps = True

  lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
  )

  if args.train_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
  else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
      )

  num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
  if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
  args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch) 

  if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

  total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
  
  global_step = 0
  first_epoch = 0

  progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
  )

  for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                pixel_values = batch["pixel_values"]
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                model_input = model_input.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)

                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids

                add_time_ids = torch.cat(
                    [compute_time_ids(s, c) for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])]
                )

                # Predict the noise residual
                unet_added_conditions = {"time_ids": add_time_ids}
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[text_encoder_one, text_encoder_two],
                    tokenizers=None,
                    prompt=None,
                    text_input_ids_list=[batch["input_ids_one"], batch["input_ids_two"]],
                )
                unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=unet_added_conditions,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
 
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

               
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
  
  accelerator.wait_for_everyone()
  if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_two = unwrap_model(text_encoder_two)

            text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_one))
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_two))
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        state_dict = {}
        if unet_lora_state_dict:
            state_dict.update({f"unet.{k}": v for k, v in unet_lora_state_dict.items()})
        if text_encoder_lora_layers:
            state_dict.update({f"text_encoder.{k}": v for k, v in text_encoder_lora_layers.items()})
        if text_encoder_2_lora_layers:
            state_dict.update({f"text_encoder_2.{k}": v for k, v in text_encoder_2_lora_layers.items()})

        save_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        save_safetensors(state_dict, save_path)
        logger.info(f"Saved final safetensors LoRA weights to {save_path}")

        del unet
        del text_encoder_one
        del text_encoder_two
        del text_encoder_lora_layers
        del text_encoder_2_lora_layers
        torch.cuda.empty_cache()

        if args.mixed_precision == "fp16":
          vae.to(weight_dtype)
        
        accelerator.end_training()




if __name__ == "__main__":
  args = parse_args()
  main(args)



  
  










