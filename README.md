# 🎓 Stable Diffusion & ComfyUI for Laerdal-Style Image Generation (Master's Thesis)

This repository contains the work from our master's thesis project:  
**"Stable Diffusion and Image Generation to Assist Designers"**  
in collaboration with **Laerdal Medical**.


## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [LoRA Training Process](#lora-training-process)
- [Guidelines Based on Our Experiments](#guidelines-based-on-our-experiments)
- [Inference: Generate Images with LoRA](#inference-generate-images-with-lora)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)


---

##  Project Overview

Designers often invest a significant amount of time creating illustrations manually. This project aims to reduce that effort by using Stable Diffusion and LoRA (Low-Rank Adaptation) models to automate image generation in the Laerdal Medical style.

By integrating these trained LoRA models into ComfyUI, designers can generate consistent, high-quality images by simply writing prompts—no coding or design tools needed.

---

## Key Features

- Trained LoRA models to replicate Laerdal-style illustrations  
- Support for both Stable Diffusion v1.5 and SDXL  
- LoRA training using both Kohya_ss GUI and Python scripts  
- Inference support through ComfyUI and Python CLI  
- Designed to speed up the design process for non-technical users

---

## Repository Structure

```
.
├── Dataset
│   ├── SD_v1.5
│   │   └── 1_trainingimages
│   └── SDXL_1.0
│       └── 1_Images
│
├── laerdalstyle_lora
│   ├── kohya_lora
│   │   ├──inference
│   │   │    ├── inference_Output
│   │   │    ├── SDv.15_inference.py
│   │   │    ├── Sample_Argument.txt
│   │   ├── Laerstyle_LoRa_With_Kohya.safetensors 
│   │   └── lora_config.json
│   └── python_scripts
│       ├── SD_v1.5
│       │   ├── inference
│       │       ├─ inference Output
│       │       ├─ SD_v_1.5_LoRa_inference.py
│       │       ├─ Sample_argument_for _Inference.txt
│       │   ├── LoRa_Output
│       │   │   └── SD_V_1.5 LoRa.safetensors
│       │   ├── LoRA_train.py
│       │   ├── Ppoetry.lock
│       │   └── pyproject.toml
│       └── SDXL_v1.0
│           ├── inference
│           │    ├──inference_Output
│           │    ├──inference argument.txt
│           │    └── SDXL_LoRa_inference.py
│           ├── SDXL_LoRa_Output 
│           │    └── SDXL_LoRa.safetensors
│           │
│           ├── xformers
│           ├── Sample_Argument_for_training.txt
│           ├── poetry.lock
│           ├── train_SDXL_LoRa.py
│           └── pyproject.toml
│
├── presentation
│   ├── First
│   ├── 2nd
│   └── Final
│
├── Workflows
│   ├── SD1.5
│   │   ├── Image to Image
│   │   │   └── Image to Images(SD1.5).json
│   │   └── Text to Image
│   │        └── Text to Image(sD1.5).json
│   ├── SDXL
│   │   ├── Image to Image
│   │   │   └── Image to Images(SDXL).json
│   │   └── Text to Image
│   │        └── Text to Image(SDXL).json

```

----

## LoRA Training Process

We trained LoRA models on two diffusion bases:

- Stable Diffusion v1.5
- Stable Diffusion XL (SDXL)

### SD v1.5 Training

**Method 1: Kohya_ss GUI**

> 📌 [Kohya_ss GUI installation guide](https://github.com/bmaltais/kohya_ss#installation)

1. Prepare your dataset (images and `.txt` captions).
2. Open the Kohya_ss GUI.
3. Select the base model: `runwayml/stable-diffusion-v1-5`.
4. Use the training parameters defined in [`Lora_config.json`](https://github.com/Laerdal-Medical/dna-ds-master2025-stable-diffusion/blob/main/LaerdalStyle_Lora/Kohya_Lora/Lora_config.json):
5. Start training — the output will be saved as a `.safetensors` file.


### SD v1.5 Training

**Method 2: Python Script**

1. Clone this repository and navigate to the training script directory:

   ```bash
   git clone https://github.com/Laerdal-Medical/dna-ds-master2025-stable-diffusion.git
   cd dna-ds-master2025-stable-diffusion/LaerdalStyle_Lora/Python_scripts/SD_v1.5
2. nstall dependencies using Poetry:
   ```bash
   poetry install

3. Run the training script with the following arguments:
   ```bash

   python Lora_train.py \
    --pretrained_model="runwayml/stable-diffusion-v1-5" \
    --train_data_dir="./dataset" \
    --output_dir="./lora_models/sd15_lora_python" \
    --resolution=512 \
    --train_batch_size=1 \
    --learning_rate=1e-4 \
    --num_train_epochs=10 \
    --checkpointing_steps=500 \
    --lora_rank=4 \
    --lora_alpha=16


### SDXL Training (Python Only)

We also support LoRA training on **Stable Diffusion XL (SDXL 1.0)** using a custom Python script.

> 🔗 **Script location**: [`train_SDXL_LoRa.py`](https://github.com/Laerdal-Medical/dna-ds-master2025-stable-diffusion/blob/main/LaerdalStyle_Lora/Python_scripts/SDXL_v1.0/train_SDXL_LoRa.py)

#### ✅ How to Run

1. Navigate to the SDXL training script directory:

   ```bash
   cd LaerdalStyle_Lora/Python_scripts/SDXL_v1.0

2. Install dependencies using Poetry:
   ```bash
   poetry install

3. Start training:
   ```bash
    poetry run python train_SDXL_LoRa.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir="/mnt/c/Users/Laerdal/Desktop/dna-ds-master2025-stable-diffusion/Dataset/SDXL-v-1.0/1_Images" \
    --output_dir="/mnt/c/Users/Laerdal/Desktop/dna-ds-master2025-stable-diffusion/LoRA_output" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-5 \
    --mixed_precision="fp16" \
    --train_text_encoder \
    --gradient_checkpointing \
    --checkpointing_steps=100 \
    --checkpoints_total_limit=7 \
    --max_train_steps=500 \
    --seed=42

------

##  Guidelines Based on Our Experiments

Through extensive testing and visual evaluation by design professionals, we compared multiple LoRA models trained using different methods and base models. Here's what we found:

### ✅ Best Results

#### 🥇 Kohya_ss + SD v1.5 LoRA
Our best-performing model was trained using the **Kohya_ss GUI** on **Stable Diffusion v1.5**.  
It produced the most visually consistent and high-quality Laerdal-style illustrations with minimal training time and effort.  
**Recommendation:** Use this for production-level image generation.

#### 🥈 SDXL LoRA (Python Script)
The LoRA trained on **Stable Diffusion XL 1.0** using our custom Python script also delivered good results, especially for higher-resolution outputs.  
However, it required more GPU resources and was slightly less consistent in style.



###  Recommendation

If you're aiming for optimal visual quality and simplicity, we recommend using the **Kohya-trained SD v1.5 LoRA** model with **ComfyUI** for inference.

------


## Inference: Generate Images with LoRA

You can generate images using the trained LoRA models through either Python scripts or ComfyUI.

### Method 1: Python CLI

#### 🔹 SD v1.5 Inference (via Kohya or Python script)

1. Use the inference script located at:

   ```bash
   LaerdalStyle_Lora/Kohya_Lora/Inference/SD_v_1.5_inference.py
2. Run the script using the following command:

   ```bash
   python SDv.15_inference.py \
    --prompt "." \
    --negative_prompt "" \
    --base_model runwayml/stable-diffusion-v1-5 \
    --lora_path "LaerdalStyle_Lora/Kohya_Lora" \
    --lora_weight LaerderStyle_LoRa_With_Kohya.safetensors \
    --output_dir "LaerdalStyle_Lora/Kohya_Lora/Inference/Output" \
    --width 512 \
    --height 768 \
    --guidance_scale 9

3. This will generate a Laerdal-style image and save it under the specified output directory.

### Method 2: ComfyUI

You can also use the trained LoRA models in **ComfyUI** for a visual, drag-and-drop image generation experience.

#### Step-by-Step Instructions:

1. **Place your LoRA model**  
   Copy your `.safetensors` LoRA file into the following directory:

   ```bash
   ComfyUI/models/loras/

2. Load the workflow
   Import the provided workflow into ComfyUI:

   ```bash
   comfyui_workflows/laerdal_lora_workflow.json

3. Run ComfyUI
   Open ComfyUI in your browser or desktop interface.

4. Generate images
   - Enter your prompt in the designated input node.

   - Click Generate to produce the Laerdal-style illustration using the loaded LoRA model.


----
  

## Acknowledgments

- **Laerdal Medical** – for design guidelines and valuable collaboration  
- **Kohya_ss** – for providing the GUI and tools for LoRA training  
- **Hugging Face Diffusers** – for the powerful diffusion and training libraries  
- **ComfyUI** – for enabling no-code, visual workflows for Stable Diffusion



----

## Contributing

Pull requests and suggestions are welcome!  
If you'd like to contribute improvements, new workflows, or training support for other styles:

- Fork the repository
- Make your changes
- Open a pull request (PR)

We appreciate all contributions that help make this project better.




   



   
   







