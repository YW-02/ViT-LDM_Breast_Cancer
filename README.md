## ViT-LDM_Breast_Cancer
Source core of Synthesizing Breast Cancer Ultrasound Images from Healthy Samples Using Latent Diffusion Models, including model training, data pre-processing and hyper-pramaters


Finetuning: Dataset & Prompt CSV

project_root/
├─ data/
│  ├─ images/
│  │  ├─ img0001.png
│  │  ├─ img0002.png
│  │  ├─ img0003.png
│  │  └─ ...
│  └─ prompts.csv
├─ checkpoints/
└─ final/

Image requirements
	•	All images are single-channel grayscale PNGs.
	•	Size is 240 × 240 pixels (or will be resized/cropped to this).
	•	Filenames must match the names used in prompts.csv exactly (case-sensitive).

prompts.csv format
	•	CSV with two columns:
	1.	image filename (e.g., img0001.png)
	2.	text prompt describing that image
	•	Either with header:

filename,prompt
img0001.png,"Ultrasound of breast with benign lesion"
img0002.png,"Ultrasound of breast with suspicious mass"

	•	Or without header, with the same column order:

img0001.png,"Ultrasound of breast with benign lesion"
img0002.png,"Ultrasound of breast with suspicious mass"

	•	Every training image must have exactly one row in prompts.csv.

⸻

Inference: Required Files & Usage Notes

project_root/
├─ base_model/          # SD v1.5 in diffusers format (unet, vae, text_encoder, tokenizer, etc.)
├─ lora/
│  └─ breast_240x240.safetensors
└─ scripts/
   └─ inference.py

	•	Base model: Stable Diffusion v1.5 in diffusers format (from HuggingFace or locally converted).
	•	LoRA file: a .safetensors file containing the fine-tuned LoRA weights.
	•	During inference:
	•	Load the base SD v1.5 model.
	•	Load the LoRA weights into the UNet (e.g., unet.load_attn_procs("lora/breast_240x240.safetensors")).
	•	Use any suitable text prompt to generate images in the trained domain; outputs can be resized or processed as needed around the 240×240 target resolution.


