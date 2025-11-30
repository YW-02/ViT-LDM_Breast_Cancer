from diffusers import StableDiffusionPipeline

base_model = "/Users/rayno/Workspace/ML/Models/Diffusion/sd_v1_5"
pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to("cuda")

# load LoRA
pipe.unet.load_attn_procs("/Users/rayno/Workspace/ML/BreastCancer/Train/Final/lora_sd15_breast_240x240_rank4.safetensors")

prompt = "your breast ultrasound style description..."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("sample_with_lora.png")
