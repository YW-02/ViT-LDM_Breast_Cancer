
import os
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor, AttnProcsLayers
from transformers import CLIPTokenizer, CLIPTextModel, get_cosine_schedule_with_warmup

from safetensors.torch import save_file as safe_save_file


# =========================

@dataclass
class TrainConfig:
    # --- paths ---
    base_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"  # or your local diffusers SD1.5 folder
    image_dir: str = "/Users/rayno/Workspace/ML/BreastCancer/Train/trainset07v2/image/"
    prompt_csv: str = "/Users/rayno/Workspace/ML/BreastCancer/Train/trainset07v2/prompt.csv"
    checkpoint_dir: str = "/Users/rayno/Workspace/ML/BreastCancer/Train/checkpoints/"
    final_dir: str = "/Users/rayno/Workspace/ML/BreastCancer/Train/Final/"

    # --- image / data ---
    image_size: int = 240                # 240x240 grayscale
    center_crop: bool = True
    random_flip: bool = True
    # --- LoRA hyperparams ---
    lora_rank: int = 4                   # 常用 4 / 8 / 16
    lora_alpha: int = 4
    lora_dropout: float = 0.0

    # --- training hyperparams ---
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # --- early stopping ---
    early_stop_patience: int = 5
    early_stop_min_delta: float = 1e-4

    # --- misc ---
    seed: int = 42
    num_workers: int = 4
    use_amp: bool = True
    log_interval: int = 50
    save_every_n_epochs: int = 1


# =========================

class BreastCancerDataset(Dataset):
    """
    读取单通道灰度 PNG，并配对 prompt。
    - 图像会 resize 到 240x240，转换为 3 通道，归一化到 [-1, 1]
    """
    def __init__(
        self,
        image_dir: str,
        prompt_csv: str,
        image_size: int,
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.image_size = image_size

        df = pd.read_csv(prompt_csv)
        
        self.samples = []
        for _, row in df.iterrows():
            filename = str(row.iloc[0]).strip()
            prompt = str(row.iloc[1]).strip()
            img_path = os.path.join(image_dir, filename)
            if os.path.isfile(img_path):
                self.samples.append((img_path, prompt))
            else:
                print(f"[WARN] Image not found, skip: {img_path}")

        if len(self.samples) == 0:
            raise ValueError("No valid samples found. Please check image_dir and prompt_csv.")

        transform_list = [transforms.Grayscale(num_output_channels=3)]
        if center_crop:
            transform_list.append(transforms.CenterCrop(image_size))
        transform_list.extend(
            [
                transforms.Resize((image_size, image_size)),
            ]
        )
        if random_flip:
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]
            )
        transform_list.append(transforms.ToTensor())  # -> [0,1]
        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, prompt = self.samples[idx]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)             # -> [3, H, W], 0~1

        # Stable Diffusion VAE 期望 [-1, 1]
        image = image * 2.0 - 1.0

        return {
            "pixel_values": image,
            "prompt": prompt,
            "filename": os.path.basename(img_path),
        }


# =========================


def add_lora_to_unet(unet: UNet2DConditionModel, rank: int, alpha: int, dropout: float) -> AttnProcsLayers:

    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        # self-attention (attn1) doesn't have cross_attention_dim
        if name.endswith("attn1.processor"):
            cross_attention_dim = None
        else:
            cross_attention_dim = unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks."):].split(".")[0])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks."):].split(".")[0])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            
            hidden_size = unet.config.block_out_channels[-1]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
            network_alpha=alpha,
            dropout=dropout,
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    return lora_layers


# =========================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(examples: List[Dict[str, Any]], tokenizer: CLIPTokenizer, device: torch.device):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    prompts = [e["prompt"] for e in examples]

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids

    batch = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "prompts": prompts,
        "filenames": [e["filename"] for e in examples],
    }
    return batch


def evaluate(
    dataloader: DataLoader,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    unet.eval()
    vae.eval()
    text_encoder.eval()

    total_loss = 0.0
    num_steps = 0

    autocast_context = torch.autocast("cuda", dtype=torch.float16) if use_amp and device.type == "cuda" else torch.no_grad()

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
                # Encode text
                encoder_hidden_states = text_encoder(input_ids)[0]

                # Encode images to latents
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # UNet
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction_type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            total_loss += loss.item()
            num_steps += 1

    avg_loss = total_loss / max(num_steps, 1)
    return avg_loss


def main():
    config = TrainConfig()

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.final_dir, exist_ok=True)

    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================
    
    print("Loading tokenizer, text encoder, VAE, UNet from:", config.base_model_name_or_path)
    tokenizer = CLIPTokenizer.from_pretrained(config.base_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.base_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.base_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.base_model_name_or_path, subfolder="unet")

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(config.base_model_name_or_path, subfolder="scheduler")

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # train LoRA only
    unet.requires_grad_(False)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    # gradient checkpointing
    unet.enable_gradient_checkpointing()

    # add LoRA layers
    print("Adding LoRA layers to UNet...")
    lora_layers = add_lora_to_unet(
        unet,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    print(f"LoRA layers: {sum(p.numel() for p in lora_layers.parameters())} trainable parameters.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        lora_layers.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # =========================
    
    print("Building dataset...")
    full_dataset = BreastCancerDataset(
        image_dir=config.image_dir,
        prompt_csv=config.prompt_csv,
        image_size=config.image_size,
        center_crop=config.center_crop,
        random_flip=config.random_flip,
    )

    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.seed),
    )

    print(f"Total samples: {n_total} | Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    def _collate_fn(examples):
        return collate_fn(examples, tokenizer, device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    #
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    max_train_steps = config.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")

    # =========================
    
    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("Start training...")
    for epoch in range(config.num_epochs):
        unet.train()
        vae.eval()
        text_encoder.eval()

        running_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=config.use_amp and device.type == "cuda"):
                # Encode text
                encoder_hidden_states = text_encoder(input_ids)[0]

                # Encode images to latents
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise & timestep
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction_type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = loss / config.gradient_accumulation_steps

            if config.use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp and device.type == "cuda":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(lora_layers.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(lora_layers.parameters(), config.max_grad_norm)
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            running_loss += loss.item() * config.gradient_accumulation_steps

            if (step + 1) % config.log_interval == 0:
                avg_loss = running_loss / config.log_interval
                print(
                    f"Epoch [{epoch+1}/{config.num_epochs}] "
                    f"Step [{step+1}/{len(train_dataloader)}] "
                    f"Global Step [{global_step}] "
                    f"Loss: {avg_loss:.6f}"
                )
                running_loss = 0.0

        # ---- validation ----
        val_loss = evaluate(
            val_dataloader,
            vae,
            text_encoder,
            tokenizer,
            unet,
            noise_scheduler,
            device,
            use_amp=config.use_amp,
        )
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.6f}")

        # ---- checkpoint ----
        if (epoch + 1) % config.save_every_n_epochs == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"lora_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "lora_state_dict": lora_layers.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": config.__dict__,
                },
                ckpt_path,
            )
            print(f"Checkpoint saved to: {ckpt_path}")

        # ---- early stopping ----
        if val_loss + config.early_stop_min_delta < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # save final LoRA as safetensors
            best_path = os.path.join(config.checkpoint_dir, "best_lora.safetensors")
            safe_save_file(lora_layers.state_dict(), best_path)
            print(f"New best model. Saved LoRA weights to: {best_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement} epoch(s). "
                f"(Best val loss: {best_val_loss:.6f})"
            )
            if epochs_without_improvement >= config.early_stop_patience:
                print("Early stopping triggered.")
                break

    # =========================
    
    final_lora_path = os.path.join(config.final_dir, "lora_sd15_breast_240x240_rank{}.safetensors".format(config.lora_rank))
    safe_save_file(lora_layers.state_dict(), final_lora_path)
    print(f"Training finished. Final LoRA weights saved to: {final_lora_path}")

    print("Done.")


if __name__ == "__main__":
    main()
