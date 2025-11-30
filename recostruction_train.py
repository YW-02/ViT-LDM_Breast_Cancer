import os
import math
import csv
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

from torchvision import transforms

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
)

from peft import LoraConfig, get_peft_model


# ==========================

@dataclass
class TrainConfig:
    # pretrained model
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-inpainting"
    # pretrained_model_name_or_path: str = "/Users/rayno/Workspace/ML/Models/Diffusion/sd_v1_5_inpaint"

    # training data
    perfect_dir: str = "/Users/rayno/Workspace/ML/BreastCancer/Train/img2img/imageRec/"
    flawed_dir: str = "/Users/rayno/Workspace/ML/BreastCancer/Train/img2img/imageFal/"
    prompt_csv: str = "/Users/rayno/Workspace/ML/BreastCancer/Train/img2img/auto-prompt.csv"

    # model output
    checkpoint_dir: str = "/Users/rayno/Workspace/ML/BreastCancer/Train/RecCheckpoint/"
    final_dir: str = "/Users/rayno/Workspace/ML/BreastCancer/Train/RecFinal/"

    
    image_size: int = 240   # 240x240
    mask_threshold: float = 0.15

    train_ratio: float = 0.8
    val_ratio: float = 0.1

    train_batch_size: int = 8
    val_batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # LoRA prameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = None

    # early stop
    early_stopping_patience: int = 5  # 若 val loss 连续 N 个 epoch 不下降则停止

    seed: int = 42
    use_amp: bool = True
    num_workers: int = 4  # DataLoader workers

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["to_q", "to_k", "to_v", "to_out.0"]


config = TrainConfig()


# ==========================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def gray_to_rgb(img: Image.Image) -> Image.Image:
    """把灰度图复制成 3 通道 RGB。"""
    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img)
    rgb_arr = np.stack([arr, arr, arr], axis=-1)  # H, W, 3
    return Image.fromarray(rgb_arr.astype(np.uint8), mode="RGB")


# ==========================


class InpaintLoraDataset(Dataset):
    """
    return：
      - perfect_image (Tensor): reconstructed [-1, 1], 3xHxW
      - flawed_image  (Tensor): with adjacent noises [-1, 1], 3xHxW
      - mask          (Tensor): ajcavent noisy region [0,1], 1xHxW
      - input_ids, attention_mask: prompt tokens
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        perfect_dir: str,
        flawed_dir: str,
        tokenizer: CLIPTokenizer,
        image_size: int = 240,
        mask_threshold: float = 0.15,
        augment: bool = False,
    ):
        self.samples = samples
        self.perfect_dir = perfect_dir
        self.flawed_dir = flawed_dir
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.mask_threshold = mask_threshold
        self.augment = augment

        # image transform：Resize + ToTensor + Normalize [-1, 1]
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1,1]
        ])

        # mask transform: [0,1] Tensor
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.samples)

    def _generate_mask_from_diff(self, perfect_img: Image.Image, flawed_img: Image.Image) -> Image.Image:
        
        p = perfect_img.convert("L")
        f = flawed_img.convert("L")
        p_arr = np.array(p, dtype=np.float32) / 255.0
        f_arr = np.array(f, dtype=np.float32) / 255.0
        diff = np.abs(p_arr - f_arr)
        mask = (diff > self.mask_threshold).astype(np.uint8) * 255  # 0 or 255
        return Image.fromarray(mask, mode="L")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        filename = sample["filename"]
        prompt = sample["prompt"]

        perfect_path = os.path.join(self.perfect_dir, filename)
        flawed_path = os.path.join(self.flawed_dir, filename)

        perfect_img = Image.open(perfect_path).convert("L")
        flawed_img = Image.open(flawed_path).convert("L")

        # gray2rgb
        perfect_img_rgb = gray_to_rgb(perfect_img)
        flawed_img_rgb = gray_to_rgb(flawed_img)

        # data enhancements
        if self.augment:
            if random.random() < 0.5:
                perfect_img_rgb = perfect_img_rgb.transpose(Image.FLIP_LEFT_RIGHT)
                flawed_img_rgb = flawed_img_rgb.transpose(Image.FLIP_LEFT_RIGHT)

        perfect_tensor = self.image_transform(perfect_img_rgb)
        flawed_tensor = self.image_transform(flawed_img_rgb)

        # get the mask
        mask_img = self._generate_mask_from_diff(perfect_img, flawed_img)
        if self.augment:
            if random.random() < 0.5:
                mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)

        mask_tensor = self.mask_transform(mask_img)  # [1,H,W], 0~1


        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "perfect": perfect_tensor,
            "flawed": flawed_tensor,
            "mask": mask_tensor,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
        }


# ==========================

def create_inpaint_components(pretrained_model_name_or_path: str, device: torch.device):

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.float32,
        safety_checker=None
    )

    vae: AutoencoderKL = pipe.vae
    unet: UNet2DConditionModel = pipe.unet
    text_encoder: CLIPTextModel = pipe.text_encoder
    tokenizer: CLIPTokenizer = pipe.tokenizer
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    return vae, unet, text_encoder, tokenizer, scheduler


def add_lora_to_unet(unet: UNet2DConditionModel, config: TrainConfig) -> UNet2DConditionModel:
    """
    使用 PEFT 在 UNet 的注意力模块上挂 LoRA。
    """
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="UNET"
    )

    unet_lora = get_peft_model(unet, lora_config)
    print(unet_lora)
    trainable_params = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet_lora.parameters())
    print(f"LoRA UNet trainable params: {trainable_params} / {total_params}")
    return unet_lora


# ==========================

def get_dataloaders(config: TrainConfig, tokenizer: CLIPTokenizer):
    samples = []
    with open(config.prompt_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            filename, prompt = row[0], row[1]
            perfect_path = os.path.join(config.perfect_dir, filename)
            flawed_path = os.path.join(config.flawed_dir, filename)
            if os.path.isfile(perfect_path) and os.path.isfile(flawed_path):
                samples.append({"filename": filename, "prompt": prompt})

    print(f"Total valid samples: {len(samples)}")

    random.shuffle(samples)

    n_total = len(samples)
    n_train = int(n_total * config.train_ratio)
    n_val = int(n_total * config.val_ratio)
    n_test = n_total - n_train - n_val

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}, Test samples: {len(test_samples)}")

    train_dataset = InpaintLoraDataset(
        train_samples,
        config.perfect_dir,
        config.flawed_dir,
        tokenizer,
        image_size=config.image_size,
        mask_threshold=config.mask_threshold,
        augment=True
    )

    val_dataset = InpaintLoraDataset(
        val_samples,
        config.perfect_dir,
        config.flawed_dir,
        tokenizer,
        image_size=config.image_size,
        mask_threshold=config.mask_threshold,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_samples  # test_samples


def compute_lr(step: int, config: TrainConfig):
    """简单的 linear warmup + constant LR。"""
    if step < config.lr_warmup_steps:
        return config.learning_rate * (step + 1) / config.lr_warmup_steps
    else:
        return config.learning_rate


def train():
    set_seed(config.seed)
    ensure_dir(config.checkpoint_dir)
    ensure_dir(config.final_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. load model
    vae, unet, text_encoder, tokenizer, noise_scheduler = create_inpaint_components(
        config.pretrained_model_name_or_path, device
    )

    # 2. Add LoRA to UNet
    unet = add_lora_to_unet(unet, config)
    unet.train()

    # train LoRA only
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    # 3. DataLoader
    train_loader, val_loader, _ = get_dataloaders(config, tokenizer)

    # 4. AMP
    scaler = GradScaler(enabled=config.use_amp)

    # 5. training
    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config.num_epochs):
        unet.train()
        train_losses = []

        for batch in train_loader:
            global_step += 1

            perfect = batch["perfect"].to(device)      # [B,3,H,W], [-1,1]
            flawed = batch["flawed"].to(device)        # [B,3,H,W], [-1,1]
            mask = batch["mask"].to(device)            # [B,1,H,W], [0,1]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 1) encode image to latent space
            with torch.no_grad():
                # reconstructed image
                latents = vae.encode(perfect).latent_dist.sample()
                latents = latents * 0.18215

                # noisy image
                masked_image_latents = vae.encode(flawed).latent_dist.sample()
                masked_image_latents = masked_image_latents * 0.18215

                # prompt
                encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask)[0]

            # 2) time step adding noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,),
                device=device, dtype=torch.long
            )

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 3) inpainting UNet
            mask_down = F.interpolate(mask, size=noisy_latents.shape[-2:], mode="nearest")

            # StableDiffusionInpaintPipeline
            # [latents, mask, masked_image_latents] -> in_channels = 4 + 1 + 4 = 9
            latent_model_input = torch.cat([noisy_latents, mask_down, masked_image_latents], dim=1)

            # 4) predict noise
            with autocast(device_type="cuda", enabled=config.use_amp):
                noise_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states
                ).sample

                loss = F.mse_loss(noise_pred, noise, reduction="mean")

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            # adjust LR (warmup)
            lr = compute_lr(global_step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            train_losses.append(loss.item())

            if global_step % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{config.num_epochs}] "
                    f"Step [{global_step}] "
                    f"Loss: {loss.item():.4f} "
                    f"LR: {lr:.6f}"
                )

        avg_train_loss = sum(train_losses) / max(1, len(train_losses))

        # ===== val =====
        unet.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                perfect = batch["perfect"].to(device)
                flawed = batch["flawed"].to(device)
                mask = batch["mask"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # encode
                latents = vae.encode(perfect).latent_dist.sample()
                latents = latents * 0.18215
                masked_image_latents = vae.encode(flawed).latent_dist.sample()
                masked_image_latents = masked_image_latents * 0.18215

                encoder_hidden_states = text_encoder(input_ids, attention_mask=attention_mask)[0]

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=device, dtype=torch.long
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                mask_down = F.interpolate(mask, size=noisy_latents.shape[-2:], mode="nearest")
                latent_model_input = torch.cat([noisy_latents, mask_down, masked_image_latents], dim=1)

                with autocast(device_type="cuda", enabled=config.use_amp):
                    noise_pred = unet(
                        latent_model_input,
                        timesteps,
                        encoder_hidden_states
                    ).sample
                    val_loss = F.mse_loss(noise_pred, noise, reduction="mean")

                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / max(1, len(val_losses))

        print(
            f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # checkpoint
        ckpt_path = os.path.join(config.checkpoint_dir, f"epoch_{epoch+1:03d}")
        ensure_dir(ckpt_path)
        # LoRA
        unet.save_pretrained(ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        # earlystop
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0

            # save best checkpoint
            best_path = os.path.join(config.final_dir, "best_lora")
            ensure_dir(best_path)
            unet.save_pretrained(best_path)
            print(f"New best model saved to {best_path} (val_loss={best_val_loss:.4f})")

        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

            if epochs_without_improvement >= config.early_stopping_patience:
                print("Early stopping triggered.")
                break

    print("Training finished.")


if __name__ == "__main__":
    train()
