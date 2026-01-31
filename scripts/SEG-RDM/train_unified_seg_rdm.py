#!/usr/bin/env python3
"""
Training script for Unified Segmentation RDM.
Trains a diffusion model on both global vectors (from I-JEPA) and 
segmentation tokens (from SAM embeddings).
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rdm.models.diffusion.ddpm import UnifiedSegRDM
from rdm.data.seg_dataset import SegmentationMaskDataset, collate_seg_batch
from rdm.util import instantiate_from_config


class UnifiedSegRDMTrainer:
    """Trainer for Unified Segmentation RDM."""
    
    def __init__(self, config_path: str, resume_from: str = None):
        """
        Args:
            config_path: Path to config YAML file
            resume_from: Path to checkpoint to resume from
        """
        # Load config
        self.config = OmegaConf.load(config_path)
        print(f"Loaded config from: {config_path}")
        print(OmegaConf.to_yaml(self.config))
        
        # Validate data alignment before training
        print("\n=== Validating Data Alignment ===")
        self.validate_data_alignment()
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Setup directories
        self.setup_directories()
        
        # Initialize model
        print("\n=== Initializing Model ===")
        self.model = self.build_model()
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize optimizer and scheduler
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        # Mixed precision
        self.use_fp16 = self.config.training.get('use_fp16', False)
        self.scaler = GradScaler() if self.use_fp16 else None
        print(f"Mixed precision (FP16): {self.use_fp16}")
        
        # Initialize dataloaders
        print("\n=== Initializing Data ===")
        self.train_loader = self.build_dataloader()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Initialize logging
        self.setup_logging()
        
    def validate_data_alignment(self):
        """Validate that images have corresponding SAM embeddings."""
        import numpy as np
        from PIL import Image
        
        image_dir = Path(self.config.data.params.image_dir)
        mask_npz_dir = Path(self.config.data.params.mask_npz_dir)
        
        if not image_dir.exists():
            print(f"Warning: Image directory not found: {image_dir}")
            return
        
        if not mask_npz_dir.exists():
            print(f"Warning: SAM embeddings directory not found: {mask_npz_dir}")
            return
        
        # Find sample images
        image_exts = ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG']
        sample_images = []
        for ext in image_exts:
            sample_images.extend(list(image_dir.rglob(f'*{ext}'))[:100])
            if len(sample_images) >= 100:
                break
        
        if len(sample_images) == 0:
            print(f"Warning: No images found in {image_dir}")
            return
        
        missing = []
        invalid = []
        
        for img_path in sample_images[:20]:  # Check first 20
            rel_path = img_path.relative_to(image_dir)
            npz_path = mask_npz_dir / rel_path.parent / f"{rel_path.stem}.npz"
            
            if not npz_path.exists():
                missing.append((str(img_path.name), str(npz_path)))
            else:
                try:
                    data = np.load(npz_path)
                    if 'emb' not in data:
                        invalid.append((str(img_path.name), "Missing 'emb' key"))
                    else:
                        emb = data['emb']
                        if emb.ndim != 2 or emb.shape[1] != 256:
                            invalid.append((str(img_path.name), f"Invalid shape: {emb.shape}"))
                except Exception as e:
                    invalid.append((str(img_path.name), str(e)))
        
        if missing:
            print(f"⚠ Warning: {len(missing)} sample images missing SAM embeddings")
            for img, npz in missing[:3]:
                print(f"  - {img} → {npz}")
            if len(missing) > 3:
                print(f"  ... and {len(missing) - 3} more")
        
        if invalid:
            print(f"⚠ Warning: {len(invalid)} sample embeddings invalid")
            for img, error in invalid[:3]:
                print(f"  - {img}: {error}")
        
        if not missing and not invalid:
            print(f"✓ Data alignment verified ({len(sample_images)} samples checked)")
        else:
            print(f"⚠ Please fix alignment issues before training")
            print(f"  Run: python scripts/verify_data_alignment.py ...")
    
    def setup_directories(self):
        """Create necessary directories."""
        self.checkpoint_dir = Path(self.config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.logging.get('use_tensorboard', False):
            self.tensorboard_dir = Path(self.config.logging.tensorboard_dir)
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    def build_model(self) -> UnifiedSegRDM:
        """Build the UnifiedSegRDM model from config."""
        model_config = self.config.model
        model = instantiate_from_config(model_config)
        return model
    
    def build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer."""
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.get('weight_decay', 0.0)
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        print(f"Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
        return optimizer
    
    def build_scheduler(self):
        """Build learning rate scheduler."""
        max_steps = self.config.training.max_steps
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps,
            eta_min=1e-6
        )
        print(f"Scheduler: CosineAnnealingLR (T_max={max_steps})")
        return scheduler
    
    def build_dataloader(self) -> DataLoader:
        """Build training dataloader."""
        data_config = self.config.data
        
        # Create dataset
        dataset = SegmentationMaskDataset(
            image_dir=data_config.params.image_dir,
            mask_npz_dir=data_config.params.mask_npz_dir,
            max_segments=data_config.params.max_segments,
            image_size=data_config.params.image_size,
            file_ext=data_config.params.get('file_ext', '*.jpg'),
            normalize=data_config.params.get('normalize', True),
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            shuffle=data_config.get('shuffle', True),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=data_config.get('pin_memory', True),
            collate_fn=collate_seg_batch,
            drop_last=True,
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Batch size: {data_config.batch_size}")
        print(f"Steps per epoch: {len(dataloader)}")
        
        return dataloader
    
    def setup_logging(self):
        """Initialize logging (wandb, tensorboard)."""
        if self.config.logging.get('use_wandb', False):
            wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.get('wandb_entity'),
                name=self.config.logging.get('wandb_run_name'),
                config=OmegaConf.to_container(self.config, resolve=True),
                resume='allow',
            )
            print("Weights & Biases logging enabled")
        
        if self.config.logging.get('use_tensorboard', False):
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.tensorboard_dir)
            print("TensorBoard logging enabled")
        else:
            self.tb_writer = None
    
    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()
        
        # Forward pass with mixed precision
        if self.use_fp16:
            with autocast():
                loss, loss_dict = self.model(x=None, c=None, batch=batch)
        else:
            loss, loss_dict = self.model(x=None, c=None, batch=batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_fp16:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config.training.get('gradient_clip_val', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_val
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if self.config.training.get('gradient_clip_val', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_val
                )
            
            self.optimizer.step()
        
        self.scheduler.step()
        
        return loss_dict
    
    def log_metrics(self, loss_dict: dict, step: int):
        """Log metrics to wandb/tensorboard."""
        # Convert tensors to scalars
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
        
        # Log to wandb
        if self.config.logging.get('use_wandb', False):
            wandb.log(metrics, step=step)
        
        # Log to tensorboard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)
    
    def save_checkpoint(self, step: int):
        """Save checkpoint."""
        checkpoint = {
            'global_step': step,
            'current_epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': OmegaConf.to_container(self.config, resolve=True),
        }
        
        if self.use_fp16:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_step_{step:07d}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
        
        # Save latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Also save with EMA if available
        if self.model.use_ema:
            with self.model.ema_scope():
                ema_checkpoint = {
                    'global_step': step,
                    'model_state_dict': self.model.state_dict(),
                }
                ema_path = self.checkpoint_dir / f"checkpoint_ema_step_{step:07d}.pt"
                torch.save(ema_checkpoint, ema_path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        print(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint.get('current_epoch', 0)
        
        if self.use_fp16 and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Resumed from step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        print("\n=== Starting Training ===")
        max_steps = self.config.training.max_steps
        log_every = self.config.training.log_every_n_steps
        save_every = self.config.training.save_checkpoint_every_n_steps
        
        pbar = tqdm(total=max_steps, initial=self.global_step, desc="Training")
        
        while self.global_step < max_steps:
            self.current_epoch += 1
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Train step
                loss_dict = self.train_step(batch)
                self.global_step += 1
                
                # Logging
                if self.global_step % log_every == 0:
                    self.log_metrics(loss_dict, self.global_step)
                    pbar.set_postfix({
                        'loss': f"{loss_dict.get('train/loss', 0):.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                
                # Save checkpoint
                if self.global_step % save_every == 0:
                    self.save_checkpoint(self.global_step)
                
                pbar.update(1)
                
                if self.global_step >= max_steps:
                    break
        
        pbar.close()
        
        # Final checkpoint
        self.save_checkpoint(self.global_step)
        print("\n=== Training Complete ===")
        
        # Cleanup
        if self.config.logging.get('use_wandb', False):
            wandb.finish()
        if self.tb_writer is not None:
            self.tb_writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Unified Segmentation RDM")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    # Create trainer
    trainer = UnifiedSegRDMTrainer(
        config_path=args.config,
        resume_from=args.resume
    )
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(trainer.global_step)
        print("Checkpoint saved. Exiting.")


if __name__ == "__main__":
    main()
