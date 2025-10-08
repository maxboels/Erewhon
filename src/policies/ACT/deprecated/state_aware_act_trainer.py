#!/usr/bin/env python3
"""
State-Aware ACT Training Script for Tracer RC Car
Uses both camera observations AND current state (steering, throttle) as inputs
Optimized for edge deployment on Raspberry Pi 5 with AI HAT
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from datetime import datetime
import csv
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedTracerDataset(Dataset):
    """Dataset for Tracer RC car that includes both images and current state"""
    
    def __init__(self, data_dir: str, transforms=None, episode_length: int = None):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.episode_length = episode_length
        
        # Load all episodes
        self.samples = []
        episode_dirs = sorted([d for d in self.data_dir.iterdir() 
                              if d.is_dir() and (d / "episode_data.json").exists()])
        
        for episode_dir in episode_dirs:
            with open(episode_dir / "episode_data.json", 'r') as f:
                episode_data = json.load(f)
            
            frame_samples = episode_data.get('frame_samples', [])
            control_samples = episode_data.get('control_samples', [])
            
            # Sort by timestamp
            frame_samples.sort(key=lambda x: x['timestamp'])
            control_samples.sort(key=lambda x: x['system_timestamp'])
            
            # Synchronize frames with controls
            for i, frame_sample in enumerate(frame_samples):
                frame_time = frame_sample['timestamp']
                
                # Find closest control sample
                closest_control = min(control_samples, 
                                    key=lambda c: abs(c['system_timestamp'] - frame_time))
                
                time_diff = abs(closest_control['system_timestamp'] - frame_time)
                
                if time_diff < 0.05:  # 50ms tolerance
                    self.samples.append({
                        'image_path': episode_dir / frame_sample['image_path'],
                        'steering': closest_control['steering_normalized'],
                        'throttle': closest_control['throttle_normalized'],
                    })
            
            # Limit episode length if specified
            if episode_length and len(self.samples) > episode_length:
                self.samples = self.samples[:episode_length]
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(episode_dirs)} episodes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            if self.transforms:
                image = self.transforms(image)
        except Exception as e:
            logger.error(f"Error loading image {sample['image_path']}: {e}")
            image = torch.zeros(3, 480, 640, dtype=torch.float32)
        
        # Current state vector [steering, throttle]
        state = torch.tensor([sample['steering'], sample['throttle']], dtype=torch.float32)
        
        # Action vector (same as state for imitation learning)
        action = torch.tensor([sample['steering'], sample['throttle']], dtype=torch.float32)
        
        return image, state, action


class StateAwareACTModel(nn.Module):
    """ACT-like model that uses BOTH visual observations AND current state"""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (360, 640),  # Actual data: 640x360 (W×H) = (360,640) in PyTorch (H×W)
        state_dim: int = 2,  # [steering, throttle]
        action_dim: int = 2,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        chunk_size: int = 32
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # Vision encoder (CNN for images - optimized for 640x360 input)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),  # 360x640 -> 180x320
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),      # 180x320 -> 90x160
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 90x160 -> 45x80
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           # 45x80 -> 22x40
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 10)),             # -> 6x10 for 360 height
            
            nn.Flatten(),
            nn.Linear(256 * 6 * 10, hidden_dim),       # Adjusted for 360 height
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # State encoder (MLP for current state)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Fusion layer (combine visual and state features)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Transformer encoder for processing fused features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Action decoder
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, action_dim * chunk_size)
        )
    
    def forward(self, images, states):
        """
        Args:
            images: [batch_size, 3, H, W] - camera observations
            states: [batch_size, state_dim] - current state [steering, throttle]
        
        Returns:
            actions: [batch_size, chunk_size, action_dim] - predicted action sequence
        """
        batch_size = images.shape[0]
        
        # Encode visual features
        visual_features = self.vision_encoder(images)  # [batch_size, hidden_dim]
        
        # Encode state features
        state_features = self.state_encoder(states)    # [batch_size, 128]
        
        # Fuse visual and state features
        fused_features = torch.cat([visual_features, state_features], dim=1)  # [batch_size, hidden_dim + 128]
        fused_features = self.fusion(fused_features)  # [batch_size, hidden_dim]
        
        # Add sequence dimension for transformer
        fused_features = fused_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Process through transformer
        encoded_features = self.transformer_encoder(fused_features)  # [batch_size, 1, hidden_dim]
        
        # Decode to actions
        actions_flat = self.action_head(encoded_features.squeeze(1))  # [batch_size, action_dim * chunk_size]
        actions = actions_flat.view(batch_size, self.chunk_size, self.action_dim)
        
        return actions


class CSVLogger:
    """Simple CSV logger for tracking training progress"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_log_path = self.log_dir / 'batch_metrics.csv'
        self.epoch_log_path = self.log_dir / 'epoch_metrics.csv'
        
        with open(self.batch_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'batch_loss', 'learning_rate'])
        
        with open(self.epoch_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'learning_rate'])
    
    def log_batch(self, step: int, batch_loss: float, learning_rate: float):
        with open(self.batch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, batch_loss, learning_rate])
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, best_val_loss: float, learning_rate: float):
        with open(self.epoch_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, best_val_loss, learning_rate])


class StateAwareACTTrainer:
    """Trainer for state-aware ACT policy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup CSV logging
        self.csv_logger = CSVLogger(self.output_dir / 'logs')
        logger.info(f"CSV logs will be saved to: {self.output_dir / 'logs'}")
        
        # Save config
        with open(self.output_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Setup dataset and data loaders
        self.setup_dataset()
        
        # Setup model
        self.setup_model()
        
        # Setup training components
        self.setup_training()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def setup_dataset(self):
        """Setup dataset with image transforms"""
        logger.info("Setting up state-aware dataset...")
        
        # Image transforms
        transform = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        full_dataset = SimplifiedTracerDataset(
            self.config['data_dir'],
            transforms=transform,
            episode_length=self.config.get('episode_length')
        )
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        logger.info(f"Dataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} validation")
        logger.info(f"Using actual resolution: 360x640 pixels (H×W) with state input")
    
    def setup_model(self):
        """Setup the state-aware model"""
        self.model = StateAwareACTModel(
            image_size=(360, 640),  # Actual data resolution (H×W)
            state_dim=2,  # [steering, throttle]
            action_dim=2,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            chunk_size=self.config['chunk_size']
        ).to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        logger.info(f"Model architecture: State-aware (image + current state)")
    
    def setup_training(self):
        """Setup training components"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['max_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )
        
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, (images, states, actions) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device)
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            # Forward pass - model uses both image and state
            predicted_actions = self.model(images, states)  # [batch, chunk_size, action_dim]
            
            # Use first action from chunk for training
            predicted_actions_first = predicted_actions[:, 0, :]  # [batch, action_dim]
            
            # Compute loss
            loss = self.criterion(predicted_actions_first, actions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            # Log batch metrics
            if self.global_step % self.config['log_freq'] == 0:
                self.csv_logger.log_batch(self.global_step, loss.item(), self.optimizer.param_groups[0]['lr'])
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.6f}")
        
        return np.mean(epoch_losses)
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = []
        
        with torch.no_grad():
            for images, states, actions in self.val_loader:
                images = images.to(self.device)
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                predicted_actions = self.model(images, states)
                predicted_actions_first = predicted_actions[:, 0, :]
                
                loss = self.criterion(predicted_actions_first, actions)
                epoch_losses.append(loss.item())
        
        return np.mean(epoch_losses)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        if (epoch + 1) % self.config['save_freq'] == 0:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"✨ Saved new best model: {best_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting state-aware ACT training...")
        logger.info(f"Training for {self.config['max_epochs']} epochs")
        logger.info(f"Model receives: Image (640x360) + Current State [steering, throttle]")
        
        for epoch in range(self.config['max_epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch metrics
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.csv_logger.log_epoch(
                epoch, train_loss, val_loss, self.best_val_loss,
                self.optimizer.param_groups[0]['lr']
            )
            
            logger.info(f"Epoch {epoch+1}/{self.config['max_epochs']} | "
                       f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                       f"Best Val: {self.best_val_loss:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
        
        logger.info("🎉 Training completed!")
        logger.info(f"📊 Final Results:")
        logger.info(f"  Best validation loss: {self.best_val_loss:.6f}")
        logger.info(f"  Model checkpoint: {self.output_dir / 'best_model.pth'}")


def get_default_config():
    """Get default training configuration for state-aware ACT"""
    return {
        # Data
        'data_dir': '/home/maxboels/projects/Erewhon/src/robots/rover/episodes',
        'batch_size': 8,
        'num_workers': 4,
        'episode_length': None,  # Use all samples
        
        # Model (state-aware)
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'chunk_size': 32,
        
        # Training
        'max_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'log_freq': 10,
        'save_freq': 5,
        
        # Hardware
        'device': 'cuda',
        
        # Output
        'output_dir': f'./outputs/state_aware_act_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }


def main():
    parser = argparse.ArgumentParser(description='Train state-aware ACT policy for RC car')
    parser.add_argument('--data_dir', type=str, help='Path to episode data directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--max_epochs', type=int, help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Get default config
    config = get_default_config()
    
    # Override with command line arguments
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.max_epochs:
        config['max_epochs'] = args.max_epochs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.device:
        config['device'] = args.device
    
    # Validate data directory
    if not Path(config['data_dir']).exists():
        logger.error(f"Data directory does not exist: {config['data_dir']}")
        sys.exit(1)
    
    logger.info("🤖 State-Aware ACT Training for RC Car")
    logger.info("=" * 60)
    logger.info("This model uses BOTH:")
    logger.info("  ✅ Camera observations (640x480)")
    logger.info("  ✅ Current state [steering, throttle]")
    logger.info("=" * 60)
    
    # Create and run trainer
    trainer = StateAwareACTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
