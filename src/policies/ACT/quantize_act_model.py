#!/usr/bin/env python3
"""
Post-Training Quantization for ACT Policy on Raspberry Pi 5

Quantizes the trained ACT model to INT8 for faster inference on edge devices.
Supports multiple quantization strategies:
1. Dynamic Quantization (fastest to apply, good for transformer layers)
2. Static Quantization (best accuracy, requires calibration)
3. Mixed Precision (optimal balance)

Usage:
    # Dynamic quantization (recommended for quick deployment)
    python quantize_act_model.py \
        --checkpoint outputs/lerobot_act/best_model.pth \
        --mode dynamic \
        --output outputs/lerobot_act/best_model_quantized.pth

    # Static quantization (best accuracy, needs calibration data)
    python quantize_act_model.py \
        --checkpoint outputs/lerobot_act/best_model.pth \
        --mode static \
        --calibration_data src/robots/rover/episodes \
        --output outputs/lerobot_act/best_model_static_quant.pth
"""

import sys
import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import quantize_dynamic
# PyTorch 2.0+ moved FX quantization to torch.ao.quantization.quantize_fx
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping
import argparse
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Any
from PIL import Image
from torchvision import transforms

# Setup LeRobot imports
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import PolicyFeature, FeatureType

# Setup dataset loader
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "datasets"))
from local_dataset_loader import TracerLocalDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ACTModelQuantizer:
    """Handles quantization of ACT models for edge deployment"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cpu',  # Quantization typically done on CPU
        image_size: tuple = (360, 640),
    ):
        """
        Initialize quantizer
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device for quantization (use 'cpu' for deployment preparation)
            image_size: Input image size (height, width)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.image_size = image_size
        
        logger.info(f"Loading model from {checkpoint_path}")
        # PyTorch 2.6+ requires weights_only=False for models saved with older versions
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get training config from checkpoint (this is the dict, not ACTConfig)
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint does not contain config! Cannot load model.")
        
        training_config = checkpoint['config']
        logger.info("✅ Loaded training config from checkpoint")
        
        # Create ACT configuration using the same approach as trainer
        act_config = ACTConfig(
            # Input/Output features for RC car (same as trainer)
            input_features={
                "observation.images.cam_front": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, training_config['image_height'], training_config['image_width'])
                ),
                "observation.state": PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(2,)  # [steering, throttle]
                ),
            },
            output_features={
                "action": PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(2,)  # [steering, throttle] PWM outputs
                ),
            },
            
            # Configuration from training
            n_obs_steps=training_config['n_obs_steps'],
            chunk_size=training_config['chunk_size'],
            n_action_steps=training_config['n_action_steps'],
            dim_model=training_config['hidden_size'],
            n_encoder_layers=training_config['n_encoder_layers'],
            n_decoder_layers=training_config['n_decoder_layers'],
            n_heads=training_config['n_heads'],
            dim_feedforward=training_config['feedforward_dim'],
            dropout=training_config['dropout'],
            kl_weight=training_config['kl_weight'],
            vision_backbone=training_config['vision_encoder'],
            pretrained_backbone_weights=None,  # Not needed for quantization
            use_vae=True,
            latent_dim=32,
            device='cpu',  # Always load on CPU for quantization
        )
        
        # Save configs for later use
        self.config = training_config  # Original training config dict
        self.act_config = act_config   # ACTConfig object
        
        # Create and load model
        self.model = ACTPolicy(act_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to('cpu')
        self.model.eval()
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
    
    def dynamic_quantization(self, output_path: str):
        """
        Apply dynamic quantization (easiest method, good for transformers)
        
        Quantizes weights to INT8 and keeps activations in FP32.
        Good for: Transformer layers (self-attention, feedforward)
        Benefits: 2-4x speedup, 4x memory reduction, minimal accuracy loss
        
        Args:
            output_path: Where to save quantized model
        """
        logger.info("🔧 Applying dynamic quantization...")
        
        # Quantize linear and attention layers
        quantized_model = quantize_dynamic(
            self.model,
            {nn.Linear, nn.MultiheadAttention},  # Quantize these layer types
            dtype=torch.qint8,  # Use INT8
            inplace=False
        )
        
        # Save quantized model
        self._save_quantized_model(quantized_model, output_path, "dynamic")
        
        return quantized_model
    
    def static_quantization(
        self,
        calibration_data_dir: str,
        output_path: str,
        num_calibration_batches: int = 100,
    ):
        """
        Apply static quantization (best accuracy, needs calibration)
        
        Quantizes both weights and activations to INT8.
        Requires calibration data to determine activation ranges.
        Benefits: Best latency, minimal accuracy loss with proper calibration
        
        Args:
            calibration_data_dir: Directory with calibration episodes
            output_path: Where to save quantized model
            num_calibration_batches: Number of batches for calibration
        """
        logger.info("🔧 Applying static quantization with calibration...")
        
        # Set qconfig - use float_qparams for embeddings
        # For ARM (Raspberry Pi), we'll set qnnpack backend
        import torch.backends.quantized
        torch.backends.quantized.engine = 'qnnpack'  # ARM optimization
        
        # Configure quantization - embeddings need special handling
        logger.info("   Configuring quantization settings...")
        
        # Set default qconfig for most layers
        self.model.qconfig = quant.get_default_qconfig('qnnpack')
        
        # Override qconfig for embedding layers (they need float_qparams)
        def set_embedding_qconfig(module):
            """Recursively set qconfig for embedding layers"""
            for name, child in module.named_children():
                if isinstance(child, nn.Embedding):
                    child.qconfig = quant.float_qparams_weight_only_qconfig
                    logger.info(f"      Set float_qparams for embedding: {name}")
                else:
                    set_embedding_qconfig(child)
        
        set_embedding_qconfig(self.model)
        
        # Fuse operations (Conv+BN+ReLU, etc.) for better efficiency
        logger.info("   Fusing layers...")
        self.model = self._fuse_model(self.model)
        
        # Prepare for quantization (insert observers)
        logger.info("   Inserting observers...")
        prepared_model = quant.prepare(self.model, inplace=False)
        
        # Calibration: run inference on sample data to collect statistics
        logger.info("   Running calibration...")
        self._calibrate_model(prepared_model, calibration_data_dir, num_calibration_batches)
        
        # Convert to quantized model
        logger.info("   Converting to quantized model...")
        quantized_model = quant.convert(prepared_model, inplace=False)
        
        # Save quantized model
        self._save_quantized_model(quantized_model, output_path, "static")
        
        return quantized_model
    
    def mixed_precision_quantization(
        self,
        calibration_data_dir: str,
        output_path: str,
        num_calibration_batches: int = 100,
    ):
        """
        Apply mixed precision quantization (optimal balance)
        
        Uses INT8 for most layers, FP16 for sensitive layers (vision encoder).
        Best for: Preserving accuracy while maximizing speedup
        
        Args:
            calibration_data_dir: Directory with calibration episodes
            output_path: Where to save quantized model
            num_calibration_batches: Number of batches for calibration
        """
        logger.info("🔧 Applying mixed precision quantization...")
        
        # Clone model for mixed precision
        model = self.model
        
        # Apply dynamic quantization to transformer layers only
        logger.info("   Quantizing transformer layers to INT8...")
        if hasattr(model, 'model') and hasattr(model.model, 'transformer'):
            model.model.transformer = quantize_dynamic(
                model.model.transformer,
                {nn.Linear, nn.MultiheadAttention},
                dtype=torch.qint8,
                inplace=False
            )
        
        # Keep vision encoder in FP32 (or could use FP16)
        logger.info("   Keeping vision encoder in FP32 for accuracy...")
        
        # Save mixed precision model
        self._save_quantized_model(model, output_path, "mixed")
        
        return model
    
    def _fuse_model(self, model):
        """Fuse consecutive operations for efficiency"""
        # For ResNet backbone: fuse Conv+BN+ReLU
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'vision_encoder'):
                vision_encoder = model.model.vision_encoder
                # Fuse Conv-BN-ReLU patterns
                torch.quantization.fuse_modules(
                    vision_encoder,
                    [['conv1', 'bn1', 'relu']],  # Example fusion
                    inplace=True
                )
        except Exception as e:
            logger.warning(f"Could not fuse vision encoder layers: {e}")
        
        return model
    
    def _calibrate_model(self, model, data_dir: str, num_batches: int):
        """Run calibration to collect activation statistics"""
        # Load calibration dataset
        dataset = TracerLocalDataset(data_dir=data_dir)
        
        # Image preprocessing (dataset already applies transforms, but we create it with our settings)
        image_transform = transforms.Compose([
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Recreate dataset with our transform
        dataset = TracerLocalDataset(
            data_dir=data_dir,
            transforms=image_transform,
            sync_tolerance=0.05
        )
        
        # Run inference on calibration samples
        model.eval()
        num_samples = min(num_batches, len(dataset))
        
        logger.info(f"   Calibrating with {num_samples} samples...")
        
        chunk_size = self.config['chunk_size']  # Get from training config
        successful_calibrations = 0
        
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Calibrating"):
                try:
                    sample = dataset[i]
                    
                    # Prepare batch - must match training format exactly
                    # The model expects action with shape [B, chunk_size, action_dim]
                    action = sample['action'].unsqueeze(0)  # [1, 2]
                    action = action.unsqueeze(1).repeat(1, chunk_size, 1)  # [1, chunk_size, 2]
                    
                    batch = {
                        "observation.images.cam_front": sample["observation.images.cam_front"].unsqueeze(0),
                        "observation.state": sample["observation.state"].unsqueeze(0),
                        "action": action,
                        "action_is_pad": torch.zeros(1, chunk_size, dtype=torch.bool),
                    }
                    
                    # Forward pass (observers collect statistics)
                    # Model returns (loss, loss_dict) during training mode
                    output = model(batch)
                    successful_calibrations += 1
                    
                except Exception as e:
                    logger.warning(f"Skipping calibration sample {i}: {e}")
                    continue
        
        logger.info(f"   ✅ Calibration complete ({successful_calibrations}/{num_samples} successful)")
    
    def _save_quantized_model(self, model, output_path: str, mode: str):
        """Save quantized model with metadata"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get model size
        original_size = Path(self.checkpoint_path).stat().st_size / (1024 * 1024)  # MB
        
        # For quantized models, we need to save the actual model object, not just state_dict
        # This preserves the quantization metadata and allows loading without reconfiguration
        save_dict = {
            'model': model,  # Save the full quantized model
            'model_state_dict': model.state_dict(),  # Also save state_dict for compatibility
            'config': self.config,
            'quantization_mode': mode,
            'image_size': self.image_size,
            'original_checkpoint': str(self.checkpoint_path),
        }
        
        torch.save(save_dict, output_path)
        
        quantized_size = output_path.stat().st_size / (1024 * 1024)  # MB
        compression_ratio = original_size / quantized_size
        
        logger.info(f"✅ Quantized model saved to {output_path}")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Original size: {original_size:.2f} MB")
        logger.info(f"   Quantized size: {quantized_size:.2f} MB")
        logger.info(f"   Compression: {compression_ratio:.2f}x")
        logger.info(f"   Size reduction: {(1 - quantized_size/original_size) * 100:.1f}%")
    
    def benchmark_inference(self, model, num_iterations: int = 100):
        """Benchmark inference latency"""
        import time
        
        # Create dummy input
        dummy_batch = {
            "observation.images.cam_front": torch.randn(1, 3, self.image_size[0], self.image_size[1]),
            "observation.state": torch.randn(1, 2),
        }
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_batch)
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in tqdm(range(num_iterations), desc="Benchmarking"):
                start = time.perf_counter()
                _ = model(dummy_batch)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        logger.info("📊 Inference Latency Benchmark:")
        logger.info(f"   Average: {avg_latency:.2f} ± {std_latency:.2f} ms")
        logger.info(f"   P50: {p50_latency:.2f} ms")
        logger.info(f"   P95: {p95_latency:.2f} ms")
        logger.info(f"   P99: {p99_latency:.2f} ms")
        logger.info(f"   Throughput: {1000/avg_latency:.1f} FPS")
        
        return {
            'avg': avg_latency,
            'std': std_latency,
            'p50': p50_latency,
            'p95': p95_latency,
            'p99': p99_latency,
        }


def compare_models(original_checkpoint: str, quantized_checkpoint: str, test_data_dir: str):
    """Compare original vs quantized model accuracy on test data"""
    logger.info("🔍 Comparing original vs quantized model...")
    
    # Load both models
    logger.info("Loading original model...")
    original_checkpoint_data = torch.load(original_checkpoint, map_location='cpu')
    
    # Create config
    config = ACTConfig(
        input_shapes={
            "observation.images.cam_front": [3, 360, 640],
            "observation.state": [2],
        },
        output_shapes={"action": [2]},
        n_obs_steps=1,
        chunk_size=32,
        n_action_steps=32,
        vision_backbone="resnet18",
        use_vae=True,
        latent_dim=32,
        device='cpu',
    )
    
    original_model = ACTPolicy(config)
    original_model.load_state_dict(original_checkpoint_data['model_state_dict'])
    original_model.eval()
    
    logger.info("Loading quantized model...")
    quantized_data = torch.load(quantized_checkpoint, map_location='cpu')
    quantized_model = ACTPolicy(config)
    quantized_model.load_state_dict(quantized_data['model_state_dict'])
    quantized_model.eval()
    
    # Load test data
    dataset = TracerLocalDataset(data_dir=test_data_dir)
    num_samples = min(100, len(dataset))
    
    logger.info(f"Comparing on {num_samples} test samples...")
    
    mse_errors = []
    max_errors = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Testing"):
            try:
                sample = dataset[i]
                
                batch = {
                    "observation.images.cam_front": sample["observation.images.cam_front"].unsqueeze(0),
                    "observation.state": sample["observation.state"].unsqueeze(0),
                }
                
                # Get predictions
                original_output = original_model(batch)
                quantized_output = quantized_model(batch)
                
                # Compare actions
                mse = torch.mean((original_output - quantized_output) ** 2).item()
                max_err = torch.max(torch.abs(original_output - quantized_output)).item()
                
                mse_errors.append(mse)
                max_errors.append(max_err)
                
            except Exception as e:
                logger.warning(f"Skipping sample {i}: {e}")
                continue
    
    avg_mse = np.mean(mse_errors)
    avg_max_error = np.mean(max_errors)
    
    logger.info("📊 Accuracy Comparison:")
    logger.info(f"   Average MSE: {avg_mse:.6f}")
    logger.info(f"   Average Max Error: {avg_max_error:.6f}")
    logger.info(f"   Max Error Range: [{min(max_errors):.6f}, {max(max_errors):.6f}]")
    
    if avg_mse < 0.001:
        logger.info("   ✅ Excellent: Negligible accuracy loss")
    elif avg_mse < 0.01:
        logger.info("   ✅ Good: Minor accuracy loss, acceptable for deployment")
    else:
        logger.info("   ⚠️  Warning: Significant accuracy loss, consider recalibration")
    
    return {
        'mse': avg_mse,
        'max_error': avg_max_error,
    }


def main():
    parser = argparse.ArgumentParser(description='Quantize ACT Model for Edge Deployment')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--mode', type=str, default='dynamic', 
                       choices=['dynamic', 'static', 'mixed'],
                       help='Quantization mode')
    parser.add_argument('--output', type=str, required=True, help='Output path for quantized model')
    parser.add_argument('--calibration_data', type=str, help='Path to calibration data (for static/mixed)')
    parser.add_argument('--num_calibration_batches', type=int, default=100,
                       help='Number of batches for calibration')
    parser.add_argument('--benchmark', action='store_true', help='Run inference benchmark')
    parser.add_argument('--compare', action='store_true', help='Compare accuracy with original')
    parser.add_argument('--test_data', type=str, help='Test data for comparison')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['static', 'mixed'] and not args.calibration_data:
        parser.error(f"--calibration_data required for {args.mode} quantization")
    
    if args.compare and not args.test_data:
        parser.error("--test_data required for accuracy comparison")
    
    # Create quantizer
    quantizer = ACTModelQuantizer(
        checkpoint_path=args.checkpoint,
        device='cpu',
        image_size=(360, 640),
    )
    
    print("=" * 80)
    print("🔧 ACT MODEL QUANTIZATION FOR RASPBERRY PI 5")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Input: {args.checkpoint}")
    print(f"Output: {args.output}")
    print("=" * 80)
    print()
    
    # Apply quantization
    if args.mode == 'dynamic':
        quantized_model = quantizer.dynamic_quantization(args.output)
    elif args.mode == 'static':
        quantized_model = quantizer.static_quantization(
            args.calibration_data,
            args.output,
            args.num_calibration_batches
        )
    elif args.mode == 'mixed':
        quantized_model = quantizer.mixed_precision_quantization(
            args.calibration_data,
            args.output,
            args.num_calibration_batches
        )
    
    # Benchmark if requested
    if args.benchmark:
        print()
        print("=" * 80)
        print("🚀 BENCHMARKING QUANTIZED MODEL")
        print("=" * 80)
        quantizer.benchmark_inference(quantized_model, num_iterations=100)
    
    # Compare accuracy if requested
    if args.compare:
        print()
        print("=" * 80)
        print("📊 ACCURACY COMPARISON")
        print("=" * 80)
        compare_models(args.checkpoint, args.output, args.test_data)
    
    print()
    print("=" * 80)
    print("✅ QUANTIZATION COMPLETE!")
    print("=" * 80)
    print(f"📁 Quantized model saved to: {args.output}")
    print()
    print("Next steps:")
    print("1. Test inference: python lerobot_act_inference_quantized.py --checkpoint", args.output)
    print("2. Deploy to Raspberry Pi 5")
    print("3. Benchmark on-device performance")
    print("=" * 80)


if __name__ == "__main__":
    main()
