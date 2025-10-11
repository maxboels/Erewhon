#!/usr/bin/env python3
"""
LeRobot ACT Quantized Inference for RC Car on Raspberry Pi 5

Optimized inference using quantized INT8 models for low-latency edge deployment.
Supports dynamic, static, and mixed precision quantized models.

Usage:
    # Run inference with quantized model
    python lerobot_act_inference_quantized.py \
        --checkpoint outputs/lerobot_act/best_model_quantized.pth \
        --test_image test.jpg \
        --benchmark

    # Real-time control mode
    python lerobot_act_inference_quantized.py \
        --checkpoint outputs/lerobot_act/best_model_quantized.pth \
        --camera_id 0 \
        --arduino_port /dev/ttyUSB0
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import logging
import argparse
import time
from typing import Tuple, Optional, Dict, Any
from torchvision import transforms
import cv2

# Setup LeRobot imports
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuantizedACTInference:
    """Optimized inference with quantized ACT models"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cpu',  # Quantized models run on CPU
        image_size: Tuple[int, int] = (360, 640),
    ):
        """
        Initialize quantized ACT inference
        
        Args:
            checkpoint_path: Path to quantized model checkpoint
            device: Device to run inference on (typically 'cpu' for quantized models)
            image_size: Input image size (height, width)
        """
        self.device = device
        self.image_size = image_size
        
        # Load quantized checkpoint
        logger.info(f"Loading quantized model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check if it's a quantized model
        if 'quantization_mode' in checkpoint:
            logger.info(f"✅ Detected {checkpoint['quantization_mode']} quantized model")
            self.quantization_mode = checkpoint['quantization_mode']
        else:
            logger.warning("⚠️  Checkpoint doesn't appear to be quantized")
            self.quantization_mode = 'none'
        
        # Load or create config
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            # Fallback: create config
            self.config = ACTConfig(
                input_shapes={
                    "observation.images.cam_front": [3, image_size[0], image_size[1]],
                    "observation.state": [2],
                },
                output_shapes={
                    "action": [2],
                },
                n_obs_steps=1,
                chunk_size=32,
                n_action_steps=32,
                vision_backbone="resnet18",
                pretrained_backbone_weights=None,
                use_vae=True,
                latent_dim=32,
                device='cpu',
            )
        
        # Try to load full quantized model first (new format)
        if 'model' in checkpoint:
            logger.info("Loading full quantized model object...")
            self.policy = checkpoint['model']
            self.policy.to('cpu')
            self.policy.eval()
        else:
            # Fallback: load from state_dict (old format or non-quantized)
            logger.info("Loading from state_dict (compatibility mode)...")
            self.policy = ACTPolicy(self.config)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.policy.to('cpu')
            self.policy.eval()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # State tracking
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)
        self.action_queue = []
        
        # Performance tracking
        self.inference_times = []
        
        logger.info(f"✅ Quantized ACT model loaded successfully")
        logger.info(f"   Quantization mode: {self.quantization_mode}")
        logger.info(f"   Image size: {image_size}")
        logger.info(f"   Chunk size: {self.config.chunk_size}")
        
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.image_transform(image).unsqueeze(0)
    
    def predict(
        self,
        image: Image.Image,
        current_state: Optional[np.ndarray] = None,
        use_action_chunking: bool = True,
    ) -> Tuple[float, float]:
        """
        Predict control actions from image observation
        
        Args:
            image: Input camera frame
            current_state: Current [steering, throttle] state (optional)
            use_action_chunking: Whether to use action chunking for smooth control
            
        Returns:
            (steering, throttle) control values in range [-1, 1]
        """
        start_time = time.perf_counter()
        
        # Update state if provided
        if current_state is not None:
            self.current_state = current_state
        
        # Check if we have actions in queue (from previous chunk)
        if use_action_chunking and len(self.action_queue) > 0:
            action = self.action_queue.pop(0)
            end_time = time.perf_counter()
            self.inference_times.append((end_time - start_time) * 1000)
            return tuple(action.tolist())
        
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0)
        
        # Prepare batch
        batch = {
            "observation.images.cam_front": image_tensor,
            "observation.state": state_tensor,
        }
        
        # Forward pass
        with torch.no_grad():
            output = self.policy(batch)
        
        # Extract actions (shape: [batch, chunk_size, action_dim])
        actions = output.squeeze(0).cpu().numpy()  # [chunk_size, 2]
        
        # Store remaining actions in queue
        if use_action_chunking:
            self.action_queue = list(actions[1:])  # Store actions 1 to end
            action = actions[0]  # Use first action now
        else:
            action = actions[0]  # Just use first action
        
        end_time = time.perf_counter()
        self.inference_times.append((end_time - start_time) * 1000)
        
        # Clamp to valid range
        steering = np.clip(action[0], -1.0, 1.0)
        throttle = np.clip(action[1], -1.0, 1.0)
        
        return float(steering), float(throttle)
    
    def reset(self):
        """Reset state and action queue"""
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)
        self.action_queue = []
        self.inference_times = []
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get inference latency statistics"""
        if len(self.inference_times) == 0:
            return {}
        
        times = np.array(self.inference_times)
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'fps': 1000.0 / float(np.mean(times)),
        }


def benchmark_model(inference: QuantizedACTInference, num_iterations: int = 1000):
    """Benchmark inference performance"""
    logger.info(f"🚀 Benchmarking inference over {num_iterations} iterations...")
    
    # Create dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (inference.image_size[0], inference.image_size[1], 3), dtype=np.uint8)
    )
    
    # Warmup
    logger.info("Warming up...")
    for _ in range(10):
        _ = inference.predict(dummy_image, use_action_chunking=False)
    
    # Reset timing
    inference.reset()
    
    # Benchmark
    logger.info("Running benchmark...")
    for i in range(num_iterations):
        _ = inference.predict(dummy_image, use_action_chunking=False)
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{num_iterations}")
    
    # Get statistics
    stats = inference.get_latency_stats()
    
    logger.info("=" * 80)
    logger.info("📊 INFERENCE LATENCY BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Mean latency:  {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
    logger.info(f"Min latency:   {stats['min_ms']:.2f} ms")
    logger.info(f"Max latency:   {stats['max_ms']:.2f} ms")
    logger.info(f"P50 latency:   {stats['p50_ms']:.2f} ms")
    logger.info(f"P95 latency:   {stats['p95_ms']:.2f} ms")
    logger.info(f"P99 latency:   {stats['p99_ms']:.2f} ms")
    logger.info(f"Throughput:    {stats['fps']:.1f} FPS")
    logger.info("=" * 80)
    
    # Control frequency analysis
    logger.info("")
    logger.info("🎮 CONTROL FREQUENCY ANALYSIS")
    logger.info("=" * 80)
    
    if stats['p95_ms'] < 10:
        logger.info("✅ EXCELLENT: Can achieve 100Hz control (< 10ms P95 latency)")
    elif stats['p95_ms'] < 20:
        logger.info("✅ GOOD: Can achieve 50Hz control (< 20ms P95 latency)")
    elif stats['p95_ms'] < 33:
        logger.info("✅ ACCEPTABLE: Can achieve 30Hz control (< 33ms P95 latency)")
    else:
        logger.info("⚠️  SLOW: < 30Hz control - may need further optimization")
    
    logger.info("=" * 80)
    
    return stats


def test_single_image(inference: QuantizedACTInference, image_path: str):
    """Test inference on a single image"""
    logger.info(f"Testing inference on {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    logger.info(f"Loaded image: {image.size}")
    
    # Run inference
    steering, throttle = inference.predict(image)
    
    logger.info("=" * 80)
    logger.info("🎯 PREDICTION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Steering: {steering:+.4f} (range: -1.0 to +1.0)")
    logger.info(f"Throttle: {throttle:+.4f} (range: -1.0 to +1.0)")
    logger.info("=" * 80)
    
    # Interpret commands
    logger.info("")
    logger.info("📋 INTERPRETATION")
    logger.info("=" * 80)
    
    # Steering
    if abs(steering) < 0.1:
        steer_cmd = "STRAIGHT"
    elif steering > 0:
        steer_cmd = f"RIGHT ({steering * 100:.1f}%)"
    else:
        steer_cmd = f"LEFT ({abs(steering) * 100:.1f}%)"
    
    # Throttle
    if abs(throttle) < 0.1:
        throttle_cmd = "STOPPED"
    elif throttle > 0:
        throttle_cmd = f"FORWARD ({throttle * 100:.1f}%)"
    else:
        throttle_cmd = f"REVERSE ({abs(throttle) * 100:.1f}%)"
    
    logger.info(f"Steering: {steer_cmd}")
    logger.info(f"Throttle: {throttle_cmd}")
    logger.info("=" * 80)
    
    # Show latency
    stats = inference.get_latency_stats()
    if stats:
        logger.info(f"\n⏱️  Inference time: {stats['mean_ms']:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description='Quantized ACT Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to quantized model')
    parser.add_argument('--test_image', type=str, help='Test with single image')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--num_iterations', type=int, default=1000, help='Benchmark iterations')
    parser.add_argument('--image_size', type=int, nargs=2, default=[360, 640],
                       help='Image size (height width)')
    
    args = parser.parse_args()
    
    # Create inference engine
    inference = QuantizedACTInference(
        checkpoint_path=args.checkpoint,
        device='cpu',
        image_size=tuple(args.image_size),
    )
    
    # Test with image if provided
    if args.test_image:
        test_single_image(inference, args.test_image)
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_model(inference, args.num_iterations)
    
    # If neither test nor benchmark, show usage
    if not args.test_image and not args.benchmark:
        logger.info("No action specified. Use --test_image or --benchmark")
        logger.info("")
        logger.info("Examples:")
        logger.info("  Test image:  python lerobot_act_inference_quantized.py --checkpoint model.pth --test_image test.jpg")
        logger.info("  Benchmark:   python lerobot_act_inference_quantized.py --checkpoint model.pth --benchmark")


if __name__ == "__main__":
    main()
