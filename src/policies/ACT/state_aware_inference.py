#!/usr/bin/env python3
"""
State-Aware ACT Inference Script for RC Car
Runs inference using BOTH camera frames AND current state
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the state-aware model
from state_aware_act_trainer import StateAwareACTModel


class StateAwareInference:
    """Inference engine that uses both visual observations and current state"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        
        # Performance tracking
        self.inference_times = []
        
        # Load model
        self.load_model()
        
        # Setup preprocessing
        self.setup_preprocessing()
        
        # Initialize state
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)  # [steering, throttle]
        
        logger.info(f"State-aware inference engine initialized on {self.device}")
    
    def load_model(self):
        """Load the trained state-aware model"""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Create model
        self.model = StateAwareACTModel(
            image_size=(480, 640),
            state_dim=2,
            action_dim=2,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            chunk_size=config['chunk_size']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"Model expects: Image + Current State [steering, throttle]")
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
    
    def setup_preprocessing(self):
        """Setup image preprocessing pipeline"""
        self.image_transforms = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """Preprocess camera image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image_tensor = self.image_transforms(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor
    
    def predict(self, image, current_state=None):
        """
        Run inference on image with current state
        
        Args:
            image: Camera image (numpy array or PIL Image)
            current_state: Optional current state [steering, throttle]. 
                          If None, uses internally tracked state.
        
        Returns:
            dict with predicted actions and metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Use provided state or internal state
            if current_state is not None:
                state = np.array(current_state, dtype=np.float32)
            else:
                state = self.current_state
            
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                action_chunk = self.model(image_tensor, state_tensor)  # [1, chunk_size, 2]
            
            # Take first action from chunk
            predicted_action = action_chunk[0, 0, :].cpu().numpy()
            
            # Update internal state
            self.current_state = predicted_action
            
            # Performance tracking
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return {
                'steering': float(predicted_action[0]),
                'throttle': float(predicted_action[1]),
                'input_state': state.tolist(),
                'inference_time_ms': inference_time * 1000,
                'action_chunk': action_chunk.cpu().numpy()
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                'steering': 0.0,
                'throttle': 0.0,
                'input_state': [0.0, 0.0],
                'inference_time_ms': 0.0,
                'error': str(e)
            }
    
    def reset_state(self):
        """Reset internal state to neutral"""
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)
        logger.info("State reset to neutral [0.0, 0.0]")
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        return {
            'avg_inference_time_ms': float(np.mean(times) * 1000),
            'max_inference_time_ms': float(np.max(times) * 1000),
            'min_inference_time_ms': float(np.min(times) * 1000),
            'avg_fps': float(1.0 / np.mean(times)),
            'total_inferences': len(self.inference_times)
        }


def test_on_episode_data(inference_engine, episode_dir: str, num_samples: int = 10):
    """Test inference on episode data"""
    import json
    
    episode_path = Path(episode_dir)
    
    logger.info(f"Testing on episode: {episode_path.name}")
    
    # Load episode data
    with open(episode_path / "episode_data.json", 'r') as f:
        episode_data = json.load(f)
    
    frame_samples = episode_data['frame_samples'][:num_samples]
    control_samples = episode_data['control_samples']
    
    logger.info(f"Testing on {len(frame_samples)} frames")
    logger.info("=" * 80)
    
    # Reset state to neutral
    inference_engine.reset_state()
    
    total_steering_error = 0
    total_throttle_error = 0
    
    for i, frame_sample in enumerate(frame_samples):
        # Load frame
        frame_path = episode_path / frame_sample['image_path']
        image = Image.open(frame_path).convert('RGB')
        
        # Find closest control
        frame_time = frame_sample['timestamp']
        closest_control = min(control_samples, 
                            key=lambda c: abs(c['system_timestamp'] - frame_time))
        
        # Get current state from ground truth
        current_state = [
            closest_control['steering_normalized'],
            closest_control['throttle_normalized']
        ]
        
        # Run inference WITH current state
        result = inference_engine.predict(image, current_state)
        
        # Calculate errors
        steering_error = abs(result['steering'] - closest_control['steering_normalized'])
        throttle_error = abs(result['throttle'] - closest_control['throttle_normalized'])
        
        total_steering_error += steering_error
        total_throttle_error += throttle_error
        
        logger.info(f"Frame {i+1}:")
        logger.info(f"  Input State: S={current_state[0]:.4f}, T={current_state[1]:.4f}")
        logger.info(f"  Predicted:   S={result['steering']:.4f}, T={result['throttle']:.4f}")
        logger.info(f"  Ground Truth: S={closest_control['steering_normalized']:.4f}, T={closest_control['throttle_normalized']:.4f}")
        logger.info(f"  Error: S={steering_error:.4f}, T={throttle_error:.4f}")
        logger.info(f"  Inference: {result['inference_time_ms']:.2f}ms")
        logger.info("")
    
    # Summary
    avg_steering_error = total_steering_error / len(frame_samples)
    avg_throttle_error = total_throttle_error / len(frame_samples)
    
    logger.info("=" * 80)
    logger.info(f"Average Errors:")
    logger.info(f"  Steering: {avg_steering_error:.4f}")
    logger.info(f"  Throttle: {avg_throttle_error:.4f}")
    
    # Performance stats
    stats = inference_engine.get_performance_stats()
    logger.info(f"Performance:")
    logger.info(f"  Avg inference: {stats['avg_inference_time_ms']:.2f}ms")
    logger.info(f"  Avg FPS: {stats['avg_fps']:.1f}")


def main():
    parser = argparse.ArgumentParser(description='State-aware ACT inference for RC car')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episode', type=str, help='Path to test episode')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--num_samples', type=int, default=10, 
                       help='Number of samples to test')
    
    args = parser.parse_args()
    
    logger.info("🤖 State-Aware ACT Inference Test")
    logger.info("=" * 80)
    logger.info("This model uses BOTH:")
    logger.info("  ✅ Camera observations (640x480)")
    logger.info("  ✅ Current state [steering, throttle]")
    logger.info("=" * 80)
    
    # Load model
    inference_engine = StateAwareInference(args.model, args.device)
    
    # Test on episode if provided
    if args.episode:
        test_on_episode_data(inference_engine, args.episode, args.num_samples)
    else:
        logger.info("No test episode provided. Use --episode to test on data.")
        logger.info("\nExample usage:")
        logger.info("  python state_aware_inference.py \\")
        logger.info("    --model outputs/state_aware_act_XXX/best_model.pth \\")
        logger.info("    --episode episodes/episode_20251007_144013 \\")
        logger.info("    --device cuda")


if __name__ == "__main__":
    main()
