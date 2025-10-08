#!/usr/bin/env python3
"""
LeRobot ACT Inference for RC Car
Uses the official LeRobot ACT implementation for real-time control
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import Tuple, Optional
from torchvision import transforms

# Setup LeRobot imports
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LeRobotACTInference:
    """Inference wrapper for LeRobot ACT policy on RC car"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        image_size: Tuple[int, int] = (360, 640),  # (H, W) - actual data resolution
    ):
        """
        Initialize LeRobot ACT inference
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
            image_size: Input image size (height, width)
        """
        self.device = device
        self.image_size = image_size
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create ACT configuration (must match training config)
        self.config = ACTConfig(
            input_shapes={
                "observation.images.cam_front": [3, image_size[0], image_size[1]],
                "observation.state": [2],  # [steering, throttle]
            },
            output_shapes={
                "action": [2],  # [steering, throttle] PWM
            },
            n_obs_steps=1,
            chunk_size=32,
            n_action_steps=32,
            vision_backbone="resnet18",
            pretrained_backbone_weights=None,  # Not needed for inference
            use_vae=True,
            latent_dim=32,
            device=device,
        )
        
        # Create policy
        self.policy = ACTPolicy(self.config)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.to(device)
        self.policy.eval()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),  # Resize to training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # State tracking
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)  # [steering, throttle]
        self.action_queue = []  # For action chunking
        
        logger.info(f"✅ LeRobot ACT model loaded successfully")
        logger.info(f"   Device: {device}")
        logger.info(f"   Image size: {image_size}")
        logger.info(f"   Chunk size: {self.config.chunk_size}")
        
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference"""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.image_transform(image).unsqueeze(0)  # Add batch dimension
    
    def predict(
        self,
        image: Image.Image,
        current_state: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Predict action from current observation
        
        Args:
            image: Current camera observation (PIL Image or numpy array)
            current_state: Current robot state [steering, throttle] (optional)
        
        Returns:
            steering: Predicted steering PWM value
            throttle: Predicted throttle PWM value
        """
        # Update current state if provided
        if current_state is not None:
            self.current_state = current_state
        
        # If action queue is empty, get new action chunk
        if len(self.action_queue) == 0:
            with torch.no_grad():
                # Preprocess image
                image_tensor = self.preprocess_image(image).to(self.device)
                
                # Prepare state
                state_tensor = torch.tensor(
                    self.current_state, 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)  # Add batch dimension
                
                # Create batch
                batch = {
                    "observation.images.cam_front": image_tensor,  # [1, 3, 360, 640]
                    "observation.state": state_tensor,             # [1, 2]
                }
                
                # Predict action chunk
                actions = self.policy.select_action(batch)  # Uses internal action queue
                
                # Convert to numpy
                action = actions.cpu().numpy().squeeze()  # Remove batch dimension
                
                # Update current state with predicted action
                self.current_state = action
                
                return float(action[0]), float(action[1])
        
        # Return next action from queue (if using manual queue management)
        else:
            action = self.action_queue.pop(0)
            self.current_state = action
            return float(action[0]), float(action[1])
    
    def reset(self):
        """Reset internal state"""
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)
        self.action_queue = []
        self.policy.reset()  # Reset LeRobot's internal action queue
        logger.info("Inference state reset")


def main():
    """Test inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LeRobot ACT Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_image', type=str, help='Path to test image (optional)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Create inference engine
    inference = LeRobotACTInference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Test with image
    if args.test_image:
        logger.info(f"Testing with image: {args.test_image}")
        test_image = Image.open(args.test_image).convert('RGB')
        
        steering, throttle = inference.predict(test_image)
        
        logger.info(f"Predicted action:")
        logger.info(f"  Steering: {steering:.4f}")
        logger.info(f"  Throttle: {throttle:.4f}")
    else:
        # Test with dummy image
        logger.info("Testing with dummy image (640x360)")
        dummy_image = Image.new('RGB', (640, 360), color=(128, 128, 128))
        
        steering, throttle = inference.predict(dummy_image)
        
        logger.info(f"Predicted action:")
        logger.info(f"  Steering: {steering:.4f}")
        logger.info(f"  Throttle: {throttle:.4f}")


if __name__ == "__main__":
    main()
