#!/usr/bin/env python3
"""
Raspberry Pi 5 Optimized ACT Inference

Optimized for ARM architecture with:
- QNNPACK backend for ARM CPUs
- Multi-threading optimization
- Memory management
- Low-latency control loop

Usage:
    # Benchmark mode
    python lerobot_act_inference_rpi5.py \
        --checkpoint models/best_model_quantized.pth \
        --benchmark

    # Real-time control mode
    python lerobot_act_inference_rpi5.py \
        --checkpoint models/best_model_quantized.pth \
        --camera_id 0 \
        --arduino_port /dev/ttyUSB0 \
        --control_freq 30
"""

import sys
import torch
import torch.backends.quantized
import numpy as np
from pathlib import Path
from PIL import Image
import logging
import argparse
import time
from typing import Tuple, Optional, Dict
from collections import deque
import cv2
import serial

# Setup LeRobot imports
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def optimize_for_raspberry_pi():
    """Apply Raspberry Pi 5 specific optimizations"""
    
    # Use ARM-optimized QNNPACK backend for quantized operations
    torch.backends.quantized.engine = 'qnnpack'
    logger.info("✅ Set quantization backend to QNNPACK (ARM optimized)")
    
    # Set number of threads (Pi 5 has 4 Cortex-A76 cores)
    torch.set_num_threads(4)
    logger.info("✅ Set PyTorch threads to 4 (matching Pi 5 cores)")
    
    # Enable flush denormal for better performance
    torch.set_flush_denormal(True)
    
    # Disable gradient computation globally (inference only)
    torch.set_grad_enabled(False)
    
    logger.info("✅ Raspberry Pi optimizations applied")


class RPi5ACTController:
    """Optimized ACT controller for Raspberry Pi 5"""
    
    def __init__(
        self,
        checkpoint_path: str,
        image_size: Tuple[int, int] = (360, 640),
        control_freq: int = 30,
    ):
        """
        Initialize controller
        
        Args:
            checkpoint_path: Path to quantized model
            image_size: Camera resolution (H, W)
            control_freq: Target control frequency in Hz
        """
        self.image_size = image_size
        self.control_freq = control_freq
        self.control_period = 1.0 / control_freq
        
        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check quantization mode
        if 'quantization_mode' in checkpoint:
            logger.info(f"✅ Quantized model detected: {checkpoint['quantization_mode']}")
        
        # Create config
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            self.config = ACTConfig(
                input_shapes={
                    "observation.images.cam_front": [3, image_size[0], image_size[1]],
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
        
        # Try to load full quantized model first (new format)
        if 'model' in checkpoint:
            logger.info("Loading full quantized model object...")
            self.policy = checkpoint['model']
            self.policy.eval()
        else:
            # Fallback: load from state_dict (old format)
            logger.info("Loading from state_dict (compatibility mode)...")
            self.policy = ACTPolicy(self.config)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.policy.eval()
        
        # State tracking
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)
        self.action_queue = deque(maxlen=self.config.chunk_size)
        
        # Performance monitoring
        self.latencies = deque(maxlen=1000)
        self.frame_times = deque(maxlen=1000)
        
        # Image preprocessing (simplified for speed)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Image size: {image_size}")
        logger.info(f"   Target control frequency: {control_freq} Hz")
        logger.info(f"   Chunk size: {self.config.chunk_size}")
    
    def preprocess_image_fast(self, frame: np.ndarray) -> torch.Tensor:
        """Fast image preprocessing optimized for Pi 5"""
        # Resize if needed (use cv2, faster than PIL on ARM)
        if frame.shape[:2] != self.image_size:
            frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
        
        # Convert to tensor (BGR to RGB, HWC to CHW, normalize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        frame = (frame - self.mean) / self.std
        
        return frame.unsqueeze(0)  # Add batch dimension
    
    def predict(self, frame: np.ndarray, use_chunking: bool = True) -> Tuple[float, float]:
        """
        Predict control actions
        
        Args:
            frame: Camera frame (numpy array, BGR format)
            use_chunking: Use action chunking for smoother control
            
        Returns:
            (steering, throttle) in range [-1, 1]
        """
        start_time = time.perf_counter()
        
        # Use queued action if available
        if use_chunking and len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            latency = (time.perf_counter() - start_time) * 1000
            self.latencies.append(latency)
            return tuple(action)
        
        # Preprocess image
        image_tensor = self.preprocess_image_fast(frame)
        state_tensor = torch.from_numpy(self.current_state).unsqueeze(0)
        
        # Prepare batch
        batch = {
            "observation.images.cam_front": image_tensor,
            "observation.state": state_tensor,
        }
        
        # Inference
        output = self.policy(batch)
        actions = output.squeeze(0).cpu().numpy()
        
        # Queue actions
        if use_chunking:
            for action in actions[1:]:
                self.action_queue.append(action)
            action = actions[0]
        else:
            action = actions[0]
        
        # Update state
        self.current_state = action
        
        # Track latency
        latency = (time.perf_counter() - start_time) * 1000
        self.latencies.append(latency)
        
        # Clamp output
        steering = np.clip(action[0], -1.0, 1.0)
        throttle = np.clip(action[1], -1.0, 1.0)
        
        return float(steering), float(throttle)
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if len(self.latencies) == 0:
            return {}
        
        latencies = np.array(self.latencies)
        
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'fps': 1000.0 / float(np.mean(latencies)),
        }
    
    def reset(self):
        """Reset controller state"""
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)
        self.action_queue.clear()


def denormalize_pwm(steering: float, throttle: float) -> Tuple[int, int]:
    """
    Convert normalized actions to PWM microseconds
    
    Args:
        steering: -1.0 (left) to +1.0 (right)
        throttle: -1.0 (reverse) to +1.0 (forward)
        
    Returns:
        (steering_us, throttle_us) PWM values
    """
    # Steering: 1008us (left) to 1948us (right), center ~1478us
    steering_center = 1478
    steering_range = 470  # ±470us from center
    steering_us = int(steering_center + steering * steering_range)
    
    # Throttle: 0us (stopped) to 948us (full), neutral ~120us
    throttle_min = 120  # Stopped
    throttle_max = 948  # Full throttle
    # Map 0 to 1 range (only forward, no reverse for safety)
    throttle_normalized = (throttle + 1.0) / 2.0  # -1..1 -> 0..1
    throttle_us = int(throttle_min + throttle_normalized * (throttle_max - throttle_min))
    
    return steering_us, throttle_us


def benchmark_mode(controller: RPi5ACTController, num_iterations: int = 1000):
    """Run benchmark with dummy frames"""
    logger.info(f"🚀 Running benchmark ({num_iterations} iterations)...")
    
    # Create dummy frame
    dummy_frame = np.random.randint(
        0, 255, 
        (controller.image_size[0], controller.image_size[1], 3), 
        dtype=np.uint8
    )
    
    # Warmup
    logger.info("Warming up...")
    for _ in range(20):
        _ = controller.predict(dummy_frame, use_chunking=False)
    
    controller.reset()
    
    # Benchmark
    logger.info("Running benchmark...")
    for i in range(num_iterations):
        _ = controller.predict(dummy_frame, use_chunking=False)
        if (i + 1) % 200 == 0:
            logger.info(f"  Progress: {i + 1}/{num_iterations}")
    
    # Results
    stats = controller.get_stats()
    
    logger.info("=" * 80)
    logger.info("📊 RASPBERRY PI 5 INFERENCE BENCHMARK")
    logger.info("=" * 80)
    logger.info(f"Mean latency:  {stats['mean_latency_ms']:.2f} ± {stats['std_latency_ms']:.2f} ms")
    logger.info(f"Min latency:   {stats['min_latency_ms']:.2f} ms")
    logger.info(f"Max latency:   {stats['max_latency_ms']:.2f} ms")
    logger.info(f"P95 latency:   {stats['p95_latency_ms']:.2f} ms")
    logger.info(f"P99 latency:   {stats['p99_latency_ms']:.2f} ms")
    logger.info(f"Throughput:    {stats['fps']:.1f} FPS")
    logger.info("=" * 80)
    
    # Control frequency analysis
    logger.info("")
    logger.info("🎮 CONTROL FREQUENCY CAPABILITY")
    logger.info("=" * 80)
    
    p95 = stats['p95_latency_ms']
    if p95 < 10:
        logger.info("✅ Can achieve 100Hz control loop")
    elif p95 < 20:
        logger.info("✅ Can achieve 50Hz control loop")
    elif p95 < 33:
        logger.info("✅ Can achieve 30Hz control loop")
    else:
        logger.info("⚠️  Limited to <30Hz control loop")
    
    logger.info(f"   Recommended: {int(1000 / (p95 * 1.2))} Hz")
    logger.info("=" * 80)


def camera_mode(
    controller: RPi5ACTController,
    camera_id: int,
    arduino_port: Optional[str] = None,
    display: bool = True,
):
    """Real-time control mode with camera"""
    logger.info(f"📷 Starting camera mode (device {camera_id})")
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, controller.image_size[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, controller.image_size[0])
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        logger.error("❌ Failed to open camera")
        return
    
    logger.info("✅ Camera opened")
    
    # Open Arduino if provided
    arduino = None
    if arduino_port:
        try:
            arduino = serial.Serial(arduino_port, 115200, timeout=0.01)
            time.sleep(2)  # Wait for Arduino reset
            logger.info(f"✅ Arduino connected on {arduino_port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Arduino: {e}")
            arduino = None
    
    logger.info("🎮 Starting control loop (press 'q' to quit)...")
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️  Failed to read frame")
                continue
            
            # Predict
            steering, throttle = controller.predict(frame, use_chunking=True)
            
            # Send to Arduino
            if arduino:
                steering_us, throttle_us = denormalize_pwm(steering, throttle)
                command = f"S{steering_us}T{throttle_us}\n"
                arduino.write(command.encode())
            
            # Display
            if display:
                # Draw overlay
                h, w = frame.shape[:2]
                
                # Steering indicator
                center_x = w // 2
                steer_x = int(center_x + steering * (w // 4))
                cv2.line(frame, (center_x, h - 50), (steer_x, h - 50), (0, 255, 0), 3)
                cv2.circle(frame, (steer_x, h - 50), 5, (0, 255, 0), -1)
                
                # Throttle bar
                throttle_height = int((throttle + 1) * 40)  # 0 to 80 pixels
                cv2.rectangle(frame, (10, h - 80), (30, h - 80 + throttle_height), (0, 255, 0), -1)
                
                # Stats
                stats = controller.get_stats()
                if stats:
                    cv2.putText(frame, f"Latency: {stats['mean_latency_ms']:.1f}ms", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"FPS: {stats['fps']:.1f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.putText(frame, f"Steer: {steering:+.3f}", 
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Throttle: {throttle:+.3f}", 
                           (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('ACT Control', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Maintain control frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = controller.control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    finally:
        cap.release()
        if arduino:
            arduino.close()
        cv2.destroyAllWindows()
        
        # Print final stats
        stats = controller.get_stats()
        if stats:
            logger.info("")
            logger.info("📊 Session Statistics:")
            logger.info(f"   Average latency: {stats['mean_latency_ms']:.2f} ms")
            logger.info(f"   P95 latency: {stats['p95_latency_ms']:.2f} ms")
            logger.info(f"   Average FPS: {stats['fps']:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi 5 ACT Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Quantized model path')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark mode')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--arduino_port', type=str, help='Arduino serial port (e.g., /dev/ttyUSB0)')
    parser.add_argument('--control_freq', type=int, default=30, help='Control frequency (Hz)')
    parser.add_argument('--no_display', action='store_true', help='Disable video display')
    parser.add_argument('--iterations', type=int, default=1000, help='Benchmark iterations')
    
    args = parser.parse_args()
    
    # Apply Pi optimizations
    optimize_for_raspberry_pi()
    
    # Create controller
    controller = RPi5ACTController(
        checkpoint_path=args.checkpoint,
        image_size=(360, 640),
        control_freq=args.control_freq,
    )
    
    # Run mode
    if args.benchmark:
        benchmark_mode(controller, args.iterations)
    else:
        camera_mode(
            controller,
            args.camera_id,
            args.arduino_port,
            display=not args.no_display,
        )


if __name__ == "__main__":
    main()
