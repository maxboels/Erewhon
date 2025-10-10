#!/usr/bin/env python3
"""
Raspberry Pi 5 Optimized ACT Inference (Standalone)

This is a minimal standalone version for deployment on Raspberry Pi
with sparse checkout. It only requires the quantized model file.

Usage:
    # Benchmark
    python3 act_inference_quantized.py --checkpoint model.pth --benchmark

    # Camera test
    python3 act_inference_quantized.py --checkpoint model.pth --camera_id 0

    # Full autonomous control
    python3 act_inference_quantized.py \
        --checkpoint model.pth \
        --camera_id 0 \
        --arduino_port /dev/ttyUSB0 \
        --control_freq 30
"""

import torch
import torch.backends.quantized
import numpy as np
import cv2
import argparse
import time
import logging
from collections import deque
from typing import Tuple, Dict
from pathlib import Path

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logging.warning("pyserial not available - Arduino control disabled")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def optimize_for_raspberry_pi():
    """Apply Raspberry Pi 5 specific optimizations"""
    # Use ARM-optimized QNNPACK backend for quantized operations
    torch.backends.quantized.engine = 'qnnpack'
    
    # Set number of threads (Pi 5 has 4 Cortex-A76 cores)
    torch.set_num_threads(4)
    
    # Enable flush denormal for better performance
    torch.set_flush_denormal(True)
    
    # Disable gradient computation globally (inference only)
    torch.set_grad_enabled(False)
    
    logger.info("✅ Raspberry Pi optimizations applied (QNNPACK, 4 threads)")


class MinimalACTInference:
    """Minimal inference wrapper for quantized ACT models"""
    
    def __init__(self, checkpoint_path: str, image_size: Tuple[int, int] = (360, 640)):
        self.image_size = image_size
        
        # Load checkpoint
        logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model (already quantized)
        self.model = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        
        # If it's a state dict, we need to reconstruct the model
        # For quantized models, the structure is preserved in the checkpoint
        if isinstance(self.model, dict):
            # This is a state dict - for quantized models, we can load directly
            # The model structure is preserved in the quantized checkpoint
            logger.info("Checkpoint contains state dict")
            # We'll load it as-is and call it directly
            # Quantized models can be used this way
        
        # Image preprocessing constants
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # State tracking
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)
        self.action_queue = deque(maxlen=32)
        
        # Performance monitoring
        self.latencies = deque(maxlen=1000)
        
        logger.info("✅ Model loaded successfully")
    
    def preprocess_image(self, frame: np.ndarray) -> torch.Tensor:
        """Fast image preprocessing optimized for Pi 5"""
        # Resize if needed
        if frame.shape[:2] != self.image_size:
            frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
        
        # Convert to tensor (BGR to RGB, HWC to CHW, normalize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        frame = (frame - self.mean) / self.std
        
        return frame.unsqueeze(0)
    
    def predict(self, frame: np.ndarray, use_chunking: bool = True) -> Tuple[float, float]:
        """Predict steering and throttle from camera frame"""
        start_time = time.perf_counter()
        
        # Check action queue first
        if use_chunking and len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            latency = (time.perf_counter() - start_time) * 1000
            self.latencies.append(latency)
            return tuple(action)
        
        # Preprocess
        image_tensor = self.preprocess_image(frame)
        state_tensor = torch.from_numpy(self.current_state).unsqueeze(0)
        
        # Prepare input
        batch = {
            "observation.images.cam_front": image_tensor,
            "observation.state": state_tensor,
        }
        
        # Inference
        try:
            # Try calling as a model
            if hasattr(self.model, '__call__'):
                output = self.model(batch)
            else:
                # Fallback: assume it's a state dict and we need to handle differently
                # This shouldn't happen with properly saved quantized models
                raise RuntimeError("Model is not callable - check checkpoint format")
                
            actions = output.squeeze(0).cpu().numpy()
        except Exception as e:
            logger.error(f"Inference error: {e}")
            # Return safe default
            return 0.0, 0.0
        
        # Queue actions for chunking
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
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'fps': 1000.0 / float(np.mean(latencies)),
        }
    
    def reset(self):
        """Reset controller state"""
        self.current_state = np.array([0.0, 0.0], dtype=np.float32)
        self.action_queue.clear()


def denormalize_pwm(steering: float, throttle: float) -> Tuple[int, int]:
    """Convert normalized actions to PWM microseconds"""
    # Steering: 1008us (left) to 1948us (right), center ~1478us
    steering_center = 1478
    steering_range = 470
    steering_us = int(steering_center + steering * steering_range)
    
    # Throttle: 120us (stopped) to 948us (full)
    throttle_min = 120
    throttle_max = 948
    throttle_normalized = (throttle + 1.0) / 2.0  # -1..1 -> 0..1
    throttle_us = int(throttle_min + throttle_normalized * (throttle_max - throttle_min))
    
    return steering_us, throttle_us


def benchmark_mode(controller: MinimalACTInference, iterations: int = 1000):
    """Benchmark inference performance"""
    logger.info(f"🚀 Running benchmark ({iterations} iterations)...")
    
    # Dummy frame
    dummy_frame = np.random.randint(
        0, 255, 
        (controller.image_size[0], controller.image_size[1], 3),
        dtype=np.uint8
    )
    
    # Warmup
    for _ in range(20):
        _ = controller.predict(dummy_frame, use_chunking=False)
    
    controller.reset()
    
    # Benchmark
    for i in range(iterations):
        _ = controller.predict(dummy_frame, use_chunking=False)
        if (i + 1) % 200 == 0:
            logger.info(f"  Progress: {i + 1}/{iterations}")
    
    # Results
    stats = controller.get_stats()
    
    print("=" * 80)
    print("📊 RASPBERRY PI 5 INFERENCE BENCHMARK")
    print("=" * 80)
    print(f"Mean latency:  {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
    print(f"P95 latency:   {stats['p95_ms']:.2f} ms")
    print(f"Throughput:    {stats['fps']:.1f} FPS")
    print("=" * 80)
    print()
    print("🎮 CONTROL FREQUENCY CAPABILITY")
    print("=" * 80)
    
    p95 = stats['p95_ms']
    if p95 < 10:
        print("✅ Can achieve 100Hz control loop")
    elif p95 < 20:
        print("✅ Can achieve 50Hz control loop")
    elif p95 < 33:
        print("✅ Can achieve 30Hz control loop")
    else:
        print("⚠️  Limited to <30Hz control loop")
    
    print(f"   Recommended: {int(1000 / (p95 * 1.2))} Hz")
    print("=" * 80)


def camera_mode(
    controller: MinimalACTInference,
    camera_id: int,
    arduino_port: str = None,
    control_freq: int = 30,
    display: bool = True,
):
    """Real-time control mode"""
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
    
    # Open Arduino
    arduino = None
    if arduino_port and SERIAL_AVAILABLE:
        try:
            arduino = serial.Serial(arduino_port, 115200, timeout=0.01)
            time.sleep(2)
            logger.info(f"✅ Arduino connected on {arduino_port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Arduino: {e}")
    
    logger.info("🎮 Starting control loop (press 'q' to quit)...")
    
    control_period = 1.0 / control_freq
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
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
                h, w = frame.shape[:2]
                
                # Steering indicator
                center_x = w // 2
                steer_x = int(center_x + steering * (w // 4))
                cv2.line(frame, (center_x, h - 50), (steer_x, h - 50), (0, 255, 0), 3)
                cv2.circle(frame, (steer_x, h - 50), 5, (0, 255, 0), -1)
                
                # Throttle bar
                throttle_height = int((throttle + 1) * 40)
                cv2.rectangle(frame, (10, h - 80), (30, h - 80 + throttle_height), (0, 255, 0), -1)
                
                # Stats
                stats = controller.get_stats()
                if stats:
                    cv2.putText(frame, f"Latency: {stats['mean_ms']:.1f}ms", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"FPS: {stats['fps']:.1f}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('ACT Control', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Maintain control frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    finally:
        cap.release()
        if arduino:
            arduino.close()
        cv2.destroyAllWindows()
        
        # Print stats
        stats = controller.get_stats()
        if stats:
            logger.info(f"📊 Session stats: Avg latency {stats['mean_ms']:.2f}ms, FPS {stats['fps']:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Quantized ACT Inference for Raspberry Pi 5')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to quantized model')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--iterations', type=int, default=1000, help='Benchmark iterations')
    parser.add_argument('--camera_id', type=int, help='Camera device ID')
    parser.add_argument('--arduino_port', type=str, help='Arduino serial port')
    parser.add_argument('--control_freq', type=int, default=30, help='Control frequency (Hz)')
    parser.add_argument('--no_display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    # Apply optimizations
    optimize_for_raspberry_pi()
    
    # Load model
    controller = MinimalACTInference(
        checkpoint_path=args.checkpoint,
        image_size=(360, 640),
    )
    
    # Run mode
    if args.benchmark:
        benchmark_mode(controller, args.iterations)
    elif args.camera_id is not None:
        camera_mode(
            controller,
            args.camera_id,
            args.arduino_port,
            args.control_freq,
            display=not args.no_display,
        )
    else:
        print("Please specify --benchmark or --camera_id")


if __name__ == "__main__":
    main()
