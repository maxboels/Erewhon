#!/usr/bin/env python3
"""
Export ACT Model to ONNX Format

Converts trained PyTorch ACT model to ONNX for:
- Cross-platform deployment
- Hardware accelerators (Hailo NPU, TensorRT, etc.)
- Optimized inference runtimes

Usage:
    # Basic export
    python export_act_to_onnx.py \
        --checkpoint outputs/lerobot_act/best_model.pth \
        --output act_model.onnx

    # With validation
    python export_act_to_onnx.py \
        --checkpoint outputs/lerobot_act/best_model.pth \
        --output act_model.onnx \
        --validate \
        --test_data src/robots/rover/episodes
"""

import sys
import torch
import torch.onnx
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

# Setup LeRobot imports
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

# Setup dataset loader for validation
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "datasets"))
from local_dataset_loader import TracerLocalDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ACTONNXExporter:
    """Exports ACT model to ONNX format"""
    
    def __init__(
        self,
        checkpoint_path: str,
        image_size: Tuple[int, int] = (360, 640),
    ):
        """
        Initialize exporter
        
        Args:
            checkpoint_path: Path to trained PyTorch model
            image_size: Input image size (height, width)
        """
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        
        logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create ACT configuration
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
        
        # Load model
        self.model = ACTPolicy(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to('cpu')
        self.model.eval()
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Image size: {image_size}")
        logger.info(f"   Chunk size: {self.config.chunk_size}")
    
    def export_to_onnx(
        self,
        output_path: str,
        opset_version: int = 11,
        dynamic_batch: bool = True,
    ):
        """
        Export model to ONNX format
        
        Args:
            output_path: Where to save ONNX model
            opset_version: ONNX opset version (11 recommended for compatibility)
            dynamic_batch: Enable dynamic batch size
        """
        logger.info("🔧 Exporting to ONNX...")
        
        # Create dummy inputs
        dummy_image = torch.randn(1, 3, self.image_size[0], self.image_size[1])
        dummy_state = torch.randn(1, 2)
        
        dummy_input = {
            "observation.images.cam_front": dummy_image,
            "observation.state": dummy_state,
        }
        
        # Define input/output names
        input_names = ['image', 'state']
        output_names = ['actions']
        
        # Define dynamic axes if needed
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                'image': {0: 'batch_size'},
                'state': {0: 'batch_size'},
                'actions': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
        )
        
        # Get file size
        onnx_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        pytorch_size = Path(self.checkpoint_path).stat().st_size / (1024 * 1024)
        
        logger.info(f"✅ ONNX export successful!")
        logger.info(f"   Output: {output_path}")
        logger.info(f"   PyTorch size: {pytorch_size:.2f} MB")
        logger.info(f"   ONNX size: {onnx_size:.2f} MB")
        logger.info(f"   Opset version: {opset_version}")
        logger.info(f"   Dynamic batch: {dynamic_batch}")
    
    def validate_onnx(self, onnx_path: str, test_data_dir: str = None):
        """
        Validate ONNX model matches PyTorch model
        
        Args:
            onnx_path: Path to ONNX model
            test_data_dir: Optional test dataset for validation
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("❌ onnxruntime not installed. Install with: pip install onnxruntime")
            return
        
        logger.info("🔍 Validating ONNX model...")
        
        # Load ONNX model
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input/output info
        input_names = [i.name for i in ort_session.get_inputs()]
        output_names = [o.name for o in ort_session.get_outputs()]
        
        logger.info(f"   Inputs: {input_names}")
        logger.info(f"   Outputs: {output_names}")
        
        # Test with dummy data
        dummy_image = np.random.randn(1, 3, self.image_size[0], self.image_size[1]).astype(np.float32)
        dummy_state = np.random.randn(1, 2).astype(np.float32)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_input = {
                "observation.images.cam_front": torch.from_numpy(dummy_image),
                "observation.state": torch.from_numpy(dummy_state),
            }
            pytorch_output = self.model(pytorch_input).cpu().numpy()
        
        # ONNX inference
        onnx_input = {
            'image': dummy_image,
            'state': dummy_state
        }
        onnx_output = ort_session.run(output_names, onnx_input)[0]
        
        # Compare outputs
        diff = np.abs(pytorch_output - onnx_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        logger.info("📊 Validation Results:")
        logger.info(f"   Max difference: {max_diff:.6f}")
        logger.info(f"   Mean difference: {mean_diff:.6f}")
        
        if max_diff < 1e-5:
            logger.info("   ✅ Perfect match!")
        elif max_diff < 1e-3:
            logger.info("   ✅ Excellent match (minor numerical differences)")
        elif max_diff < 0.01:
            logger.info("   ⚠️  Acceptable match (some differences)")
        else:
            logger.info("   ❌ Poor match - check export settings")
        
        # Validate on real data if provided
        if test_data_dir:
            self._validate_on_dataset(ort_session, test_data_dir)
    
    def _validate_on_dataset(self, ort_session, data_dir: str, num_samples: int = 50):
        """Validate on real dataset"""
        logger.info(f"   Testing on {num_samples} real samples...")
        
        dataset = TracerLocalDataset(data_dir=data_dir)
        num_samples = min(num_samples, len(dataset))
        
        diffs = []
        
        for i in range(num_samples):
            try:
                sample = dataset[i]
                
                # Prepare inputs
                image = sample["observation.images.cam_front"].unsqueeze(0).numpy()
                state = sample["observation.state"].unsqueeze(0).numpy()
                
                # PyTorch inference
                with torch.no_grad():
                    pytorch_input = {
                        "observation.images.cam_front": torch.from_numpy(image),
                        "observation.state": torch.from_numpy(state),
                    }
                    pytorch_output = self.model(pytorch_input).cpu().numpy()
                
                # ONNX inference
                onnx_input = {'image': image, 'state': state}
                onnx_output = ort_session.run(['actions'], onnx_input)[0]
                
                # Compare
                diff = np.abs(pytorch_output - onnx_output).mean()
                diffs.append(diff)
                
            except Exception as e:
                logger.warning(f"   Skipping sample {i}: {e}")
                continue
        
        if diffs:
            avg_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            logger.info(f"   Real data validation:")
            logger.info(f"     Average diff: {avg_diff:.6f}")
            logger.info(f"     Max diff: {max_diff:.6f}")
            
            if avg_diff < 1e-3:
                logger.info("     ✅ Excellent accuracy on real data")
            elif avg_diff < 0.01:
                logger.info("     ✅ Good accuracy on real data")
            else:
                logger.info("     ⚠️  Accuracy degradation detected")


def simplify_onnx(onnx_path: str, output_path: str = None):
    """
    Simplify ONNX model using onnx-simplifier
    
    Args:
        onnx_path: Input ONNX model
        output_path: Output simplified model (optional)
    """
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        logger.error("❌ onnx-simplifier not installed. Install with: pip install onnx-simplifier")
        return
    
    logger.info("🔧 Simplifying ONNX model...")
    
    if output_path is None:
        output_path = onnx_path.replace('.onnx', '_simplified.onnx')
    
    # Load and simplify
    model = onnx.load(onnx_path)
    model_simplified, check = simplify(model)
    
    if check:
        onnx.save(model_simplified, output_path)
        
        orig_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        simp_size = Path(output_path).stat().st_size / (1024 * 1024)
        
        logger.info(f"✅ Simplified model saved to {output_path}")
        logger.info(f"   Original size: {orig_size:.2f} MB")
        logger.info(f"   Simplified size: {simp_size:.2f} MB")
        logger.info(f"   Size reduction: {(1 - simp_size/orig_size) * 100:.1f}%")
    else:
        logger.error("❌ Simplification failed validation")


def main():
    parser = argparse.ArgumentParser(description='Export ACT Model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, required=True, help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--validate', action='store_true', help='Validate ONNX output')
    parser.add_argument('--test_data', type=str, help='Test data directory for validation')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model')
    parser.add_argument('--image_size', type=int, nargs=2, default=[360, 640],
                       help='Image size (height width)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔄 ACT MODEL ONNX EXPORT")
    print("=" * 80)
    print(f"Input:  {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Opset:  {args.opset}")
    print("=" * 80)
    print()
    
    # Create exporter
    exporter = ACTONNXExporter(
        checkpoint_path=args.checkpoint,
        image_size=tuple(args.image_size),
    )
    
    # Export
    exporter.export_to_onnx(
        output_path=args.output,
        opset_version=args.opset,
    )
    
    # Validate if requested
    if args.validate:
        print()
        exporter.validate_onnx(args.output, args.test_data)
    
    # Simplify if requested
    if args.simplify:
        print()
        simplify_onnx(args.output)
    
    print()
    print("=" * 80)
    print("✅ ONNX EXPORT COMPLETE!")
    print("=" * 80)
    print()
    print("📋 Next Steps:")
    print()
    print("For Hailo AI HAT deployment:")
    print("1. Transfer ONNX to x86_64 machine with Hailo Dataflow Compiler")
    print("2. Parse: hailo parser onnx act_model.onnx")
    print("3. Optimize: hailo optimize --hw-arch hailo8l")
    print("4. Compile: hailo compiler --output act_model.hef")
    print()
    print("For ONNX Runtime inference:")
    print("1. Install: pip install onnxruntime")
    print("2. Run: python onnx_inference.py --model act_model.onnx")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
