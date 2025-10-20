"""
Export trained GNN model to ONNX format for Rust integration
"""

import torch
import torch.onnx
from model import MultiTaskGATv2
from torch_geometric.data import Data, Batch
import numpy as np


def export_model_to_onnx(checkpoint_path, output_path):
    """Export PyTorch model to ONNX format"""

    print("Loading model...")
    device = torch.device('cpu')  # Use CPU for export

    # Initialize model architecture
    model = MultiTaskGATv2(
        node_feature_dim=16,
        hidden_dim=256,
        num_gnn_layers=6,
        num_attention_heads=8,
        max_colors=210,
        num_graph_types=8,
        dropout=0.0,  # Disable dropout for inference
    )
    model.eval()  # Set to evaluation mode

    # Load trained weights
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')+1}")
            print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)

    # Create dummy input for tracing
    # IMPORTANT: These dimensions must match what Rust will provide
    print("\nCreating dummy input for ONNX export...")

    # Example graph with 100 nodes
    num_nodes = 100
    num_edges = 500
    batch_size = 1

    # Node features: [num_nodes, 16]
    x = torch.randn(num_nodes, 16, dtype=torch.float32)

    # Edge index: [2, num_edges]
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

    # Batch assignment (all nodes belong to graph 0 for single graph)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Create PyG Data object
    dummy_data = Data(x=x, edge_index=edge_index, batch=batch)
    dummy_batch = Batch.from_data_list([dummy_data])

    # Export to ONNX
    print(f"\nExporting to ONNX format: {output_path}")

    # We need to wrap the model to handle PyG batch input
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, edge_index, batch):
            # Reconstruct Data object for model
            data = Data(x=x, edge_index=edge_index, batch=batch)
            output = self.model(data)
            # Return all outputs as separate tensors
            return (
                output['color_logits'],
                output['chromatic'],
                output['graph_type_logits'],
                output['difficulty']
            )

    wrapped_model = ONNXWrapper(model)
    wrapped_model.eval()

    # Export with dynamic axes for variable graph sizes
    dynamic_axes = {
        'node_features': {0: 'num_nodes'},
        'edge_index': {1: 'num_edges'},
        'batch': {0: 'num_nodes'},
        'color_logits': {0: 'num_nodes'},
    }

    torch.onnx.export(
        wrapped_model,
        (dummy_batch.x, dummy_batch.edge_index, dummy_batch.batch),
        output_path,
        input_names=['node_features', 'edge_index', 'batch'],
        output_names=['color_logits', 'chromatic', 'graph_type_logits', 'difficulty'],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"✅ ONNX model exported successfully to: {output_path}")

    # Verify the export
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation passed!")

        # Print model info
        print(f"\nModel info:")
        print(f"  Input shapes:")
        print(f"    - node_features: [num_nodes, 16]")
        print(f"    - edge_index: [2, num_edges]")
        print(f"    - batch: [num_nodes]")
        print(f"  Output shapes:")
        print(f"    - color_logits: [num_nodes, 210]")
        print(f"    - chromatic: [1]")
        print(f"    - graph_type_logits: [1, 8]")
        print(f"    - difficulty: [1]")

    except ImportError:
        print("⚠️  Install 'onnx' package to verify: pip install onnx")
    except Exception as e:
        print(f"⚠️  ONNX validation warning: {e}")

    return True


def test_onnx_inference(onnx_path):
    """Test ONNX model inference"""
    try:
        import onnxruntime as ort

        print(f"\nTesting ONNX inference with: {onnx_path}")

        # Create inference session
        session = ort.InferenceSession(onnx_path)

        # Get input details
        inputs = session.get_inputs()
        print("\nModel inputs:")
        for inp in inputs:
            print(f"  - {inp.name}: {inp.shape} ({inp.type})")

        # Prepare test input
        num_nodes = 50
        num_edges = 200

        input_dict = {
            'node_features': np.random.randn(num_nodes, 16).astype(np.float32),
            'edge_index': np.random.randint(0, num_nodes, (2, num_edges)).astype(np.int64),
            'batch': np.zeros(num_nodes, dtype=np.int64),
        }

        # Run inference
        outputs = session.run(None, input_dict)

        print("\nInference results:")
        print(f"  Color logits shape: {outputs[0].shape}")
        print(f"  Chromatic prediction: {outputs[1][0]:.2f}")
        print(f"  Graph type logits: {outputs[2]}")
        print(f"  Difficulty prediction: {outputs[3][0]:.2f}")

        print("\n✅ ONNX inference test passed!")
        return True

    except ImportError:
        print("⚠️  Install 'onnxruntime' to test: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='/home/diddy/Downloads/best_model_epoch5.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str,
                        default='gnn_model.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--test', action='store_true',
                        help='Test ONNX inference after export')

    args = parser.parse_args()

    # Export model
    success = export_model_to_onnx(args.checkpoint, args.output)

    # Test inference if requested
    if success and args.test:
        test_onnx_inference(args.output)

    if success:
        print(f"\n🎉 Model ready for Rust integration!")
        print(f"Copy '{args.output}' to your Rust project's model directory")