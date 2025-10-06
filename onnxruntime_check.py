import glob
import numpy as np

import onnxruntime
import torch

from model import LightningWBoson
import train


# Path to the ONNX model
onnx_path = "./hww_pcres_regressor/hww_pcrec_regressor.onnx"
print(f"Loading ONNX model from {onnx_path}")
# Load the ONNX model
# print(onnxruntime.get_available_providers()) # debug: check available providers
ort_session = onnxruntime.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"]
)

# Create sample input (match the dimensions used during export `example_input`)
batch_size = train.BATCH_SIZE
num_features = 10
test_input = np.random.randn(batch_size, num_features).astype(np.float32)

# Run inference with ONNX Runtime
# https://onnxruntime.ai/docs/get-started/with-python.html
# print(ort_session.get_inputs()[0].name)  # debug: print input name
ort_inputs = {"inputs": test_input} # use the correct input name when exporting `input_names` to onnx.
ort_outputs = ort_session.run(None, ort_inputs) # None: to get all output nodes
ort_result = ort_outputs[0]

print(f"Input shape: {test_input.shape}")
print(f"ONNX model output shape: {ort_result.shape}")
print("\nSample predictions from ONNX model:")
print(ort_result[:2])  # Print first 2 predictions

try:
    # Find the checkpoint (search all versions)
    ckpt_files = glob.glob("./hww_pcres_regressor/logs/**/checkpoints/*.ckpt")
    if ckpt_files:
        ckpt_path = ckpt_files[0]
        print(f"\nComparing with original PyTorch model from {ckpt_path}")

        # Load the PyTorch model on CPU
        pytorch_model = LightningWBoson.load_from_checkpoint(ckpt_path, map_location=torch.device('cpu'))
        pytorch_model.eval()

        torch_input = torch.tensor(test_input, dtype=torch.float32)

        with torch.no_grad():
            pytorch_output = pytorch_model(torch_input).detach().cpu().numpy()

        # Compare results
        max_diff = np.max(np.abs(pytorch_output - ort_result))
        tolerance = 1e-3 # abs error needs to within 0.1%
        allclose = np.allclose(pytorch_output, ort_result, atol=tolerance, rtol=0.0)
        print(f"Maximum l1 diff: {max_diff}")
        if allclose:
            print("Match and within tolerance!")
        else:
            print("Differ!")
            
    else:
        print("No PyTorch ckpt found for comparison.")
        
except Exception as e:
    print(f"Could not compare with PyTorch model: {e}")

print("\nONNX model inference finished.")