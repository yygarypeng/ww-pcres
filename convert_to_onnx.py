import glob
import torch
from model import LightningWBoson
import train

# https://docs.pytorch.org/docs/2.8/onnx.html

# Find the checkpoint
ckpt_files = glob.glob(f"./hww_pcres_regressor/logs/version_0/checkpoints/*")
if not ckpt_files: 
    raise FileNotFoundError(f"No checkpoint files found in checkpoints")
ckpt_path = ckpt_files[0]  # Use the first checkpoint found
print(f"Using checkpoint: {ckpt_path}")

model = LightningWBoson.load_from_checkpoint(ckpt_path)
model.eval()

example_input = torch.randn(train.BATCH_SIZE, 10) # dummy input with batch size and 10 features (l0, l1, met)
example_input = example_input.to(device=model.device)

onnx_path = "./hww_pcres_regressor/hww_pcrec_regressor.onnx"
torch.onnx.export(
    model,
    example_input,
    onnx_path,
    input_names=['inputs'],
    export_params=True,
    dynamo=True,
    do_constant_folding=True,
    opset_version=15,
)

try:
	import onnx
	onnx_model = onnx.load(onnx_path)
	onnx.checker.check_model(onnx_model)
	print("ONNX model is valid!")
except Exception as e:
	print(f"ONNX validation error: {e}")