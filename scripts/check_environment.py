"""Check imports and run one tensor operation on the available device."""

import platform

import matplotlib
import numpy
import pandas
import scipy
import seaborn
import sklearn
import torch
import torchvision

print("Python:", platform.python_version())
for module in [torch, torchvision, numpy, pandas, scipy, matplotlib, seaborn, sklearn]:
    print(f"{module.__name__}: {module.__version__}")

print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
x = torch.ones((16, 16), device=device)
assert torch.all(x @ x == 16).item()
print(f"Tensor operation passed on {device}.")
if device.type == "cpu":
    print("CPU works. If you expected GPU support, check nvidia-smi and the PyTorch build before training.")
