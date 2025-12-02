import torch
import sys
import platform

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")

print("-" * 20)
print("Checking MPS (Mac Silicon)...")
if torch.backends.mps.is_available():
    print("✅ MPS is AVAILABLE")
else:
    print("❌ MPS is NOT available")
    if not torch.backends.mps.is_built():
        print("  (PyTorch was not built with MPS support)")

print("-" * 20)
print("Checking CUDA...")
if torch.cuda.is_available():
    print("✅ CUDA is AVAILABLE")
else:
    print("❌ CUDA is NOT available")
