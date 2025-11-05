import cupy as cp
print("CuPy:", cp.__version__)
print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
print("GPU count:", cp.cuda.runtime.getDeviceCount())
x = cp.arange(10_000_000, dtype=cp.float32)
print(float(x.sum()))