

# Resource Accounting

## Key Points

- **Tensors** are the core data structure in machine learning, used for:
  - Parameters
  - Gradients
  - Optimizer states
  - Data activations

## Tensor Memory Types

- **float32 (single precision):**
  - Standard for full precision in ML
  - Good balance of range and accuracy

- **float16 (half precision):**
  - Uses less memory
  - Not ideal for very small numbers

- **bfloat16 (brain floating point):**
  - Same memory as float32
  - More bits for exponent, fewer for fraction

- **fp8 (8-bit floating point):**
  - Very low memory usage

**Memory usage depends on:**
- Number of values
- Data type of each value

## Training Implications

- Training with float32 is stable but memory-intensive
- Training with float16, bfloat16, or fp8 saves memory but can cause instability
- Mixed precision training is common (e.g., float32 for attention, bfloat16 for feed-forward)
- Tensors are on CPU by default; use GPU for acceleration

## PyTorch Example: Device & Memory

```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i)
    print(properties)
memory_allocation = torch.cuda.memory_allocated() if device == 'cuda' else None
print(f"Memory allocated: {memory_allocation}")
```

## How Tensors Work in PyTorch

- Tensors are pointers to allocated memory
- Metadata describes how to access each element