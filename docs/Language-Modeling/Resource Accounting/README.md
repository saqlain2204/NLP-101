
# Resource Accounting

## Tensor Basics
Tensors are the fundamental building blocks for storing everything in machine learning: parameters, gradients, optimizer states, and data activations.

### Tensor Memory Types
- **float32 (single precision):**
  - Commonly called full precision in machine learning.
- **Memory usage is determined by:**
  - Number of values
  - Data type of each value
- **float16 (half precision):**
  - Not ideal for representing very small numbers.
- **bfloat16 (brain floating point):**
  - Allocates more bits to the exponent and fewer to the fraction. Uses the same memory as float32.
- **fp8 (8-bit floating point):**
  - Uses only 8 bits per value.

### Implications for Training
- Training with float32 is stable but requires a lot of memory.
- Training with fp8, float16, or bfloat16 can be risky and may lead to instability.
- Mixed precision training is often used (e.g., using float32 for attention layers and bfloat16 for feed-forward networks).
- By default, tensors are stored on the CPU.

### Example: Checking Device and Memory in PyTorch
```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    properties = torch.cuda.get_device_properties(i)
    print(properties)

memory_allocation = torch.cuda.memory_allocated()
```

In PyTorch, tensors are pointers to allocated memory, with metadata describing how to access any element of the tensor.
In pytorch tensors are pointers into allocated memeory with metadata describing how to get to any elemtn of the tensor






