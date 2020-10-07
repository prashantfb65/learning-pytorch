# PyTorch

`torch` and `torchvision` are the two important packages to install.

```bash
pip install torch
```

```bash
pip install torchvision
```

## What is PyTorch?

It’s a Python-based scientific computing package targeted at two sets of audiences:

- A replacement for NumPy to use the power of GPUs
- A deep learning research platform that provides maximum flexibility and speed

### Tensors
Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.

```python
import torch
```

An uninitialized matrix is declared, but does not contain definite known values before it is used. When an uninitialized matrix is created, whatever values were in the allocated memory at the time will appear as the initial values.

Construct a 5x3 matrix, uninitialized:
```python
x = torch.empty(5, 3)
print(x)
```

Construct a randomly initialized matrix:
```python
x = torch.rand(5, 3)
print(x)
```

Construct a matrix filled zeros and of dtype long:
```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

Converting NumPy Array to Torch Tensor
```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

Output:
```bash
[2. 2. 2. 2. 2.]
tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

Tensors can be moved onto any device using the .to method.
```python
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
```

What is CUDA ?
CUDA is a parallel computing platform and programming model that makes using a GPU for general purpose computing simple and elegant. The developer still programs in the familiar C, C++, Fortran, or an ever expanding list of supported languages, and incorporates extensions of these languages in the form of a few basic keywords.

### Autograd 
Central to all neural networks in PyTorch is the `autograd` package.
The autograd package provides automatic differentiation for all operations on Tensors. It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that every single iteration can be different.

Source credits: https://pytorch.org/

