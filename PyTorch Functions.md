# 40 Important PyTorch Functions 
---

## 1. Tensor creation

### torch.tensor(data, dtype=None, device=None, requires_grad=False)
→ Creates a tensor by copying data from Python lists, tuples, or NumPy arrays into a PyTorch format for computation.

### torch.zeros(*size, dtype=None, device=None)
→ Creates a tensor of the given shape filled entirely with zeros, commonly used for initialization.

### torch.ones(*size, dtype=None, device=None)
→ Creates a tensor filled with ones, often used for masks or initial constant values.

### torch.rand(*size, dtype=None, device=None)
→ Generates random values uniformly distributed between 0 and 1 for stochastic initialization or sampling.

### torch.randn(*size, dtype=None, device=None)
→ Generates values from a standard normal distribution (mean 0, variance 1), widely used for weight initialization.

### torch.arange(start=0, end, step=1)
→ Produces evenly spaced values in a range, similar to Python’s range but as a tensor.

### torch.linspace(start, end, steps)
→ Creates a tensor with a fixed number of evenly spaced values between start and end (inclusive).

---

## 2. Tensor shape & structure

### torch.reshape(input, shape)
→ Returns a tensor with the same data but rearranged into a new shape without changing values.

### tensor.view(*shape)
→ Reshapes a tensor using the same memory, making it faster but requiring compatible layout.

### torch.transpose(input, dim0, dim1)
→ Swaps two specified dimensions of a tensor, often used in matrix and image operations.

### torch.permute(input, dims)
→ Rearranges all tensor dimensions into a new order, commonly used for changing data layout (e.g., NCHW ↔ NHWC).

### torch.squeeze(input)
→ Removes dimensions of size 1 to simplify tensor shape without losing data.

### torch.unsqueeze(input, dim)
→ Adds a new dimension of size 1 at a specified position, useful for batching or broadcasting.

### torch.flatten(input)
→ Collapses all dimensions into a single 1D vector, often used before fully connected layers.

### torch.cat(tensors, dim=0)
→ Joins multiple tensors along an existing dimension, combining data without creating a new axis.

### torch.stack(tensors, dim=0)
→ Combines tensors along a new dimension, increasing rank instead of merging.

---

## 3. Math operations

### torch.add(a, b)
→ Performs element-wise addition between tensors of the same shape or broadcastable shapes.

### torch.sub(a, b)
→ Performs element-wise subtraction, useful in loss and gradient calculations.

### torch.mul(a, b)
→ Performs element-wise multiplication between tensors.

### torch.div(a, b)
→ Performs element-wise division, often used in normalization.

### torch.matmul(a, b)
→ Performs matrix multiplication, the core operation in neural network layers.

### torch.mm(a, b)
→ Performs matrix multiplication for 2D tensors only (special case of matmul).

### torch.mean(input)
→ Computes the average value of all elements or along a specific dimension.

### torch.sum(input)
→ Computes the total sum of tensor elements, commonly used in loss aggregation.

### torch.max(input)
→ Returns the maximum value in a tensor (or along a dimension).

### torch.min(input)
→ Returns the minimum value in a tensor (or along a dimension).

### torch.pow(input, exponent)
→ Raises each element to a power, used in polynomial and scaling operations.

### torch.exp(input)
→ Applies exponential function element-wise, often used in softmax computation.

### torch.log(input)
→ Applies natural logarithm element-wise, commonly used in loss functions.

---

## 4. Neural network building (torch.nn)

### torch.nn.Linear(in_features, out_features)
→ Fully connected layer that applies a learned linear transformation (Wx + b) between inputs and outputs.

### torch.nn.Conv2d(in_channels, out_channels, kernel_size)
→ Convolution layer that extracts spatial features from images using learnable filters.

### torch.nn.ReLU()
→ Applies a non-linear activation that sets negative values to zero, improving learning capacity.

### torch.nn.Sigmoid()
→ Squashes outputs into the range (0,1), often used for binary classification probabilities.

### torch.nn.Tanh()
→ Squashes values into (-1,1), centering outputs for certain network architectures.

### torch.nn.Softmax(dim)
→ Converts raw scores into a probability distribution that sums to 1 across a dimension.

### torch.nn.Dropout(p)
→ Randomly disables neurons during training to reduce overfitting and improve generalization.

### torch.nn.Embedding(num_embeddings, embedding_dim)
→ Learns dense vector representations for discrete tokens, essential for language models.

---

## 5. Autograd (learning system)

### tensor.requires_grad_(True)
→ Enables gradient tracking so PyTorch can compute how values change during learning.

### torch.autograd.backward(tensor)
→ Computes gradients for tensors in the computation graph (manual interface).

### tensor.backward()
→ Triggers backpropagation to compute gradients of a scalar loss automatically.

### torch.no_grad()
→ Temporarily disables gradient tracking to speed up inference and reduce memory usage.

### torch.grad.zero_()
→ Clears old gradients before a new optimization step (typically via optimizer).

---

## 6. Optimization

### torch.optim.SGD(params, lr)
→ Basic optimizer that updates parameters using gradient descent in small steps.

### torch.optim.Adam(params, lr)
→ Adaptive optimizer that adjusts learning rates automatically, widely used in deep learning.

### torch.optim.AdamW(params, lr)
→ Improved Adam variant that decouples weight decay for better training stability in transformers.

---

## 7. Device handling

### tensor.to(device)
→ Moves tensor between CPU and GPU memory for faster computation when using CUDA.

### torch.cuda.is_available()
→ Checks whether a compatible NVIDIA GPU is available for acceleration.

### torch.device("cuda" or "cpu")
→ Specifies where tensors and models should run (CPU or GPU).

---

## 8. Data handling

### torch.utils.data.Dataset
→ Base class for defining custom datasets that return individual data samples.

### torch.utils.data.DataLoader
→ Handles batching, shuffling, and parallel loading of dataset samples for training loops.

---

## 9. Neuron-level / low-level control

### torch.nn.Parameter(data)
→ Wraps a tensor as a learnable model weight that will be updated during training.

### torch.nn.init.xavier_uniform_(tensor)
→ Initializes weights to maintain stable gradients in deep networks by balancing variance.

### torch.nn.init.normal_(tensor)
→ Initializes weights using a normal distribution for stochastic symmetry breaking.

### torch.sigmoid(x)
→ Applies smooth nonlinear activation that maps values into (0,1).

### torch.relu(x)
→ Applies non-linearity by zeroing negative values while keeping positive ones unchanged.

### torch.where(condition, x, y)
→ Selects elements from two tensors based on a condition, acting like element-wise logic gates.

### torch.sum(x, dim)
→ Aggregates values across dimensions, often representing neuron outputs in layers.

### torch.mean(x, dim)
→ Computes average activations across neurons or batches for normalization and analysis.