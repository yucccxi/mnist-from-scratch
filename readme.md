# MNIST from scratch

A neural network implementation from scratch in C.

## Features

- Scalar-based automatic differentiation
- Forward propagation
- Backpropagation
- MLP implementation
- MNIST handwritten digit classification

## Implementation Details

### Scalar computation graph

The automatic differentiation engine represents every scalar value as a
`compute_node`. A node stores its current value (`data`), accumulated gradient
(`grad`), input nodes, and function pointers for its forward and backward
operations. Model inputs, weights, and biases are leaf nodes, while arithmetic
operations create the internal nodes of the graph.

The graph is built once when the MLP is initialized. A depth-first search then
produces a topological ordering of all nodes reachable from the loss:

- Forward propagation evaluates nodes in topological order.
- Backpropagation resets the gradients, sets the loss gradient to `1`, and
  evaluates the backward functions in reverse topological order.
- Gradients are accumulated with `+=`, so a value shared by several operations
  receives the sum of all gradient contributions.

The engine currently implements scalar addition, multiplication, ReLU, and a
combined softmax cross-entropy loss.

### MLP architecture

The classifier is a fully connected network with the following architecture:

```text
784 inputs -> 128 ReLU hidden units -> 10 output logits
```

For a flattened, normalized MNIST image $x$, the network computes

$$
h_j = \operatorname{ReLU}\left(\sum_{i=1}^{784} W^{(1)}_{ji}x_i + b^{(1)}_j\right),
$$

followed by

$$
z_k = \sum_{j=1}^{128} W^{(2)}_{kj}h_j + b^{(2)}_k,
$$

where the ten values $z_k$ are logits. The predicted digit is the index of
the largest logit.

Weights use He initialization,
$W \sim \mathcal{N}(0, 2/\text{fan-in})$, which is suitable for ReLU layers.
Standard normal samples are generated with the Box-Muller transform. Biases
start at zero.

### Numerically stable cross-entropy

Softmax is not stored as a separate layer in the computation graph. Instead,
the loss node receives all ten logits and the target class through its `ctx`
field. For target class $y$, it computes

$$
L = -z_y + \log\left(\sum_i e^{z_i}\right).
$$

The implementation subtracts $m = \max_i z_i$ before exponentiation:

$$
L = -(z_y-m) + \log\left(\sum_i e^{z_i-m}\right),
$$

which avoids overflow without changing the result. Its backward pass directly
uses the fused gradient

$$
\frac{\partial L}{\partial z_i}
= \operatorname{softmax}(z)_i - \mathbf{1}[i=y].
$$

This keeps the graph scalar-based while avoiding ten additional softmax output
nodes and an explicit one-hot target vector.

### Training loop

MNIST pixels are read as bytes and normalized from `[0, 255]` to `[0, 1]`.
Training uses online stochastic gradient descent: after each image, the program
runs a forward pass, computes all gradients, and immediately updates every
weight and bias with

$$
\theta \leftarrow \theta - 0.005\,\frac{\partial L}{\partial \theta}.
$$

By default, the program trains for 10 epochs. Each epoch processes all 60,000
training images and then reports average loss and accuracy on the 10,000-image
test set.

### MNIST IDX reader

The dataset loader reads the original IDX binary files directly. Image headers
contain the magic number, image count, row count, and column count; label
headers contain the magic number and label count. Because IDX stores 32-bit
header fields in big-endian order, each field is byte-swapped before use. Image
and label records are then streamed sequentially from their files rather than
loading the entire dataset into memory.

## Build & Run

### Requirements

* GCC
* GNU Make

### Build

```bash
make build
```

The executable will be generated at:

```text
./out/main
```

### Run

```bash
make run
```

### Clean

Remove generated build files:

```bash
make clean
```
