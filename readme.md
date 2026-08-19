# MNIST from scratch

A neural network implementation from scratch in C.

## Features

* Scalar-based automatic differentiation
* Forward propagation
* Backpropagation
* MLP implementation
* MNIST handwritten digit classification
* Model weight serialization to file

## Implementation Details

### Scalar computation graph

The automatic differentiation engine represents every scalar value as a
`compute_node`. A node stores its current value (`data`), accumulated gradient
(`grad`), input nodes, and function pointers for its forward and backward
operations. Model inputs, weights, and biases are leaf nodes, while arithmetic
operations create the internal nodes of the graph.

The graph is built once when the MLP is initialized. A depth-first search then
produces a topological ordering of all nodes reachable from the loss:

* Forward propagation evaluates nodes in topological order.
* Backpropagation resets the gradients, sets the loss gradient to `1`, and
  evaluates the backward functions in reverse topological order.
* Gradients are accumulated with `+=`, so a value shared by several operations
  receives the sum of all gradient contributions.

The engine currently implements scalar addition, multiplication, ReLU, and a
combined softmax cross-entropy loss.

### MLP architecture

The classifier is a fully connected network with the following architecture:

```text
784 inputs -> 128 ReLU hidden units -> 10 output logits
```

For a flattened and normalized MNIST image, each hidden unit computes a weighted
sum of all 784 input pixels, adds a bias, and applies ReLU:

```text
hidden[j] = ReLU(sum(W1[j][i] * input[i]) + b1[j])
```

Each output logit is then computed as a weighted sum of the 128 hidden-unit
outputs plus a bias:

```text
logit[k] = sum(W2[k][j] * hidden[j]) + b2[k]
```

The network produces ten logits, one for each digit from `0` to `9`. The
predicted digit is the index of the largest logit.

Weights use He initialization. Each weight is sampled from a normal distribution
with zero mean and a variance scaled according to the number of inputs to the
layer:

```text
variance = 2 / fan_in
```

This initialization is well suited to ReLU networks. Standard normal samples
are generated with the Box-Muller transform. Biases are initialized to zero.

### Numerically stable cross-entropy

Softmax is not stored as a separate layer in the computation graph. Instead,
the loss node receives all ten logits and the target class through its `ctx`
field.

For target class `y`, the cross-entropy loss is computed conceptually as:

```text
loss = -logit[y] + log(sum(exp(logit[i])))
```

To avoid overflow when exponentiating large logits, the implementation first
finds the maximum logit:

```text
m = max(logit[i])
```

and evaluates the equivalent expression:

```text
loss = -(logit[y] - m) + log(sum(exp(logit[i] - m)))
```

Subtracting the same maximum value from every logit does not change the softmax
probabilities, while keeping the exponential values numerically stable.

The backward pass directly computes the fused softmax cross-entropy gradient:

```text
grad[i] = softmax(logit)[i] - (i == y ? 1 : 0)
```

This keeps the graph scalar-based while avoiding ten additional softmax output
nodes and an explicit one-hot target vector.

### Training loop

MNIST pixels are read as bytes and normalized from `[0, 255]` to `[0, 1]`.

Training uses online stochastic gradient descent. For every training image, the
program:

1. Loads and normalizes the image.
2. Runs forward propagation.
3. Computes the cross-entropy loss.
4. Runs backward propagation.
5. Immediately updates every weight and bias.

Each parameter is updated using:

```text
parameter -= 0.005 * gradient
```

By default, the program trains for 10 epochs. Each epoch processes all 60,000
training images and then reports average loss and accuracy on the 10,000-image
test set.

### Model weight serialization

The MLP supports saving its trained parameters to a file. This allows the
weights produced during training to be persisted instead of existing only for
the lifetime of the process.

The serialized model contains the trainable parameters of the network:

* Input-to-hidden weights
* Hidden-layer biases
* Hidden-to-output weights
* Output-layer biases

Because the computation graph itself is reconstructed when the MLP is
initialized, only the trainable parameter values need to be stored. Internal
operation nodes and intermediate activations are not part of the saved model.

The `./model` directory contains several pretrained models. The naming format is:

```text
MLP-epoch%02d-lr%f-t%lu.bin
```

where the fields represent the number of training epochs, the learning rate, and the Unix timestamp, respectively. Below is the training output log:

```text
Toposort finished. Node count: 204453
Running epoch #1 ...
[OK] 60000 images finished. Avg loss: 0.248510. Avg accuracy: 0.929083
[OK] 10000 images finished. Avg loss: 0.150995. Avg accuracy: 0.956000
[OK] Successfully saved model to: './model/MLP-epoch01-lr0.005000-t1787155610.bin'.
Running epoch #2 ...
[OK] 60000 images finished. Avg loss: 0.116388. Avg accuracy: 0.966067
[OK] 10000 images finished. Avg loss: 0.110250. Avg accuracy: 0.966900
[OK] Successfully saved model to: './model/MLP-epoch02-lr0.005000-t1787155744.bin'.
Running epoch #3 ...
[OK] 60000 images finished. Avg loss: 0.082198. Avg accuracy: 0.976417
[OK] 10000 images finished. Avg loss: 0.096585. Avg accuracy: 0.971100
[OK] Successfully saved model to: './model/MLP-epoch03-lr0.005000-t1787155881.bin'.
Running epoch #4 ...
[OK] 60000 images finished. Avg loss: 0.063053. Avg accuracy: 0.982167
[OK] 10000 images finished. Avg loss: 0.093596. Avg accuracy: 0.972000
[OK] Successfully saved model to: './model/MLP-epoch04-lr0.005000-t1787156016.bin'.
Running epoch #5 ...
[OK] 60000 images finished. Avg loss: 0.050338. Avg accuracy: 0.986083
[OK] 10000 images finished. Avg loss: 0.088799. Avg accuracy: 0.973400
[OK] Successfully saved model to: './model/MLP-epoch05-lr0.005000-t1787156171.bin'.
Running epoch #6 ...
[OK] 60000 images finished. Avg loss: 0.040765. Avg accuracy: 0.989083
[OK] 10000 images finished. Avg loss: 0.087828. Avg accuracy: 0.973700
[OK] Successfully saved model to: './model/MLP-epoch06-lr0.005000-t1787156334.bin'.
Running epoch #7 ...
[OK] 60000 images finished. Avg loss: 0.033030. Avg accuracy: 0.991567
[OK] 10000 images finished. Avg loss: 0.086342. Avg accuracy: 0.974300
[OK] Successfully saved model to: './model/MLP-epoch07-lr0.005000-t1787156510.bin'.
Running epoch #8 ...
[OK] 60000 images finished. Avg loss: 0.026674. Avg accuracy: 0.993667
[OK] 10000 images finished. Avg loss: 0.083407. Avg accuracy: 0.975400
[OK] Successfully saved model to: './model/MLP-epoch08-lr0.005000-t1787156672.bin'.

```

### MNIST IDX reader

The dataset loader reads the original IDX binary files directly. Image headers
contain the magic number, image count, row count, and column count; label
headers contain the magic number and label count.

Because IDX stores 32-bit header fields in big-endian order, each field is
byte-swapped before use. Image and label records are then streamed sequentially
from their files rather than loading the entire dataset into memory.

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
