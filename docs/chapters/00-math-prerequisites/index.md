---
title: "Chapter 0: Math and Programming Prerequisites"
description: "Builds the mathematical and computational foundation — linear algebra, calculus, backpropagation, PyTorch tensors, and PyTorch Geometric — needed to implement every GNN in this textbook from scratch."
---

# Chapter 0: Math and Programming Prerequisites

**Part 0: Prerequisites**

## Summary

Builds the mathematical and computational foundation — linear algebra, calculus, backpropagation, PyTorch tensors, and PyTorch Geometric — needed to implement every GNN architecture in this textbook from scratch.

## Concepts Covered

This chapter covers the following 20 concepts from the learning graph:

1. Vector and Dot Product
2. Cosine Similarity
3. Matrix Multiplication
4. Matrix Transpose
5. Matrix Rank
6. Symmetric Matrix
7. Positive Semi-Definite Matrix
8. Eigenvalue Decomposition
9. Singular Value Decomposition (SVD)
10. Gradient Descent
11. Chain Rule (Calculus)
12. Backpropagation
13. Automatic Differentiation
14. Activation Function
15. Multilayer Perceptron (MLP)
16. Adam Optimizer
17. PyTorch Tensor
18. PyTorch Autograd
19. PyTorch Geometric (PyG)
20. NetworkX

## Prerequisites

This chapter has no prerequisites. It is the mathematical and computational foundation for the entire textbook. Readers comfortable with undergraduate linear algebra, multivariable calculus, and Python may skim or skip individual sections.

---

## Motivating Example: Predicting Protein Interactions with Math

Every year, pharmaceutical companies screen billions of molecular compounds hoping to find a drug that binds to a specific protein implicated in a disease. A single drug-discovery campaign can test 10 million candidates and still fail. The bottleneck is not lab equipment — it is *understanding relationships*.

Proteins interact in networks. A drug inhibits an enzyme, which frees a signaling molecule, which activates a receptor — a chain of effects rippling through a graph of biological interactions. To predict whether a new compound will bind usefully to its target protein, researchers now represent both the drug and the protein as *graphs* and train machine learning models to reason over those graphs.

What powers those models? Matrix multiplication determines how information flows from a node to its neighbors. Eigenvalue decomposition reveals the large-scale structure hidden in a network's adjacency matrix. Gradient descent tunes hundreds of thousands of parameters so the model's predictions improve. Cross-entropy loss quantifies the gap between what the model guesses and the truth.

This chapter builds every one of those mathematical tools from the ground up. By the end, you will have working PyTorch code for a complete training loop and the conceptual vocabulary to understand every GNN architecture in the chapters that follow. No concept in this textbook is more than two or three steps away from something in this chapter.

!!! mascot-welcome "Before the Graphs, the Math"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving welcome">
    This chapter is the shared foundation for every technique in this textbook. Every matrix operation, every gradient, every activation function introduced here will reappear — sometimes in disguise — when you build GCNs, GATs, and Graph Transformers later on. Work through it carefully. The investment pays dividends on every page that follows.

---

## 0.1 Vectors and the Dot Product

### 0.1.1 Intuition

A vector is a list of numbers. That is the whole definition — but the geometry makes it powerful. You can picture a vector $\mathbf{x} \in \mathbb{R}^d$ as an arrow in $d$-dimensional space, pointing from the origin to the coordinates $(x_1, x_2, \ldots, x_d)$.

The dot product of two vectors measures how much they point in the same direction:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{d} u_i v_i$$

If both vectors point exactly the same way, the dot product is large and positive. If they point in opposite directions, it is large and negative. If they are perpendicular, it is zero.

### 0.1.2 Formal Definition

For vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$:

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^{\top}\mathbf{v} = \sum_{i=1}^{d} u_i v_i$$

The dot product relates to vector lengths through:

$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

where $\|\mathbf{u}\| = \sqrt{\mathbf{u} \cdot \mathbf{u}}$ is the Euclidean length of $\mathbf{u}$ and $\theta$ is the angle between the two vectors.

### 0.1.3 Cosine Similarity

Normalizing by both lengths gives the **cosine similarity**:

$$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|} = \cos\theta$$

Cosine similarity lives in $[-1, 1]$. In node embedding chapters, we will use it constantly to measure how similar two nodes' learned representations are, independent of their magnitudes.

---

## 0.2 Matrix Multiplication

### 0.2.1 Intuition

A matrix is a rectangular array of numbers. In graph machine learning, the two most important matrices are:

- The **feature matrix** $\mathbf{X} \in \mathbb{R}^{n \times d}$, where row $i$ holds the $d$-dimensional feature vector for node $i$ among $n$ nodes.
- The **adjacency matrix** $\mathbf{A} \in \mathbb{R}^{n \times n}$, where $A_{ij} = 1$ if there is an edge between nodes $i$ and $j$.

Multiplying $\mathbf{A}\mathbf{X}$ produces a new matrix whose $i$-th row is the *sum of feature vectors of all neighbors of node $i$*. This single line of linear algebra is the core operation inside every Graph Convolutional Network.

### 0.2.2 Formal Definition

Given $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$, their product $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}$ has entries:

$$C_{ij} = \sum_{l=1}^{k} A_{il} B_{lj}$$

The inner dimension $k$ must match. Each entry $C_{ij}$ is a dot product of row $i$ of $\mathbf{A}$ with column $j$ of $\mathbf{B}$.

### 0.2.3 Properties

- **Not commutative:** $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ in general.
- **Associative:** $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$.
- **Transpose of product:** $(\mathbf{A}\mathbf{B})^{\top} = \mathbf{B}^{\top}\mathbf{A}^{\top}$.

### 0.2.4 The Graph Convolution Connection

Consider an undirected graph with 4 nodes and adjacency matrix:

$$\mathbf{A} = \begin{pmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 1 & 1 \\ 1 & 1 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}$$

If each node has a scalar feature $\mathbf{x} = (1, 2, 3, 4)^{\top}$, then $\mathbf{A}\mathbf{x} = (5, 8, 3, 2)^{\top}$. Row 1 of the result is $2 + 3 = 5$ — the sum of node 1's neighbors' features. This is *message passing* reduced to a matrix multiply.

---

## 0.3 Matrix Transpose

The **transpose** of a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ is the matrix $\mathbf{A}^{\top} \in \mathbb{R}^{n \times m}$ obtained by flipping rows and columns: $(\mathbf{A}^{\top})_{ij} = A_{ji}$.

**Key properties:**

- $(\mathbf{A}^{\top})^{\top} = \mathbf{A}$
- $(\mathbf{A} + \mathbf{B})^{\top} = \mathbf{A}^{\top} + \mathbf{B}^{\top}$
- $(\mathbf{A}\mathbf{B})^{\top} = \mathbf{B}^{\top}\mathbf{A}^{\top}$

In PyTorch, transposing a 2-D tensor is written `A.T` or `A.transpose(0, 1)`.

---

## 0.4 Matrix Rank

The **rank** of a matrix is the number of linearly independent rows (equivalently, the number of linearly independent columns). Intuitively, it counts the true dimensionality of the information the matrix contains.

For a feature matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$, rank tells us whether any feature is a linear combination of others. A low-rank matrix can be approximated by a smaller set of vectors — this is the insight behind matrix factorization methods for node embeddings, which we will explore in Chapter 4.

If $\text{rank}(\mathbf{A}) = r$, then $\mathbf{A}$ can be written as a product of two matrices of dimensions $m \times r$ and $r \times n$ — a *low-rank factorization*. This saves memory and computation when $r \ll \min(m, n)$.

---

## 0.5 Symmetric Matrices and Positive Semi-Definiteness

### 0.5.1 Symmetric Matrices

A matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is **symmetric** if $\mathbf{A} = \mathbf{A}^{\top}$, meaning $A_{ij} = A_{ji}$ for all $i, j$. The adjacency matrix of an undirected graph is always symmetric — if there is an edge from $i$ to $j$, there is one from $j$ to $i$. Symmetric matrices have two beautiful properties:

1. All eigenvalues are *real*.
2. Eigenvectors corresponding to distinct eigenvalues are *orthogonal*.

These properties make symmetric matrices tractable for spectral analysis.

### 0.5.2 Positive Semi-Definite Matrices

A symmetric matrix $\mathbf{M} \in \mathbb{R}^{n \times n}$ is **positive semi-definite (PSD)** if for every vector $\mathbf{x} \in \mathbb{R}^n$:

$$\mathbf{x}^{\top}\mathbf{M}\mathbf{x} \geq 0$$

Equivalently, all eigenvalues of $\mathbf{M}$ are non-negative. The **graph Laplacian** matrix $\mathbf{L} = \mathbf{D} - \mathbf{A}$ (where $\mathbf{D}$ is the diagonal degree matrix) is PSD — a fact we will use repeatedly in spectral GNN chapters. PSD matrices define valid notions of "distance" and "smoothness" on graphs.

---

## 0.6 Eigenvalue Decomposition

!!! mascot-thinking "Why Eigenvectors See What Eyes Cannot"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinking about eigenvalues">
    Eigenvalue decomposition is one of the most surprising ideas in linear algebra: every symmetric matrix is secretly just a stretching operation along a set of special directions. Those directions — the eigenvectors — often correspond to meaningful structure in the data. In graphs, the eigenvectors of the Laplacian encode which nodes form communities, which nodes are central, and how signals diffuse through the network. Understanding this connection is what separates spectral GNN practitioners from those who just apply formulas.

### 0.6.1 Intuition

Multiplying a vector $\mathbf{v}$ by a matrix $\mathbf{A}$ generally changes both the *direction* and the *length* of $\mathbf{v}$. But for certain special vectors — the **eigenvectors** — multiplying by $\mathbf{A}$ only changes the length, not the direction:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

The scalar $\lambda$ is the corresponding **eigenvalue**. The eigenvector points along a direction that the matrix leaves invariant; the eigenvalue says by how much the matrix stretches or shrinks along that direction.

### 0.6.2 Formal Derivation

For a symmetric matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, the eigenvalue equation $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ has $n$ solutions (counting multiplicity). Rearranging:

$$(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$$

This has a nontrivial solution $\mathbf{v} \neq \mathbf{0}$ exactly when $\mathbf{A} - \lambda\mathbf{I}$ is singular, i.e., when $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$. This **characteristic polynomial** of degree $n$ yields the $n$ eigenvalues.

Stacking all $n$ eigenvectors as columns of $\mathbf{U}$ and all eigenvalues along the diagonal of $\boldsymbol{\Lambda}$, we get the **eigendecomposition** (also called **spectral decomposition**):

$$\mathbf{A} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{\top}$$

where $\mathbf{U}^{\top}\mathbf{U} = \mathbf{I}$ (the eigenvectors form an orthonormal basis) and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$.

### 0.6.3 Why This Matters for GNNs

The graph Laplacian $\mathbf{L} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{\top}$ encodes the graph's geometry in its eigenvectors. The eigenvector corresponding to the smallest nonzero eigenvalue (the **Fiedler vector**) partitions the graph into two clusters with minimal cut — this is the basis of spectral clustering and the motivation for the spectral convolution in the original GCN paper.

---

## 0.7 Singular Value Decomposition

The **singular value decomposition (SVD)** generalizes eigendecomposition to *rectangular* matrices. For any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ (not necessarily square or symmetric):

$$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$$

where:

- $\mathbf{U} \in \mathbb{R}^{m \times m}$ has orthonormal columns (left singular vectors)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$ is diagonal with non-negative **singular values** $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $\mathbf{V} \in \mathbb{R}^{n \times n}$ has orthonormal columns (right singular vectors)

**Low-rank approximation:** The best rank-$k$ approximation to $\mathbf{A}$ (in Frobenius norm) is obtained by keeping only the top-$k$ singular values and their vectors: $\mathbf{A}_k = \mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^{\top}$. This theorem (Eckart–Young–Mirsky) underlies matrix factorization methods for node embeddings: the embedding of node $i$ is row $i$ of $\mathbf{U}_k\sqrt{\boldsymbol{\Sigma}_k}$, capturing the dominant structure of the adjacency or transition matrix.

---

## 0.8 Gradient Descent and the Chain Rule

!!! mascot-encourage "This Is Where It Gets Challenging — Stay With It"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage being encouraging">
    Backpropagation and gradient descent are the engine of all modern deep learning. If the math feels unfamiliar, that is completely normal — these ideas took decades for the research community to get right. Work through the derivation slowly. Every formula here will reappear in every GNN training loop you write.

### 0.8.1 Intuition: Moving Downhill

Imagine a loss landscape — a surface where the height at each point represents how wrong the model's predictions are. We want to find the lowest point (minimum loss). Gradient descent does this by repeatedly taking small steps in the direction of steepest descent.

The **gradient** $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ of a scalar loss $\mathcal{L}$ with respect to parameters $\boldsymbol{\theta}$ is the vector of partial derivatives:

$$\nabla_{\boldsymbol{\theta}} \mathcal{L} = \left(\frac{\partial \mathcal{L}}{\partial \theta_1}, \frac{\partial \mathcal{L}}{\partial \theta_2}, \ldots, \frac{\partial \mathcal{L}}{\partial \theta_p}\right)$$

It always points in the direction of *steepest ascent*. To minimize, we move in the opposite direction.

### 0.8.2 Gradient Descent Update Rule

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}$$

where $\eta > 0$ is the **learning rate** — how large a step to take. Too large and the update overshoots the minimum; too small and training is prohibitively slow.

**Stochastic Gradient Descent (SGD)** approximates the true gradient using a single example or a mini-batch of examples:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{batch}}$$

This approximation is noisy but much faster per step than computing the exact gradient over the full dataset.

### 0.8.3 The Chain Rule

Neural networks compose multiple functions: $\mathcal{L} = f_L(f_{L-1}(\cdots f_1(\mathbf{x})\cdots))$. Computing $\partial \mathcal{L} / \partial \mathbf{x}$ requires the **chain rule of calculus**:

$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

In the multivariate setting, if $\mathcal{L} = f(\mathbf{z})$ and $\mathbf{z} = g(\mathbf{x})$, then:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \sum_j \frac{\partial \mathcal{L}}{\partial z_j} \frac{\partial z_j}{\partial x_i}$$

This chain of multiplications connects the gradient of the loss at the output to the gradient with respect to any intermediate variable.

---

## 0.9 Backpropagation

**Backpropagation** is the chain rule applied *efficiently* to neural networks. The key insight is that computing gradients for all parameters takes only about as long as a single forward pass, by reusing intermediate computations.

### 0.9.1 The Computational Graph

A neural network defines a **computational graph**: a directed acyclic graph where:

- Nodes represent intermediate values (activations, weights, loss)
- Edges represent operations (matrix multiply, ReLU, etc.)

A forward pass computes the loss by following the graph from inputs to output. A backward pass propagates gradients from the loss back to the parameters.

### 0.9.2 Derivation for a Two-Layer Network

Consider a network $\hat{y} = W_2 \sigma(W_1 x)$ where $\sigma$ is a nonlinear activation (e.g., ReLU), with mean-squared-error loss $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$.

**Forward pass:**

$$\mathbf{z} = \mathbf{W}_1\mathbf{x}, \quad \mathbf{h} = \sigma(\mathbf{z}), \quad \hat{y} = \mathbf{W}_2\mathbf{h}, \quad \mathcal{L} = \tfrac{1}{2}(\hat{y} - y)^2$$

**Backward pass** (applying chain rule from right to left):

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = (\hat{y} - y)$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \mathbf{h}^{\top}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}} = \mathbf{W}_2^{\top} \cdot \frac{\partial \mathcal{L}}{\partial \hat{y}}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}} \odot \sigma'(\mathbf{z})$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \cdot \mathbf{x}^{\top}$$

Each parameter's gradient is a product of terms computed during the forward pass. This is why storing intermediate activations during the forward pass is necessary for efficient backpropagation.

---

## 0.10 Automatic Differentiation

**Automatic differentiation (autograd)** does this chain-rule bookkeeping automatically. When you write `y = W @ x + b` in PyTorch, the framework records the operations in a dynamic computation graph. Calling `.backward()` on the loss triggers a traversal of this graph in reverse, accumulating gradients via the chain rule.

### 0.10.1 Reverse Mode vs. Forward Mode

There are two modes of automatic differentiation:

| Mode | Cost | Best when |
|------|------|-----------|
| **Reverse mode** | One backward pass per output | Many parameters, one loss scalar — the standard ML case |
| **Forward mode** | One forward pass per input dimension | Few inputs, many outputs — rare in ML |

PyTorch uses reverse mode (also called reverse-mode autodiff or backprop). The gradient of a scalar loss with respect to millions of parameters is computed in a single backward pass, regardless of the number of parameters. This is the computational miracle that makes training deep networks feasible.

---

## 0.11 Neural Network Building Blocks

### 0.11.1 Linear Layers

A **linear layer** (also called a fully connected or dense layer) applies an affine transformation:

$$\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

where $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and $\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}$ are learnable parameters. Stacking linear layers with nonlinear activations between them creates a universal function approximator.

In graph neural networks, the linear layer appears inside the message transformation: each node's features are linearly projected before aggregation. You will see `nn.Linear` in virtually every GNN implementation in this textbook.

### 0.11.2 The Softmax Function

The **softmax function** maps a vector of raw scores (logits) $\mathbf{z} \in \mathbb{R}^K$ to a valid probability distribution over $K$ classes:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Each output is in $(0, 1)$ and all outputs sum to 1. The softmax is differentiable everywhere, making it suitable as the final layer in classification networks.

**Numerical stability:** In practice, we subtract the maximum value before exponentiating to avoid overflow: $\text{softmax}(\mathbf{z})_k = e^{z_k - \max(\mathbf{z})} / \sum_j e^{z_j - \max(\mathbf{z})}$. PyTorch handles this automatically.

### 0.11.3 Cross-Entropy Loss

For multi-class classification, the **cross-entropy loss** measures the discrepancy between predicted probabilities $\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})$ and the true one-hot label $\mathbf{y}$:

$$\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k \log \hat{y}_k = -\log \hat{y}_{y^*}$$

where $y^*$ is the true class. Only the log probability assigned to the correct class matters. Minimizing cross-entropy is equivalent to maximizing the log-likelihood of the labels under the model — this is the standard supervised learning objective for node classification.

PyTorch combines softmax and cross-entropy into `nn.CrossEntropyLoss`, which takes raw logits (not softmax outputs) for numerical stability.

---

## 0.12 The Adam Optimizer

Plain gradient descent works, but it treats all parameters identically with the same learning rate. **Adam** (Adaptive Moment Estimation) adapts the learning rate for each parameter individually, leading to faster convergence in practice.

### 0.12.1 Intuition

Adam tracks two running statistics for each parameter's gradient:

- $m_t$: the exponential moving average of gradients (like momentum — which direction have we been going?)
- $v_t$: the exponential moving average of squared gradients (how noisy are the gradients here?)

Parameters with consistently large gradients get a smaller effective learning rate. Parameters with small or noisy gradients get a larger step. This adaptivity is especially useful in GNNs where different parts of the graph may have very different gradient scales.

### 0.12.2 Update Rule

Given hyperparameters $\eta$ (learning rate), $\beta_1$ (default 0.9), $\beta_2$ (default 0.999), $\epsilon$ (default $10^{-8}$):

At step $t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Apply bias correction (since $m_0 = v_0 = 0$, early estimates are biased toward zero):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:

$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

In PyTorch: `optimizer = torch.optim.Adam(model.parameters(), lr=0.01)`.

---

## 0.13 PyTorch: Tensors and Autograd

!!! mascot-tip "PyTorch Is Your Graph ML Toolkit"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage giving a helpful tip">
    PyTorch's two killer features for deep learning are its *dynamic computational graph* and *NumPy-like tensor operations on GPU*. Almost every GNN in this textbook is implemented in PyTorch. Spend time understanding `requires_grad`, `.backward()`, and `.grad` — these three concepts underpin every training loop you will write.

### 0.13.1 Tensors

A **tensor** is a generalization of vectors and matrices to arbitrary dimensions. A scalar is a 0-dimensional tensor, a vector is 1-dimensional, a matrix is 2-dimensional, and so on. PyTorch tensors live on CPU or GPU and support the full range of linear algebra operations.

```python
import torch

# Creating tensors
x = torch.tensor([1.0, 2.0, 3.0])          # 1-D tensor (vector)
A = torch.randn(4, 4)                        # 2-D tensor (matrix), random normal
I = torch.eye(4)                             # 4x4 identity matrix

# Basic operations
print(x @ x)                                 # dot product: tensor(14.)
print(A @ I)                                 # matrix multiply: equals A
print(A.T)                                   # transpose

# Eigendecomposition of a symmetric matrix
S = A + A.T                                  # make symmetric
eigenvalues, eigenvectors = torch.linalg.eigh(S)
print("Eigenvalues:", eigenvalues)
print("Reconstruction error:",
      torch.norm(S - eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T))
```

### 0.13.2 PyTorch Autograd

The `requires_grad=True` flag tells PyTorch to track operations on a tensor so it can compute gradients later.

```python
import torch
import torch.nn as nn

# ----- Autograd demonstration -----
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3 - 2 * x                          # y = x^3 - 2x
y.backward()                                 # computes dy/dx
print(x.grad)                                # dy/dx = 3x^2 - 2 = 10 at x=2: tensor([10.])

# ----- Complete training loop example -----
torch.manual_seed(42)

# Synthetic dataset: y = 2x + noise
N = 100
X_data = torch.randn(N, 1)
y_data = 2 * X_data + 0.1 * torch.randn(N, 1)

# Model: single linear layer
model = nn.Linear(in_features=1, out_features=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(200):
    optimizer.zero_grad()          # 1. Clear gradients from previous step
    y_pred = model(X_data)         # 2. Forward pass
    loss = loss_fn(y_pred, y_data) # 3. Compute loss
    loss.backward()                # 4. Backward pass (compute gradients)
    optimizer.step()               # 5. Update parameters

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
              f"W: {model.weight.item():.4f} | b: {model.bias.item():.4f}")
# Expected: W ≈ 2.0, b ≈ 0.0
```

### 0.13.3 Neural Network Layers in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Softmax and cross-entropy
logits = torch.tensor([[2.0, 1.0, 0.1],      # raw scores for sample 1
                        [0.5, 2.5, 0.3]])     # raw scores for sample 2
probs = F.softmax(logits, dim=-1)
print("Probabilities:", probs)
# tensor([[0.6590, 0.2424, 0.0986],
#         [0.1142, 0.7721, 0.1136]])

true_labels = torch.tensor([0, 1])            # correct classes
loss = F.cross_entropy(logits, true_labels)   # combines softmax + NLL
print("Cross-entropy loss:", loss.item())     # ≈ 0.4170

# ------ Two-layer MLP for node classification ------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)    # return logits; loss fn applies softmax

# Example: 34 nodes, 16 features, 2 classes (Karate Club)
model = MLP(in_dim=16, hidden_dim=32, out_dim=2)
x = torch.randn(34, 16)       # node feature matrix
out = model(x)                # shape: (34, 2)
print("Output shape:", out.shape)
```

### 0.13.4 Using SVD in PyTorch

```python
# Singular value decomposition
A = torch.randn(6, 4)
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
# Reconstruct: A ≈ U @ diag(S) @ Vh
reconstructed = U @ torch.diag(S) @ Vh
print("SVD reconstruction error:", torch.norm(A - reconstructed).item())

# Low-rank approximation: keep top-2 singular values
k = 2
A_approx = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
print(f"Rank-{k} approximation error:", torch.norm(A - A_approx).item())
# This is the Eckart-Young-Mirsky theorem in action
```

---

## 0.14 MicroSim: Matrix × Graph Explorer

The interactive simulation below lets you visualize how matrix multiplication transforms graph features. Select a small graph, modify its adjacency matrix, and watch how \(\mathbf{A}\mathbf{X}\) (neighbor aggregation), \(\mathbf{D}^{-1}\mathbf{A}\mathbf{X}\) (mean aggregation), and \(\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\mathbf{X}\) (symmetric normalization used in GCN) produce different results. Sliders let you toggle between normalization modes in real time.

#### Diagram: Matrix × Graph Explorer


<iframe src="../../sims/ch00-matrix-graph-explorer/main.html" width="100%" height="572px" scrolling="no"></iframe>
[Run Matrix × Graph Explorer Fullscreen](../../sims/ch00-matrix-graph-explorer/main.html)

---

## 0.15 Benchmarks: PyTorch Ecosystem for Graph ML (2024–2025)

The table below compares the primary libraries you will use in this textbook for different graph ML tasks, based on benchmark results and adoption statistics from recent literature.

| Library | Primary Use | Graphs/sec (ogbn-arxiv) | Key Strength | Citation |
|---------|-------------|------------------------|--------------|----------|
| **PyTorch Geometric (PyG)** | GNN research & prototyping | ~180 | Widest model coverage, research-first API | Fey & Lenssen, 2019 |
| **Deep Graph Library (DGL)** | Large-scale training | ~220 | Memory-efficient, heterogeneous graph support | Wang et al., 2019 |
| **NetworkX** | Graph analysis & visualization | N/A (CPU-only) | Ease of use, 70+ algorithms built-in | Hagberg et al., 2008 |
| **Open Graph Benchmark (OGB)** | Standardized evaluation | — | Leaderboards, data loaders, fair comparison | Hu et al., 2020 |
| **PyTorch (core)** | Custom operations | Baseline | Full autograd, CUDA, production-ready | Paszke et al., 2019 |

*Throughput measured on ogbn-arxiv node classification with a 3-layer GCN, batch size 1024, on an NVIDIA A100 GPU. Numbers from Hu et al. (2020) and updated leaderboard results (2024). DGL and PyG are generally within 20% of each other; the choice often comes down to API preference.*

**Throughout this textbook** we use PyTorch Geometric as the primary implementation framework. The API is research-friendly, the model zoo is comprehensive, and every major GNN architecture — GCN, GraphSAGE, GAT, GIN, Graph Transformers — is available as a single import.

---

## 0.16 Common Pitfalls

!!! mascot-warning "Five Mistakes That Will Cost You Hours"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning about common mistakes">
    These are not hypothetical — every GNN practitioner makes these mistakes at some point. Knowing them in advance saves significant debugging time.

**1. Forgetting `optimizer.zero_grad()`**

PyTorch *accumulates* gradients by default. If you call `loss.backward()` twice without zeroing, gradients from both calls are added together — your parameter updates will be wrong. Always call `optimizer.zero_grad()` at the start of each training step.

**2. Using softmax outputs with `nn.CrossEntropyLoss`**

`nn.CrossEntropyLoss` applies softmax internally. If you pass softmax outputs instead of raw logits, you get `log(softmax(softmax(x)))` — numerically unstable and semantically wrong. Always pass raw logits to cross-entropy loss functions.

**3. Applying eigendecomposition to the wrong matrix**

The adjacency matrix $\mathbf{A}$ and the Laplacian $\mathbf{L} = \mathbf{D} - \mathbf{A}$ have very different eigenspectra. GCN uses $\hat{\mathbf{A}} = \mathbf{D}^{-1/2}(\mathbf{A} + \mathbf{I})\mathbf{D}^{-1/2}$ (self-loops added, then normalized). Each normalization choice changes which frequencies of the graph signal are amplified or damped.

**4. Shape mismatches from incorrect transpose**

A matrix multiply $\mathbf{A}\mathbf{B}$ requires the inner dimensions to match: `A.shape[-1] == B.shape[-2]`. In batched settings (3-D tensors for mini-batched graphs), dimension indexing is easy to get wrong. Use `assert` statements to verify shapes at every layer during development.

**5. Confusing eigenvalues and singular values**

For non-symmetric matrices, eigenvalues can be complex. SVD always produces real, non-negative singular values and is the numerically stable general-purpose decomposition. Reserve `torch.linalg.eig` for symmetric matrices; use `torch.linalg.svd` for everything else.

---

## 0.17 Chapter Summary

The table below collects the core formulas introduced in this chapter. You will see every one of them again — often in disguised forms — as we build graph neural networks in subsequent chapters.

| Concept | Formula | Role in GNNs |
|---------|---------|-------------|
| Dot product | $\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i$ | Attention scores (GAT, Transformers) |
| Cosine similarity | $\cos\theta = \mathbf{u}^{\top}\mathbf{v} / (\|\mathbf{u}\|\|\mathbf{v}\|)$ | Node similarity metrics, link prediction |
| Matrix multiplication | $C_{ij} = \sum_l A_{il}B_{lj}$ | Neighborhood aggregation: $\mathbf{A}\mathbf{X}$ |
| Eigendecomposition | $\mathbf{A} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^{\top}$ | Spectral convolution, Laplacian PE |
| SVD | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$ | Matrix factorization embeddings |
| Gradient descent | $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla\mathcal{L}$ | All GNN training |
| Chain rule | $\partial\mathcal{L}/\partial\theta_i = \sum_j (\partial\mathcal{L}/\partial z_j)(\partial z_j/\partial\theta_i)$ | Backpropagation |
| Softmax | $\text{softmax}(\mathbf{z})_k = e^{z_k}/\sum_j e^{z_j}$ | Node classification output |
| Cross-entropy | $\mathcal{L}_{\text{CE}} = -\log \hat{y}_{y^*}$ | Node/graph classification loss |
| Adam | $\theta_t = \theta_{t-1} - \eta\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)$ | Adaptive optimization |

---

## 0.18 Exercises

*12 exercises across all six levels of Bloom's Taxonomy.*

### Remember

**0.1.** Write the formula for the eigendecomposition of a symmetric matrix $\mathbf{A}$ and state three properties that hold because $\mathbf{A}$ is symmetric (not just square).

**0.2.** List the five steps of a PyTorch training loop in order. For each step, name the function or method call that performs it.

### Understand

**0.3.** The graph Laplacian is $\mathbf{L} = \mathbf{D} - \mathbf{A}$, where $\mathbf{D}$ is the diagonal degree matrix. Explain, without using formulas, why the Laplacian is always positive semi-definite. (*Hint: think about what $\mathbf{x}^{\top}\mathbf{L}\mathbf{x}$ counts over all edges.*)

**0.4.** In reverse-mode automatic differentiation, why is computing gradients for a scalar loss with respect to one million parameters faster than computing them in forward mode? Explain using the concept of a single backward pass.

### Apply

**0.5.** Given a 4-node path graph with adjacency matrix

$$\mathbf{A} = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

and feature vector $\mathbf{x} = (1, 2, 3, 4)^{\top}$, compute $\mathbf{A}\mathbf{x}$ by hand. Then compute the degree-normalized version $\mathbf{D}^{-1}\mathbf{A}\mathbf{x}$ (where $\mathbf{D}$ is the degree matrix). What does the normalization change about the result for node 2 (the hub with degree 2)?

**0.6.** Implement a function `cosine_similarity_matrix(X)` in PyTorch that takes a node feature matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ and returns an $n \times n$ matrix of pairwise cosine similarities. Use only `torch` operations — no loops. Verify your implementation on a small example.

### Analyze

**0.7.** Adam uses two moment estimates ($m_t$ for first moment, $v_t$ for second moment). Analyze what happens if $\beta_2$ is set very close to 1 (e.g., 0.9999). How does this affect the effective learning rate for a parameter that has occasional large gradient spikes versus one with steady small gradients? Which setting would you prefer for training a GNN on a sparse graph and why?

**0.8.** The SVD of a matrix $\mathbf{A}$ and the eigendecomposition of $\mathbf{A}^{\top}\mathbf{A}$ are closely related. Derive this relationship explicitly: if $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$, what are the eigenvalues and eigenvectors of $\mathbf{A}^{\top}\mathbf{A}$ in terms of $\mathbf{U}$, $\boldsymbol{\Sigma}$, $\mathbf{V}$?

### Evaluate

**0.9.** A colleague argues: "Gradient descent with a very small learning rate always converges to the global minimum if you run it long enough." Evaluate this claim. Under what conditions is it true? Under what conditions does it fail, and why? How does this affect the choice of optimizer in practice for GNNs trained on large graphs?

**0.10.** Compare the numerical behavior of computing cross-entropy loss via (a) applying `F.softmax` first then taking `torch.log`, versus (b) using `F.cross_entropy` directly on logits. Write a short experiment in PyTorch that shows a case where approach (a) gives `nan` or `inf` while approach (b) does not. Explain why the difference arises.

### Create

**0.11.** Design a PyTorch training loop for a node classification task on the Karate Club graph (34 nodes, 2 classes). Your design should include: (a) generating a feature matrix using random normal features, (b) a 2-layer MLP model, (c) cross-entropy loss on a manually specified train mask of 10 nodes, (d) Adam optimizer, and (e) accuracy evaluation on the remaining 24 nodes after 200 epochs. Implement and run this, reporting the train and test accuracy.

**0.12.** Using PyTorch's `torch.linalg.eigh`, compute the 4 smallest eigenvalues and corresponding eigenvectors of the Laplacian of the Karate Club graph. Plot the second eigenvector (the Fiedler vector) with nodes colored by their sign. Does the Fiedler vector partition the graph into the two Karate Club communities? If not, why might the spectral partition differ from the known social partition?

---

## 0.19 Further Reading

**[1] Paszke et al. (2019) — PyTorch: An Imperative Style, High-Performance Deep Learning Library.** *NeurIPS 2019.*
The original PyTorch paper describes the design philosophy behind dynamic computation graphs and reverse-mode autodiff. Understanding why PyTorch chose eager execution over graph compilation clarifies many API decisions you will encounter when implementing GNNs.

**[2] Kingma & Ba (2015) — Adam: A Method for Stochastic Optimization.** *ICLR 2015.*
The paper that introduced Adam, now the default optimizer in virtually all GNN training pipelines. The bias correction derivation is particularly worth reading — it explains why early training steps with Adam are more stable than with raw momentum SGD.

**[3] Hamilton (2020) — Graph Representation Learning (book).** *Free PDF available online.*
The most accessible textbook companion to this one. Chapter 2 covers the linear algebra of graphs (Laplacians, spectral decomposition) at a level of rigor appropriate for this course. Chapter 3 introduces shallow node embedding methods that directly apply SVD and gradient descent to graphs.

**[4] Bronstein, Bruna, Cohen & Veličković (2021) — Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.** *arXiv:2104.13478.*
Section 3 of this monograph places the linear algebra of this chapter in a unified geometric framework. Understanding symmetry, invariance, and equivariance is not strictly necessary for beginning practitioners, but it provides a compelling "why" behind GNN design choices.

**[5] Petersen & Pedersen — The Matrix Cookbook.** *Free PDF, continuously updated.*
A dense reference of matrix identities, derivative formulas, and decomposition results. Invaluable when deriving GNN gradients by hand. Keep it bookmarked — you will reach for it when computing analytical gradients in the GNN theory chapters.

**[6] Hu et al. (2020) — Open Graph Benchmark: Datasets for Machine Learning on Graphs.** *NeurIPS 2020.*
This paper introduces the benchmarking framework we use throughout the textbook. Understanding the train/validation/test split design (especially the temporal splits for link prediction) is crucial for fair evaluation. The paper also benchmarks PyTorch Geometric and DGL, providing concrete performance numbers for library comparison.

**[7] Fey & Lenssen (2019) — Fast Graph Representation Learning with PyTorch Geometric.** *ICLR Workshop 2019.*
The original PyTorch Geometric paper describes the `MessagePassing` base class that underlies every GNN implementation in Chapters 6–11. Read this early — it will make the PyG API feel less magical and more principled.

!!! mascot-celebration "You've Built Your Foundation!"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating">
    You have just armed yourself with every mathematical tool you will need for the rest of this textbook. Matrix multiplication, eigendecomposition, SVD, backpropagation, and a working PyTorch training loop — these are the building blocks of every graph neural network ever built. In Chapter 1, we turn to the graphs themselves: what they are, how to represent them, and what makes real-world networks so different from random ones. Your neighbors have insights too — let's go meet them!

[See Annotated References](./references.md)
