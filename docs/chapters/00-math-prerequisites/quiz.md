# Quiz: Math and Programming Prerequisites

Test your understanding of the mathematical and computational foundations covered in this chapter.

---

#### Question 1

What is the geometric interpretation of the dot product \(\mathbf{u} \cdot \mathbf{v}\) for two vectors in \(\mathbb{R}^n\)?

<div class="upper-alpha" markdown>
1. The cross product of the two vectors projected onto a single axis
2. The product of their magnitudes multiplied by the cosine of the angle between them
3. The Euclidean distance between the endpoints of the vectors
4. The sum of the squared differences of their corresponding elements
</div>

??? question "Show Answer"
    The correct answer is **B**. The dot product \(\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos \theta\), where \(\theta\) is the angle between the vectors. This makes the dot product large when vectors point in the same direction and exactly zero when they are perpendicular (orthogonal). This geometric relationship is why dot products appear everywhere in GNNs: computing similarity, projecting features, and measuring alignment between representations.

    **Concept Tested:** Vector and Dot Product

---

#### Question 2

Which expression correctly defines cosine similarity between vectors \(\mathbf{u}\) and \(\mathbf{v}\)?

<div class="upper-alpha" markdown>
1. \(\|\mathbf{u} - \mathbf{v}\| / (\|\mathbf{u}\| \times \|\mathbf{v}\|)\)
2. \(\sum_i u_i v_i - \sum_i u_i + \sum_i v_i\)
3. \((\mathbf{u} \cdot \mathbf{v}) / (\|\mathbf{u}\| \times \|\mathbf{v}\|)\)
4. \(\|\mathbf{u} + \mathbf{v}\| / \|\mathbf{u} - \mathbf{v}\|\)
</div>

??? question "Show Answer"
    The correct answer is **C**. Cosine similarity divides the dot product by the product of the two vector norms, yielding a value in \([-1, 1]\). A score of 1 means the vectors point in exactly the same direction; 0 means they are orthogonal; −1 means they are exactly opposite. Because it normalizes out magnitude, cosine similarity measures directional (semantic) similarity — which is why it is the standard metric for comparing node embeddings.

    **Concept Tested:** Cosine Similarity

---

#### Question 3

Why does the matrix product \((AB)_{xy} = \sum_k A_{xk} B_{ky}\) require the number of columns in A to equal the number of rows in B?

<div class="upper-alpha" markdown>
1. So that the transpose of the product equals the product of the transposes in the same order
2. So that the resulting matrix is always square
3. So that the matrices can be stored in contiguous memory
4. So that each output element is a valid dot product between a row of A and a column of B
</div>

??? question "Show Answer"
    The correct answer is **D**. Each entry \((AB)_{xy}\) is the dot product of row \(x\) of \(A\) with column \(y\) of \(B\). For that dot product to be defined, both vectors must have the same length — which requires \(A\)'s column count to equal \(B\)'s row count. This inner-dimension constraint is fundamental to all GNN operations: the feature transformation at each layer is a matrix multiplication \(H^{(l+1)} = \hat{A} \cdot H^{(l)} \cdot W\), where dimensions must align.

    **Concept Tested:** Matrix Multiplication

---

#### Question 4

What does the backpropagation algorithm compute during neural network training?

<div class="upper-alpha" markdown>
1. The forward pass predictions for a batch of inputs
2. The eigenvalues of the weight matrices at each layer
3. The gradient of the loss with respect to every trainable parameter
4. The optimal learning rate for each layer automatically
</div>

??? question "Show Answer"
    The correct answer is **C**. Backpropagation applies the chain rule of calculus from the output layer backward through the network, computing \(\partial L / \partial \theta\) for every parameter \(\theta\). These gradients tell the optimizer which direction to adjust each weight to reduce the loss. Without backprop, training deep networks would require finite-difference approximation of every parameter — computationally infeasible for models with millions of weights.

    **Concept Tested:** Backpropagation

---

#### Question 5

What is the key advantage of automatic differentiation (autograd) over symbolic differentiation for deep learning?

<div class="upper-alpha" markdown>
1. It computes exact gradients efficiently by tracking operations at runtime rather than differentiating symbolic expressions
2. It only works with linear functions, making it significantly faster
3. It produces closed-form mathematical expressions for all derivatives
4. It eliminates the need for the chain rule by using finite-difference approximation
</div>

??? question "Show Answer"
    The correct answer is **A**. Automatic differentiation records operations in a computational graph during the forward pass, then applies the chain rule backward through that graph to compute exact gradients. This is both exact (unlike finite differences) and efficient (unlike symbolic differentiation, which produces exponentially growing expressions for deep networks). PyTorch's autograd engine implements reverse-mode autodiff, making gradient computation essentially free compared to the forward pass.

    **Concept Tested:** Automatic Differentiation

---

#### Question 6

A model's training loss decreases from 3.5 to 3.2 in epoch 2, then rises back to 3.5 in epoch 3. What is the most likely cause?

<div class="upper-alpha" markdown>
1. Backpropagation is computing incorrect gradients due to a numerical error
2. The model has converged and no further training is needed
3. The batch size is too small, causing the model to underfit
4. The learning rate is too high, causing the optimizer to overshoot the minimum
</div>

??? question "Show Answer"
    The correct answer is **D**. Oscillating loss — decreasing then increasing across epochs — is a hallmark of a learning rate that is too large. The optimizer takes steps so large that it overshoots the loss minimum, bouncing back and forth rather than converging smoothly. Reducing the learning rate (or using a learning rate scheduler such as cosine annealing) would smooth convergence toward a stable minimum.

    **Concept Tested:** Gradient Descent

---

#### Question 7

Why must activation functions in a multilayer perceptron (MLP) be non-linear?

<div class="upper-alpha" markdown>
1. Linear activation functions always cause vanishing gradients during backprop
2. Without non-linearity, stacking multiple layers is mathematically equivalent to a single linear transformation
3. Linear functions cannot process inputs with negative values correctly
4. Non-linear functions always produce sparse outputs, making computation faster
</div>

??? question "Show Answer"
    The correct answer is **B**. A composition of linear functions is itself linear: \(W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1)\mathbf{x} + (W_2 \mathbf{b}_1 + \mathbf{b}_2)\). No matter how many linear layers are stacked, the network can only represent affine mappings. Non-linear activations like ReLU or tanh break this equivalence, allowing MLPs to approximate arbitrary continuous functions (the universal approximation theorem).

    **Concept Tested:** Activation Function

---

#### Question 8

What distinguishes the Adam optimizer from standard stochastic gradient descent (SGD)?

<div class="upper-alpha" markdown>
1. Adam maintains exponential moving averages of both gradients and squared gradients to adapt the learning rate per parameter
2. Adam uses a fixed, global learning rate for all parameters throughout training
3. Adam uses second-order derivatives (the Hessian matrix) to scale each update
4. Adam replaces backpropagation with random parameter perturbations
</div>

??? question "Show Answer"
    The correct answer is **A**. Adam (Adaptive Moment Estimation) tracks \(m_t\), an exponential moving average of gradients (first moment), and \(v_t\), an exponential moving average of squared gradients (second moment). It then scales each parameter update by \(m_t / (\sqrt{v_t} + \varepsilon)\), adapting the effective step size to the gradient history of each parameter. This makes Adam converge much faster than SGD on most deep learning tasks without careful learning rate tuning.

    **Concept Tested:** Adam Optimizer

---

#### Question 9

For a symmetric matrix \(A\), the eigendecomposition \(A = Q \Lambda Q^\top\) is especially useful because:

<div class="upper-alpha" markdown>
1. It reduces any matrix multiplication involving \(A\) to a single scalar operation
2. It allows \(A\) to be inverted only when all eigenvalues are strictly positive
3. It guarantees that all eigenvalues are complex numbers with non-zero imaginary parts
4. The eigenvectors form an orthonormal basis, making the decomposition geometrically interpretable as rotation–scale–rotation
</div>

??? question "Show Answer"
    The correct answer is **D**. For symmetric matrices, the spectral theorem guarantees real eigenvalues and mutually orthogonal unit eigenvectors, so \(Q^\top = Q^{-1}\). This means \(A = Q \Lambda Q^\top\) is a pure rotate–scale–rotate operation: \(Q^\top\) rotates into the eigenbasis, \(\Lambda\) scales each dimension, and \(Q\) rotates back. This interpretation is central to understanding the graph Laplacian's spectral structure in later chapters.

    **Concept Tested:** Eigenvalue Decomposition

---

#### Question 10

In PyTorch Geometric, node features and graph connectivity are stored separately as `x` and `edge_index`. What is the primary reason for this design?

<div class="upper-alpha" markdown>
1. It ensures the library can only be applied to undirected graphs
2. It prevents gradient flow from edge features back to node features during backpropagation
3. It enables sparse message passing that scales to graphs with millions of nodes without materializing the full \(N \times N\) adjacency matrix
4. It simplifies implementation by requiring all graphs to be homogeneous
</div>

??? question "Show Answer"
    The correct answer is **C**. Storing `edge_index` as a COO sparse tensor of shape \([2, E]\) instead of a dense adjacency matrix of shape \([N, N]\) reduces memory from \(O(N^2)\) to \(O(E)\). For ogbn-arxiv with 170K nodes, a dense adjacency matrix would require roughly 27 GB — completely infeasible. PyG's `MessagePassing` layer dispatches computations along edges without ever constructing the full adjacency, making billion-node graphs tractable with neighbor sampling.

    **Concept Tested:** PyTorch Geometric (PyG)
