# References: Math and Programming Prerequisites

1. [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) - Wikipedia - A thorough article covering the theory, geometric interpretation, and computational applications of SVD. Includes worked examples for dimensionality reduction and low-rank approximation that directly foreshadow Chapter 4's node embedding methods.

2. [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) - Wikipedia - Covers the chain-rule derivation of gradients through a computational graph, the algorithm's history, and common variants. The pseudocode section is a useful sanity check when implementing manual gradient passes in PyTorch.

3. [Eigenvalues and Eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) - Wikipedia - Thorough treatment of eigendecomposition for square matrices, including geometric intuition, characteristic polynomial, and the spectral theorem for symmetric matrices. Essential background for understanding the graph Laplacian and spectral GNNs introduced in Chapter 6.

4. Mathematics for Machine Learning - Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong - Cambridge University Press - A rigorous but accessible treatment of linear algebra, calculus, probability, and optimization all in one volume, with a consistent machine-learning framing throughout. The chapters on matrix decompositions and continuous optimization are especially relevant prerequisites for this textbook.

5. Deep Learning - Ian Goodfellow, Yoshua Bengio, Aaron Courville - MIT Press - The canonical graduate-level textbook for deep learning; Part I (Chapters 2–5) covers linear algebra, probability, numerical computation, and machine learning foundations at the depth needed before reading any GNN chapter. Freely available online at deeplearningbook.org.

6. [PyTorch Documentation — torch.Tensor](https://pytorch.org/docs/stable/tensors.html) - PyTorch Official Docs - The authoritative API reference for PyTorch's core data structure. Covers creation, indexing, broadcasting, device placement, and in-place vs. out-of-place operations — the mechanics you must understand before writing any GNN training loop.

7. [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) - NumPy Official Docs - The official guide to NumPy's array model, broadcasting rules, and linear-algebra submodule (`numpy.linalg`). A solid foundation here makes the transition to PyTorch tensors nearly seamless, since PyTorch deliberately mirrors NumPy's API.

8. [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) - Distill.pub - An interactive, visually rich article that motivates why the matrix operations covered in this chapter (adjacency multiplication, feature aggregation) naturally generalize to graph-structured data. Bridges the gap between the prerequisites here and the GNN formalism introduced in Chapter 6.

9. [CS231n: Linear Algebra Review and Reference](http://cs229.stanford.edu/section/cs229-linalg.pdf) - Stanford University (CS229) - A concise 23-page PDF covering vectors, matrices, eigenvalues, SVD, and matrix calculus at exactly the depth needed for deep learning. Particularly useful for the matrix-gradient identities required when deriving GNN backpropagation by hand.

10. [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) - arXiv:1502.05767 - A comprehensive survey of how modern frameworks implement automatic differentiation, covering forward-mode, reverse-mode (backpropagation), and their computational trade-offs. Understanding this paper clarifies why PyTorch's dynamic computation graph is both flexible and efficient for graph-structured inputs of varying size.
