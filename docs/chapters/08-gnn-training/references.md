# References: GNN Training, Augmentation, and Practical Tips

1. [Overfitting](https://en.wikipedia.org/wiki/Overfitting) - Wikipedia - Covers the bias-variance tradeoff and regularization strategies that underpin dropout, weight decay, and early stopping used in GNN training pipelines.

2. [Batch normalization](https://en.wikipedia.org/wiki/Batch_normalization) - Wikipedia - Explains the normalization technique that stabilizes deep network training; directly relevant to GraphNorm and other normalization layers applied to node embeddings between GNN layers.

3. [Dropout (regularization technique)](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) - Wikipedia - Describes the random unit-deactivation regularizer widely used in GNNs, including its graph-specific variants such as DropEdge and DropNode that operate on graph structure rather than hidden units.

4. Deep Learning - Ian Goodfellow, Yoshua Bengio, Aaron Courville - MIT Press - The authoritative deep-learning reference; Chapters 7–9 cover regularization, optimization, and normalization in detail, providing the theoretical grounding for dropout, batch normalization, and learning-rate schedules applied to GNNs.

5. Graph Representation Learning - William L. Hamilton - Morgan & Claypool Synthesis Lectures - Written by a lead GNN researcher; Chapter 5 addresses practical training considerations specific to graphs, including mini-batch strategies, neighbor sampling, and the over-smoothing problem that distinguishes GNN optimization from standard deep learning.

6. [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification (arXiv:1907.10903)](https://arxiv.org/abs/1907.10903) - arXiv - Introduces DropEdge, which randomly removes edges during training to reduce over-smoothing and over-fitting; provides theoretical analysis showing that edge dropping slows the convergence of node representations toward their over-smoothed limit.

7. [Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View (arXiv:1909.03211)](https://arxiv.org/abs/1909.03211) - arXiv - Proposes the Mean Average Distance (MAD) metric to quantify over-smoothing and introduces MADReg and adaptive edge masking to mitigate it, offering both diagnostics and remedies relevant to any deep GNN.

8. [Open Graph Benchmark: Datasets for Machine Learning on Graphs (arXiv:2005.00687)](https://arxiv.org/abs/2005.00687) - arXiv - Introduces OGB, the standardized suite of large-scale graph datasets (including ogbn-arxiv) with unified evaluation protocols; the canonical reference for reporting reproducible GNN training results.

9. [PyTorch Geometric Documentation — Advanced Mini-Batch Training](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html) - PyTorch Geometric Docs - Official documentation covering DataLoader construction, neighbor sampling (NeighborLoader, ClusterData), and batch normalization layers in PyG; the practical reference for implementing scalable GNN training loops.

10. [Papers With Code — Graph Classification Benchmarks](https://paperswithcode.com/task/graph-classification) - Papers With Code - Aggregates state-of-the-art results on graph classification and node classification tasks with linked code; useful for comparing augmentation strategies, normalization choices, and training tricks against reproducible baselines.
