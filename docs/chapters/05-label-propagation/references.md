# References: Label Propagation and Semi-Supervised Learning

1. [Label Propagation Algorithm](https://en.wikipedia.org/wiki/Label_propagation_algorithm) - Wikipedia - Covers the family of iterative label propagation methods on graphs, including the Zhu & Ghahramani harmonic function formulation and the Zhou et al. label spreading variant, with discussion of convergence conditions.

2. [Belief Propagation](https://en.wikipedia.org/wiki/Belief_propagation) - Wikipedia - Explains the sum-product and max-product message-passing algorithms on factor graphs, including exact inference on trees and the loopy BP approximation used when graphs contain cycles.

3. [SIR Model](https://en.wikipedia.org/wiki/SIR_model) - Wikipedia - Describes the Susceptible-Infected-Recovered compartmental epidemic model, its differential equation formulation, the basic reproduction number R₀, and extensions to network-structured populations.

4. Semi-Supervised Learning - Chapelle, O., Schölkopf, B., & Zien, A. (Eds.) - MIT Press - Comprehensive edited volume covering graph-based methods, manifold regularization, transductive SVMs, and generative approaches; Chapter 11 gives the definitive treatment of Gaussian fields and label propagation.

5. Networks, Crowds, and Markets: Reasoning About a Highly Connected World - Easley, D. & Kleinberg, J. - Cambridge University Press - Rigorous undergraduate-accessible text unifying game theory, economics, and network science; Chapters 19–21 cover diffusion, cascades, and epidemic models with the same formalism used in this chapter.

6. [Semi-supervised Classification with Graph Convolutional Networks (Kipf & Welling, 2017)](https://arxiv.org/abs/1609.02907) - arXiv - Introduces the GCN spectral convolution and frames node classification as a semi-supervised problem; the label propagation interpretation of GCN motivates the direct connection between this chapter and GNN foundations.

7. [Predict then Propagate: Graph Neural Networks meet Personalized PageRank (Gasteiger et al., 2019)](https://arxiv.org/abs/1810.05997) - arXiv - Derives APPNP by replacing GCN aggregation with personalized PageRank diffusion, showing that separating feature transformation from label propagation improves both depth and accuracy.

8. [Combining Label Propagation and Simple Models Out-performs Graph Neural Networks (Huang et al., 2021)](https://arxiv.org/abs/2010.13993) - arXiv - Introduces the Correct & Smooth (C&S) post-processing framework that applies two rounds of label diffusion on top of a plain MLP, achieving state-of-the-art results on ogbn-arxiv with minimal compute.

9. [Node Classification — PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/node_classification.html) - PyTorch Geometric Docs - Step-by-step tutorial for semi-supervised node classification using GCN and related models on the Cora dataset, with runnable code that bridges the label propagation ideas in this chapter to practical GNN implementations.

10. [Papers With Code — Semi-Supervised Node Classification](https://paperswithcode.com/task/semi-supervised-node-classification) - Papers With Code - Tracks state-of-the-art results and links to reproducible implementations for semi-supervised node classification benchmarks including Cora, CiteSeer, and ogbn-arxiv, making it easy to compare label propagation baselines against current GNN methods.
