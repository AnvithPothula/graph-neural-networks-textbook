# References: GNN Design Space: GraphSAGE and GAT

1. [Graph Neural Network](https://en.wikipedia.org/wiki/Graph_neural_network) - Wikipedia - Covers the foundational architecture of GNNs, including message passing, aggregation, and the distinction between spectral and spatial approaches. Useful orientation before diving into GraphSAGE and GAT as specific design instantiations.

2. [Attention Mechanism](https://en.wikipedia.org/wiki/Attention_(machine_learning)) - Wikipedia - Explains the general attention mechanism in machine learning, including the scaled dot-product formulation and softmax normalization. Provides the conceptual background needed to understand how GAT adapts attention to irregular graph neighborhoods.

3. [Inductive Learning](https://en.wikipedia.org/wiki/Transduction_(machine_learning)) - Wikipedia - Describes the transductive vs. inductive distinction in machine learning. GraphSAGE's core contribution is inductive node embedding; this article grounds the theoretical framing of that claim.

4. Deep Learning on Graphs - Ma & Tang - Cambridge University Press - A comprehensive textbook covering GNN foundations, spectral methods, and spatial methods including GraphSAGE and attention-based models. Chapter 5 provides a unified framework for comparing aggregation choices across architectures, directly relevant to the design space analysis in this chapter.

5. Graph Representation Learning - Hamilton - Morgan & Claypool Synthesis Lectures - A concise, rigorous treatment by the GraphSAGE author himself, covering the progression from shallow embeddings to deep GNNs. Chapter 4's derivation of inductive aggregation functions is the most authoritative source for understanding why the choice of aggregator matters theoretically and empirically.

6. [Hamilton et al. (2017) — GraphSAGE: Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) - arXiv - The original GraphSAGE paper introducing neighborhood sampling and aggregator-based inductive embedding. The theoretical section (Section 4) proves VC-dimension bounds showing generalization to unseen nodes, and the empirical study covers citation, Reddit, and protein-interaction benchmarks.

7. [Veličković et al. (2018) — Graph Attention Networks](https://arxiv.org/abs/1710.10903) - arXiv - Introduces GAT with learned attention coefficients over neighbors, enabling the model to assign different importances without requiring graph structure as input. Section 2.2 formalizes the attention computation and Table 1 provides a direct complexity comparison with GCN and other spatial methods.

8. [You et al. (2020) — Design Space for Graph Neural Networks](https://arxiv.org/abs/2011.08843) - arXiv - A systematic empirical study across 315,000 GNN configurations covering intra-layer design, inter-layer design, and learning configuration. Table 3's importance ranking of design choices is the most data-grounded reference available for making principled architectural decisions.

9. [PyTorch Geometric — SAGEConv and GATConv Documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv) - PyTorch Geometric Docs - Official API reference for `SAGEConv` and `GATConv`, the canonical implementations of GraphSAGE and GAT in PyG. Includes constructor arguments for aggregation type, attention heads, dropout, and edge feature support — essential when translating the chapter's theory into working code.

10. [Papers With Code — Node Classification Benchmark](https://paperswithcode.com/task/node-classification) - Papers With Code - Tracks state-of-the-art results on standard node classification benchmarks including ogbn-arxiv, Cora, and CiteSeer with linked papers and code. Useful for situating GraphSAGE and GAT results in context and identifying more recent successors (e.g., GATv2, SIGN, GAMLP) that build on the design choices covered in this chapter.
