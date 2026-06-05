# References: Designing Powerful Encoders: GIN and Beyond

1. [Graph Isomorphism](https://en.wikipedia.org/wiki/Graph_isomorphism) - Wikipedia - Defines the graph isomorphism problem and explains when two graphs are structurally identical, which is the foundation for understanding why the Weisfeiler-Leman test matters and what it means for a GNN to be maximally expressive.

2. [Weisfeiler Leman graph isomorphism test](https://en.wikipedia.org/wiki/Weisfeiler_Leman_graph_isomorphism_test) - Wikipedia - Explains the 1-WL (color refinement) algorithm and its higher-order k-WL generalizations, which serve as the theoretical yardstick against which GNN expressiveness is measured throughout this chapter.

3. [Graph neural network](https://en.wikipedia.org/wiki/Graph_neural_network) - Wikipedia - Broad overview of GNN architectures, aggregation functions, and expressive power, placing GIN and PNA in the wider landscape of message-passing networks and their connections to classical graph theory.

4. Deep Learning on Graphs - Ma & Tang - Cambridge University Press - Comprehensive graduate-level treatment of graph deep learning that devotes a full chapter to expressiveness, covering the 1-WL connection, injectivity requirements, and the construction of GIN from first principles with supporting proofs.

5. Graph Representation Learning - Hamilton - Morgan & Claypool Synthesis Lectures on AI and ML - Concise monograph covering the theoretical foundations of graph embeddings and message-passing GNNs, including a rigorous discussion of the Weisfeiler-Leman hierarchy and why sum aggregation is necessary (and sufficient) for 1-WL power.

6. [How Powerful are Graph Neural Networks? (Xu et al., 2019)](https://arxiv.org/abs/1810.00826) - arXiv - The seminal GIN paper proving that MPNNs are at most as powerful as 1-WL and constructing GIN as the unique maximally expressive MPNN; essential reading for understanding why sum aggregation and MLP updates are both required.

7. [Principal Neighbourhood Aggregation for Graph Nets (Corso et al., 2020)](https://arxiv.org/abs/2004.05718) - arXiv - Introduces PNA, which combines multiple aggregators (mean, max, min, std) scaled by degree to overcome the limitations of any single aggregation function; demonstrates significant gains on molecular benchmarks over GIN and other baselines.

8. [Equivariant Subgraph Aggregation Networks (Bevilacqua et al., 2022)](https://arxiv.org/abs/2110.02910) - arXiv - Proposes ESAN and the DS-WL hierarchy, showing that subgraph GNNs that process bags of subgraphs are strictly more powerful than 1-WL and can distinguish graphs like the Rook's graph that defeat all standard MPNNs.

9. [PyTorch Geometric — GINConv Documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html) - PyTorch Geometric Docs - Official API reference for `GINConv` and `GINEConv`, including the epsilon parameter, MLP construction patterns, and worked code examples showing how to stack GIN layers for graph classification tasks.

10. [CS224W Lecture 8: Graph Neural Networks (Stanford)](http://web.stanford.edu/class/cs224w/slides/08-GNN-expressiveness.pdf) - Stanford CS224W - Lecture slides covering GNN expressiveness, the 1-WL connection, and the construction of GIN; provides concise visual derivations of injectivity requirements and the role of sum vs. mean vs. max aggregation.
