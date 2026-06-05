# References: Graph Transformers

1. [Transformer (machine learning model)](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) - Wikipedia - Covers the original Transformer architecture including multi-head self-attention, positional encodings, and the encoder-decoder structure. Provides essential background on the base mechanism that Graph Transformers adapt for graph-structured data.

2. [Graph neural network](https://en.wikipedia.org/wiki/Graph_neural_network) - Wikipedia - Broad overview of the GNN family including message-passing frameworks, expressiveness limitations relative to the Weisfeiler–Leman test, and key application domains. Contextualizes where Graph Transformers fit within the wider GNN landscape.

3. [Spectral graph theory](https://en.wikipedia.org/wiki/Spectral_graph_theory) - Wikipedia - Explains the graph Laplacian, its eigendecomposition, and the relationship between spectral properties and graph structure. Directly underpins Laplacian positional encodings (LapPE) used in SAN, GPS, and related Graph Transformer architectures.

4. Deep Learning on Graphs - Ma, Y. and Tang, J. - Cambridge University Press - A graduate-level textbook covering spectral and spatial GNNs, scalability, and applications; Chapter 7 discusses attention mechanisms on graphs and provides mathematical grounding for Graph Transformer design choices.

5. Graph Representation Learning - Hamilton, W. L. - Morgan & Claypool (Synthesis Lectures on AI and ML) - Concise treatment of node embeddings, GNN expressiveness, and the limitations of local aggregation. Provides theoretical foundations (Weisfeiler–Leman hierarchy, spectral convolutions) that motivate the move to global self-attention in Graph Transformers.

6. [Ying, C., et al. "Do Transformers Really Perform Bad for Graph Representation?" (Graphormer)](https://arxiv.org/abs/2106.05234) - arXiv - Introduces Graphormer, which incorporates centrality encoding, spatial encoding, and edge encoding into standard Transformer attention. Won the OGB-LSC molecular property prediction track at NeurIPS 2021, establishing Graph Transformers as competitive with GNNs on real benchmarks.

7. [Rampasek, L., et al. "Recipe for a General, Powerful, Scalable Graph Transformer" (GPS)](https://arxiv.org/abs/2205.12454) - arXiv - Proposes the GPS framework combining local MPNN layers with global self-attention and flexible positional/structural encodings. Systematic ablations across 11 datasets make this the primary reference for understanding when and why the hybrid design outperforms either component alone.

8. [Kreuzer, D., et al. "Rethinking Graph Transformers with Spectral Attention" (SAN)](https://arxiv.org/abs/2106.03893) - arXiv - Introduces SAN, which uses the full Laplacian eigenvector basis and separate attention for connected vs. disconnected pairs. Proves that spectral positional encodings provably improve expressiveness beyond 1-WL, providing theoretical justification for LapPE.

9. [PyTorch Geometric — GraphGPS and Positional Encodings documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GPS.html) - PyTorch Geometric Docs - Official API reference for the GPS model implementation in PyG, including configuration of MPNN type, attention type, and positional encoding modules (LapPE, RWSE). The practical starting point for running GPS experiments.

10. [Papers With Code — Graph Transformer benchmark leaderboards](https://paperswithcode.com/methods/category/graph-transformers) - Papers With Code - Aggregates published results for Graph Transformer variants across molecular, social, and citation benchmarks with links to code repositories. Useful for tracking state-of-the-art performance and comparing Graphormer, GPS, SAN, GRIT, and Exphormer on standardized splits.
