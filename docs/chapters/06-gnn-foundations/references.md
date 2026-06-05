# References: GNN Foundations: Message Passing and GCN

1. [Graph Neural Network](https://en.wikipedia.org/wiki/Graph_neural_network) - Wikipedia - Covers the history and taxonomy of graph neural networks, from early recurrent formulations through modern message-passing architectures. A reliable entry point for situating GCN within the broader GNN family.

2. [Laplacian Matrix](https://en.wikipedia.org/wiki/Laplacian_matrix) - Wikipedia - Defines the combinatorial, normalized, and symmetric normalized Laplacians with worked examples on small graphs. Essential background for understanding the spectral convolution derivation in this chapter.

3. [Spectral Graph Theory](https://en.wikipedia.org/wiki/Spectral_graph_theory) - Wikipedia - Explains the relationship between a graph's eigenspectrum and its structural properties, including connectivity, clustering, and random walks. Provides the mathematical foundation for why convolution in the spectral domain is well-defined.

4. Deep Learning on Graphs - Yao Ma and Jiliang Tang - Cambridge University Press - A graduate-level textbook dedicated to graph deep learning; Chapters 2–4 cover spectral and spatial GNNs with full derivations. The treatment of GCN, ChebConv, and MPNN closely parallels this chapter's content.

5. Graph Representation Learning - William L. Hamilton - Morgan & Claypool (Synthesis Lectures on AI and ML) - A concise monograph covering node embeddings through GNNs; freely available from the author's website. Chapter 4 derives GCN from first principles and provides one of the clearest expositions of the spectral-to-spatial approximation chain.

6. [Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling, 2017)](https://arxiv.org/abs/1609.02907) - arXiv - The original GCN paper; derives the layer-wise propagation rule from a first-order Chebyshev approximation of spectral graph convolutions. Reading Sections 2.1–2.2 alongside this chapter's derivation is the most direct way to understand the renormalization trick.

7. [Neural Message Passing for Quantum Chemistry (Gilmer et al., 2017)](https://arxiv.org/abs/1704.01212) - arXiv - Introduces the Message Passing Neural Network (MPNN) framework that unifies GCN, GG-NN, and interaction networks under a single message–aggregate–update formalism. The notation used in this chapter follows MPNN directly.

8. [PyTorch Geometric MessagePassing Documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html) - PyTorch Geometric Docs - Official tutorial for implementing custom GNN layers using PyG's `MessagePassing` base class. Explains the `propagate`, `message`, `aggregate`, and `update` hooks that map one-to-one onto the MPNN framework covered in this chapter.

9. [Stanford CS224W: Machine Learning with Graphs — Lecture 6: Graph Neural Networks](http://web.stanford.edu/class/cs224w/) - Stanford CS224W - Lecture slides and notes covering GCN, the message passing framework, and spatial vs. spectral convolutions; used in Stanford's flagship graph ML course. Provides alternative derivations and additional intuition that complement this chapter's treatment.

10. [GCN on Papers With Code](https://paperswithcode.com/method/gcn) - Papers With Code - Aggregates benchmark results for GCN across node classification datasets including Cora, Citeseer, and ogbn-arxiv, with links to reproducible implementations. Useful for comparing the baseline numbers cited in this chapter's benchmark table against the current state of the field.
