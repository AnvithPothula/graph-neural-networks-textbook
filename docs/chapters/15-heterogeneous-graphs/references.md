# References: Heterogeneous Graphs

1. [Heterogeneous network](https://en.wikipedia.org/wiki/Heterogeneous_network) - Wikipedia - Covers the general concept of networks with multiple node and link types, including the formal definition of heterogeneous information networks and their properties; provides foundational vocabulary for understanding typed graphs in machine learning contexts.

2. [Knowledge graph](https://en.wikipedia.org/wiki/Knowledge_graph) - Wikipedia - Describes knowledge graphs as directed labeled graphs where nodes represent entities of diverse types and edges represent typed relations; directly relevant because knowledge graphs are canonical examples of heterogeneous graphs and motivate relational GNN architectures such as R-GCN.

3. [Graph neural network](https://en.wikipedia.org/wiki/Graph_neural_network) - Wikipedia - Surveys the full GNN landscape including message-passing frameworks, spectral vs. spatial methods, and extensions to heterogeneous and relational graphs; useful as a reference map for situating heterogeneous GNN models within the broader field.

4. *Graph Representation Learning* - William L. Hamilton - Morgan & Claypool, 2020 - Chapter 4 covers multi-relational and heterogeneous message passing, including the derivation of R-GCN, basis decomposition, and connections to knowledge graph embedding methods; the tightest single-source treatment of the material in this chapter.

5. *Deep Learning on Graphs* - Yao Ma and Jiliang Tang - Cambridge University Press, 2021 - Dedicated chapter on heterogeneous graph neural networks covers HAN, HGT, and meta-path construction with worked examples on academic citation graphs; complements Hamilton's treatment with more architectural detail and application case studies.

6. [Modeling Relational Data with Graph Convolutional Networks (R-GCN)](https://arxiv.org/abs/1703.06103) - arXiv - Schlichtkrull et al. (2018) introduce relation-specific weight matrices with basis and block-diagonal decompositions to keep parameter counts tractable; the foundational paper for all heterogeneous GNN work that handles typed edges without meta-paths.

7. [Heterogeneous Graph Transformer (HGT)](https://arxiv.org/abs/2003.01332) - arXiv - Hu et al. (2020) propose meta-relation-specific attention using separate key, query, and value projections for each pair of node types, demonstrated on the large-scale OGB-MAG academic graph; the standard reference for transformer-style heterogeneous GNNs.

8. [Heterogeneous Graph Attention Network (HAN)](https://arxiv.org/abs/1903.07293) - arXiv - Wang et al. (2019) introduce a two-level attention architecture: node-level attention aggregates neighbors along a single meta-path, and semantic-level attention weights the relative importance of different meta-paths for the downstream task.

9. [Heterogeneous Graph Learning — PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/heterogeneous_graphlearning.html) - PyTorch Geometric Docs - Official guide to PyG's `HeteroData` format, `to_hetero()` conversion utility, and built-in implementations of R-GCN, HGT, and HAN; includes runnable code examples that map directly to the implementations discussed in this chapter.

10. [Papers With Code: Heterogeneous Graph Benchmark](https://paperswithcode.com/task/heterogeneous-graph-classification) - Papers With Code - Aggregates leaderboard results for heterogeneous graph tasks including node classification on OGB-MAG and HGB benchmarks, with links to reproducible code for R-GCN, HAN, HGT, and more recent methods; useful for tracking state-of-the-art performance and finding open-source implementations.
