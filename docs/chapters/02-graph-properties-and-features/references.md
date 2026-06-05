# References: Graph Properties and Traditional ML Features

1. [Clustering Coefficient](https://en.wikipedia.org/wiki/Clustering_coefficient) - Wikipedia - Covers both local and global clustering coefficients, including the Watts–Strogatz formulation. Explains why triangles in a neighborhood indicate small-world structure, with examples from social and biological networks.

2. [Betweenness Centrality](https://en.wikipedia.org/wiki/Betweenness_centrality) - Wikipedia - Defines betweenness centrality as the fraction of shortest paths passing through a node, with discussion of Brandes's O(VE) algorithm. Includes comparisons with closeness and eigenvector centrality and real network examples.

3. [Weisfeiler Leman Graph Isomorphism Test](https://en.wikipedia.org/wiki/Weisfeiler_Leman_graph_isomorphism_test) - Wikipedia - Describes the 1-WL color-refinement algorithm, its iterative neighborhood aggregation procedure, and its failure cases on regular graphs. Directly relevant to understanding the expressive power of GNNs in later chapters.

4. Graph Theory - Reinhard Diestel - Springer - The standard graduate reference for graph-theoretic foundations. Chapters 1–3 cover degree sequences, paths, connectivity, and trees with full proofs; Chapter 8 covers graph minors and structural results. Free author PDF available for the fifth edition.

5. Networks: An Introduction - Mark Newman - Oxford University Press - Comprehensive treatment of network science at the intersection of physics, CS, and sociology. Chapters 7–9 cover degree distributions, clustering, centrality measures, and motifs with empirical data from real-world networks, making it ideal background reading for Chapter 2.

6. [Graphlet Kernel (Shervashidze et al., 2009)](https://arxiv.org/abs/0910.5912) - arXiv - Introduces graphlet kernels for graph classification, defining feature vectors from counts of size-3 through size-5 graphlets. Demonstrates that graphlet-based representations outperform random walk kernels on bioinformatics benchmarks.

7. [Weisfeiler-Lehman Graph Kernels (Shervashidze et al., 2011)](https://jmlr.org/papers/v12/shervashidze11a.html) - JMLR - The canonical paper formalizing the WL subtree kernel, WL shortest-path kernel, and WL edge kernel. Proves that each WL iteration computes the same refinement as one step of the 1-WL isomorphism test, connecting graph kernels to the expressivity theory covered in later chapters.

8. [Node Classification with Graph Neural Networks — PyTorch Geometric Tutorial](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html) - PyTorch Geometric Docs - Official Colab tutorials walking through node feature construction, data loading with `torch_geometric.data.Data`, and baseline graph ML pipelines. Covers the exact API used for feature engineering throughout this textbook.

9. [Stanford CS224W Lecture 2: Traditional Methods for ML on Graphs](http://web.stanford.edu/class/cs224w/slides/02-tradition-ml.pdf) - Stanford CS224W - Slide deck covering node-level features (degree, centrality, clustering coefficient), link-level features (common neighbors, Jaccard, Adamic-Adar, Katz index), and graph-level features (graphlet count vectors, WL kernel). Concise reference for the full feature taxonomy in this chapter.

10. [Papers With Code: Graph Classification](https://paperswithcode.com/task/graph-classification) - Papers With Code - Leaderboard tracking state-of-the-art results on graph classification benchmarks including MUTAG, PROTEINS, and IMDB-B. Useful for comparing traditional kernel methods against GNN baselines and for finding reproducible code for the models introduced in this chapter.
