# References: GNNs for Recommender Systems

1. [Recommender System](https://en.wikipedia.org/wiki/Recommender_system) - Wikipedia - Broad overview of recommender system paradigms including collaborative filtering, content-based filtering, and hybrid approaches; provides essential context for understanding why graph-based methods improve on classical techniques.

2. [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) - Wikipedia - Explains the foundational assumption that users who agreed in the past will agree in the future, covering both memory-based and model-based variants that motivate the user-item bipartite graph formulation.

3. [Matrix Factorization (recommender systems)](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) - Wikipedia - Describes how latent factor models decompose the user-item interaction matrix into low-dimensional embeddings; the precursor to neural and graph-based extensions such as NGCF and LightGCN.

4. Recommender Systems Handbook - Francesco Ricci, Lior Rokach, Bracha Shapira - Springer - Comprehensive reference covering collaborative filtering, content-based methods, and evaluation protocols; provides the algorithmic grounding that graph-based recommendation builds upon.

5. Mining of Massive Datasets - Jure Leskovec, Anand Rajaraman, Jeff Ullman - Cambridge University Press - Chapter on recommendation systems covers SVD-based matrix factorization and the UV-decomposition framework in depth; directly connects classical techniques to the graph embedding perspective adopted by LightGCN.

6. [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (arXiv:2002.02126)](https://arxiv.org/abs/2002.02126) - arXiv - The original LightGCN paper showing that removing feature transformation and nonlinear activation from NGCF while retaining neighborhood aggregation over the user-item bipartite graph substantially improves recommendation accuracy.

7. [Neural Graph Collaborative Filtering (arXiv:1905.08108)](https://arxiv.org/abs/1905.08108) - arXiv - Introduces NGCF, which explicitly encodes collaborative signal by propagating embeddings along the user-item interaction graph; the direct predecessor to LightGCN and a key step in applying GNNs to recommendation.

8. [Graph Convolutional Neural Networks for Web-Scale Recommender Systems (arXiv:1806.01973)](https://arxiv.org/abs/1806.01973) - arXiv - Presents PinSage, Pinterest's industrial-scale GCN-based recommender that uses random-walk-based neighborhood sampling and importance pooling to handle graphs with billions of nodes and edges.

9. [LightGCN — Papers With Code](https://paperswithcode.com/paper/lightgcn-simplifying-and-powering-graph) - Papers With Code - Aggregates benchmark results for LightGCN across standard recommendation datasets (Gowalla, Yelp2018, Amazon-Book) alongside code implementations and comparisons to competing methods.

10. [PyTorch Geometric — LightGCN Example](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.LightGCN.html) - PyTorch Geometric Docs - Official API reference and usage example for the `LightGCN` model class in PyG, covering the constructor arguments, forward pass, predict\_link method, and recommendation helper utilities.
