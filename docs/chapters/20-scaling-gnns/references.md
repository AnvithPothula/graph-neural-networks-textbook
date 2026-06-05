# References: Scaling GNNs to Billion-Node Graphs

1. [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) - Wikipedia - Covers the mini-batch optimization principle that underlies neighbor sampling and subgraph-based GNN training; explains variance reduction techniques central to scalable learning.

2. [Graph partitioning](https://en.wikipedia.org/wiki/Graph_partition) - Wikipedia - Describes the algorithmic problem of dividing a graph into balanced, low-conductance subgraphs, which is the core primitive used by Cluster-GCN and METIS-based batching strategies.

3. [Importance sampling](https://en.wikipedia.org/wiki/Importance_sampling) - Wikipedia - Explains the statistical foundation for constructing unbiased mini-batch estimators, directly applicable to GraphSAINT's node and edge samplers and their normalization corrections.

4. Graph Representation Learning - William L. Hamilton - Morgan & Claypool - Dedicated textbook treatment of scalable GNN training, covering GraphSAGE neighbor sampling and the mathematical analysis of mini-batch bias and variance in GNN optimization.

5. Deep Learning on Graphs - Yao Ma, Jiliang Tang - Cambridge University Press - Comprehensive coverage of scalability challenges in graph deep learning, including cluster-based batching, stochastic depth methods, and the neighborhood explosion problem in multi-layer GNNs.

6. [Inductive Representation Learning on Large Graphs (GraphSAGE)](https://arxiv.org/abs/1706.02216) - arXiv - Hamilton et al. (2017); the foundational paper introducing neighbor sampling for inductive GNN training, enabling scaling to graphs with millions of nodes by sampling a fixed-size neighborhood at each layer.

7. [Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks](https://arxiv.org/abs/1905.07953) - arXiv - Chiang et al. (2019); introduces graph clustering as the batching primitive, using METIS to partition graphs into dense subgraphs that minimize cross-cluster edges and reduce variance during mini-batch GCN training.

8. [GraphSAINT: Graph Sampling Based Inductive Learning Method](https://arxiv.org/abs/1907.04931) - arXiv - Zeng et al. (2020); proposes sampling full subgraphs (nodes, edges, or random walks) and using importance normalization to produce unbiased gradient estimators, achieving state-of-the-art results on large benchmarks.

9. [Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced Training (SIGN)](https://arxiv.org/abs/2004.11198) - arXiv - Rossi et al. (2020); precomputes multi-hop diffusion operators offline, reducing GNN training to standard mini-batch MLP optimization with no neighborhood sampling required at train time.

10. [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://arxiv.org/abs/2005.00687) - arXiv - Hu et al. (2020); introduces ogbn-arxiv, ogbn-products, and ogbn-papers100M — the standard billion-scale benchmarks used to evaluate scalable GNN methods, with leaderboards tracking state-of-the-art performance.
