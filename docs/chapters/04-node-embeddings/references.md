# References: Node Embeddings: DeepWalk and node2vec

1. [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) - Wikipedia - Covers the Word2Vec family of models (skip-gram and CBOW) introduced by Mikolov et al., including the negative sampling and hierarchical softmax training objectives that DeepWalk directly inherits for node embedding.

2. [Random Walk](https://en.wikipedia.org/wiki/Random_walk) - Wikipedia - Defines random walks on graphs, discusses their stationary distributions, and covers the relationship between walk length, graph structure, and the neighborhoods sampled—foundational to understanding why DeepWalk's corpus captures structural proximity.

3. [Matrix Factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) - Wikipedia - Explains low-rank matrix factorization in the context of representation learning; directly relevant to the Qiu et al. (2018) result showing that DeepWalk and node2vec implicitly factorize specific graph Laplacian-derived matrices.

4. Graph Representation Learning - William L. Hamilton - Morgan & Claypool (Synthesis Lectures on AI and Machine Learning) - The most rigorous textbook treatment of the encoder-decoder framework for node embeddings; Chapter 3 formalizes shallow methods algebraically and motivates the transition to GNNs. A free PDF is available from the author's website.

5. Networks, Crowds, and Markets: Reasoning About a Highly Connected World - David Easley & Jon Kleinberg - Cambridge University Press - Provides deep intuition on graph structure, community detection, and information diffusion that contextualizes why homophily-preserving versus structural-equivalence-preserving embeddings behave differently; Chapter 3 on strong and weak ties is directly relevant.

6. [DeepWalk: Online Learning of Social Representations (arXiv:1403.6652)](https://arxiv.org/abs/1403.6652) - arXiv - The original DeepWalk paper by Perozzi, Al-Rfou, and Skiena (KDD 2014); introduces the random-walk corpus construction, skip-gram with hierarchical softmax for nodes, and the power-law co-occurrence argument that justifies adapting Word2Vec to graphs.

7. [node2vec: Scalable Feature Learning for Networks (arXiv:1607.00653)](https://arxiv.org/abs/1607.00653) - arXiv - The node2vec paper by Grover and Leskovec (KDD 2016); defines the biased second-order random walk controlled by parameters p and q, and systematically shows how interpolating between BFS and DFS captures homophily versus structural equivalence.

8. [Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec (arXiv:1710.02971)](https://arxiv.org/abs/1710.02971) - arXiv - Qiu et al. (WSDM 2018) prove that all major shallow embedding methods implicitly factorize closed-form graph matrices; essential reading for understanding why these methods differ in what structural information they capture.

9. [node2vec — PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.Node2Vec.html) - PyTorch Geometric Docs - Official API reference for PyG's Node2Vec implementation, covering the walk sampling parameters, the skip-gram training loop, and the embedding evaluation utilities used in the code examples throughout this chapter.

10. [Papers With Code: Node Classification Benchmark](https://paperswithcode.com/task/node-classification) - Papers With Code - Aggregates state-of-the-art results and linked code for node classification across standard benchmarks including ogbn-arxiv; useful for comparing how shallow embedding baselines (DeepWalk, node2vec) stack up against modern GNN methods on the same datasets used in later chapters.
