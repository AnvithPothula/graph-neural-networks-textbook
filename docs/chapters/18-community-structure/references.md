# References: Community Structure in Networks

1. [Community Structure in Networks](https://en.wikipedia.org/wiki/Community_structure) - Wikipedia - Covers the definition of community structure, the resolution limit problem, benchmark graphs, and an overview of major detection algorithms including modularity-based and spectral methods.

2. [Modularity (Networks)](https://en.wikipedia.org/wiki/Modularity_(networks)) - Wikipedia - Explains the modularity quality function Q, its derivation from the null model (configuration model), and the known resolution limit that prevents detection of small communities in large graphs.

3. [Louvain Method](https://en.wikipedia.org/wiki/Louvain_method) - Wikipedia - Describes the two-phase greedy modularity optimization algorithm, its O(n log n) scalability, and the refinements introduced in the Leiden algorithm to fix the disconnected-community problem.

4. Networks: An Introduction - Mark Newman - Oxford University Press - The definitive reference for network science; Chapter 11 covers community detection from first principles, including spectral methods, the Girvan-Newman edge-betweenness algorithm, and modularity maximization with rigorous derivations.

5. Mining of Massive Datasets - Jure Leskovec, Anand Rajaraman, Jeff Ullman - Cambridge University Press - Chapter 10 treats community detection at scale; covers the BigCLAM overlapping-community model, the conductance-based community scoring function, and practical graph-mining considerations for networks with hundreds of millions of edges.

6. [Modularity and Community Structure in Networks (Girvan & Newman, 2004)](https://arxiv.org/abs/cond-mat/0408187) - arXiv - The foundational paper introducing the modularity function Q and demonstrating that greedy modularity maximization recovers ground-truth communities in benchmark and real-world networks.

7. [Fast Unfolding of Communities in Large Networks (Blondel et al., 2008)](https://arxiv.org/abs/0803.0476) - arXiv - Introduces the Louvain algorithm; proves two-phase greedy optimization achieves near-optimal modularity in O(n log n) time and demonstrates results on networks with up to 100 million nodes.

8. [Overlapping Community Detection at Scale: BigCLAM (Yang & Leskovec, 2013)](https://arxiv.org/abs/1303.3399) - arXiv - Introduces the BigCLAM model, which assigns each node a real-valued membership vector per community; derives the log-likelihood objective and shows the model outperforms hard-partition methods on Amazon, YouTube, and DBLP.

9. [Community Detection Algorithms — PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html) - PyTorch Geometric Docs - Documents graph clustering utilities and community-related transforms available in PyG, including DMoN pooling and spectral clustering helpers that integrate directly into GNN training pipelines.

10. [Papers With Code — Community Detection](https://paperswithcode.com/task/community-detection) - Papers With Code - Aggregates state-of-the-art results, leaderboards, and linked code repositories for community detection benchmarks including SBM synthetic graphs, Amazon co-purchase, and the OGB node-clustering suite.
