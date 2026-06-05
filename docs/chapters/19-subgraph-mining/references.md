# References: Frequent Subgraph Mining

1. [Subgraph Isomorphism](https://en.wikipedia.org/wiki/Subgraph_isomorphism_problem) - Wikipedia - Covers the NP-complete decision problem of determining whether one graph contains another as a subgraph, including complexity proofs, canonical algorithms (VF2, Ullmann), and connections to graph homomorphism.

2. [Network Motif](https://en.wikipedia.org/wiki/Network_motif) - Wikipedia - Explains recurring statistically over-represented subgraph patterns in biological and social networks, covering Z-score computation, null model construction, and landmark studies in transcriptional regulation networks.

3. [Graphlet](https://en.wikipedia.org/wiki/Graphlet) - Wikipedia - Describes small connected non-isomorphic induced subgraphs used as features for network comparison and node/edge roles, with coverage of graphlet degree vectors and the graphlet kernel for graph classification.

4. Mining of Massive Datasets - Leskovec, Rajaraman, Ullman - Cambridge University Press - Chapter 10 covers frequent itemset and subgraph mining at scale, providing the algorithmic foundations (Apriori, anti-monotonicity) that underpin systems like gSpan and SPMiner.

5. Graph Theory - Diestel - Springer - The standard rigorous reference for graph isomorphism, subgraph and minor relations, and the Ramsey-type combinatorics that characterize the hardness landscape of subgraph containment problems.

6. [SPMiner: Neural Subgraph Matching (Ying et al., 2020)](https://arxiv.org/abs/2007.10498) - arXiv - Introduces SPMiner, the order-embedding approach that maps graphs into a partially ordered embedding space so that subgraph containment becomes a max-margin geometric query, enabling sub-millisecond frequency estimation on large graphs.

7. [Order Embeddings of Images and Language (Vendrov et al., 2016)](https://arxiv.org/abs/1511.06361) - arXiv - Foundational paper on order-preserving embeddings for partial orders; SPMiner directly adapts this framework to encode the subgraph containment partial order between graph neighborhoods.

8. [Efficient Graphlet Counting for Large Networks (Ahmed et al., 2015)](https://arxiv.org/abs/1506.02745) - arXiv - Presents scalable algorithms for counting 4- and 5-node graphlets in networks with millions of edges, including combinatorial identities that reduce enumeration cost and a comparison to exact versus sampling-based approaches.

9. [PyTorch Geometric — Subgraph Utilities](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html) - PyTorch Geometric Docs - Documents `torch_geometric.utils.subgraph`, `k_hop_subgraph`, and related functions used to extract neighborhood subgraphs for training and evaluating order-embedding models like SPMiner.

10. [Papers With Code — Subgraph Isomorphism](https://paperswithcode.com/task/subgraph-isomorphism) - Papers With Code - Aggregates benchmarks, leaderboards, and linked implementations for subgraph isomorphism and neural subgraph matching, including SPMiner results on the ENZYMES, COX2, and BZR datasets used in the literature.
