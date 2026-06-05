# References: Link Analysis and PageRank

1. [PageRank](https://en.wikipedia.org/wiki/PageRank) - Wikipedia - Covers the history, mathematics, and variants of Google's PageRank algorithm, including the random surfer model, teleportation, and personalized PageRank. A reliable entry point for understanding the core ideas before diving into primary literature.

2. [HITS algorithm](https://en.wikipedia.org/wiki/HITS_algorithm) - Wikipedia - Explains Kleinberg's Hyperlink-Induced Topic Search algorithm, including the mutual reinforcement between hub and authority scores and the connection to singular value decomposition. Useful for comparing HITS with PageRank in terms of both computation and design intent.

3. [Random walk](https://en.wikipedia.org/wiki/Random_walk) - Wikipedia - Provides the probabilistic foundations underlying PageRank, including Markov chain theory, stationary distributions, and convergence conditions. Understanding random walks as a general framework clarifies why the teleportation trick guarantees a unique stationary distribution.

4. *Google's PageRank and Beyond: The Science of Search Engine Rankings* - Amy N. Langville and Carl D. Meyer - Princeton University Press - The most mathematically thorough textbook treatment of PageRank, covering linear algebra foundations, power iteration convergence, personalization, TrustRank, and spam resistance. The standard reference for readers who want proofs, not just intuitions.

5. *Mining of Massive Datasets* - Jure Leskovec, Anand Rajaraman, and Jeffrey D. Ullman - Cambridge University Press - Chapter 5 gives a self-contained treatment of link analysis including PageRank, HITS, and topic-sensitive PageRank, with a focus on scalable algorithms for web-scale graphs. Freely available online and used in multiple top-ranked university courses.

6. [The Anatomy of a Large-Scale Hypertextual Web Search Engine (Brin & Page, 1998)](https://arxiv.org/abs/cs/9811017) - arXiv - The original PageRank paper describing the algorithm's derivation, the teleportation fix for dangling nodes and spider traps, and early experiments on the Stanford web crawl. Reading the primary source is worth the time; the paper is accessible and historically illuminating.

7. [Predict then Propagate: Graph Neural Networks Meet Personalized PageRank (Klicpera et al., 2019)](https://arxiv.org/abs/1810.05997) - arXiv - Introduces APPNP, which replaces standard GNN propagation with the fixed-point iteration of personalized PageRank, achieving better node classification while mitigating over-smoothing. The clearest example of how PageRank theory directly informs modern GNN design.

8. [CS224W: Machine Learning with Graphs — Lecture 4: PageRank](http://web.stanford.edu/class/cs224w/) - Stanford University - Stanford's graduate course on graph machine learning covers PageRank, random walks, and HITS in depth, with slides and recorded lectures freely available. Lecture 4 in particular bridges the theory in this chapter to its role as a precursor to graph neural networks.

9. [NetworkX: PageRank Documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html) - NetworkX Docs - Official API reference for NetworkX's `pagerank` function, including parameters for damping factor, personalization vector, and convergence tolerance. The implementation follows the power iteration approach covered in this chapter and is the easiest way to run PageRank experiments on small graphs.

10. [Graph Convolutional Neural Networks for Web-Scale Recommender Systems (Ying et al., 2018)](https://arxiv.org/abs/1806.01973) - arXiv - Describes Pinterest's PinSage system, which uses PPR-based neighborhood sampling to train GNNs on a graph with three billion nodes and two billion edges. A production-scale validation that personalized PageRank remains the workhorse for scalable graph learning far beyond the web search context where it was invented.
