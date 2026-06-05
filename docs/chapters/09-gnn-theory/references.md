# References: Theory of GNNs: Expressiveness and the WL Test

1. [Weisfeiler–Leman graph isomorphism test](https://en.wikipedia.org/wiki/Weisfeiler%E2%80%93Leman_graph_isomorphism_test) - Wikipedia - Covers the color-refinement algorithm underlying the 1-WL test, its iterative label propagation procedure, and the known classes of graphs it fails to distinguish, including regular graphs of the same degree.

2. [Graph isomorphism](https://en.wikipedia.org/wiki/Graph_isomorphism) - Wikipedia - Explains the mathematical definition of graph isomorphism, describes the graph isomorphism problem (GI), and surveys its complexity-theoretic status as a problem believed to lie strictly between P and NP-complete.

3. [Graph isomorphism problem](https://en.wikipedia.org/wiki/Graph_isomorphism_problem) - Wikipedia - Focuses specifically on the computational complexity of deciding isomorphism, reviewing Babai's quasipolynomial-time algorithm and the canonical forms used in practice, providing context for why WL serves as a practical heuristic.

4. Graph Neural Networks: A Review of Methods and Applications - Jie Zhou et al. - AI Open, Elsevier - Comprehensive survey bridging theoretical expressiveness analysis with architectural practice; the chapters on message passing unification and expressiveness limitations are directly relevant to understanding the 1-WL ceiling.

5. Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges - Michael Bronstein, Joan Bruna, Taco Cohen, Petar Velickovic - arXiv monograph / Cambridge University Press - Develops the symmetry-based theoretical foundation of GNNs; Chapter 4 on graphs rigorously derives why permutation equivariance implies WL-bounded expressiveness, situating the result within the broader blueprint of geometric learning.

6. [How Powerful are Graph Neural Networks? (arXiv:1810.00826)](https://arxiv.org/abs/1810.00826) - arXiv - The landmark Xu et al. (2019) ICLR paper proving that any MPNN is at most as powerful as the 1-WL test and that Graph Isomorphism Network (GIN) achieves this upper bound; the primary theoretical reference for Chapter 9.

7. [Equivariant Subgraph Aggregation Networks (arXiv:2110.02910)](https://arxiv.org/abs/2110.02910) - arXiv - Introduces ESAN, which breaks the 1-WL barrier by applying GNNs to bags of subgraphs; explains both the expressiveness gap of standard MPNNs and how subgraph-based approaches surpass it with well-controlled complexity overhead.

8. [CS224W: Machine Learning with Graphs — Lecture 6: Graph Neural Networks: Expressiveness](https://web.stanford.edu/class/cs224w/) - Stanford University - The Stanford graph ML course covers GNN expressiveness, the WL hierarchy, and GIN in dedicated lecture slides and notes; a clear pedagogical walk-through complementing the formal proofs.

9. [Graph Isomorphism Network (GIN) — Papers With Code](https://paperswithcode.com/method/gin) - Papers With Code - Aggregates benchmark results, code repositories, and derivative works for GIN; useful for tracking how WL-aligned architectures perform on standard graph classification suites such as MUTAG, COLLAB, and IMDB.

10. [PyTorch Geometric: GINConv documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html) - PyTorch Geometric Docs - Official API reference for the GIN convolution layer, with parameter descriptions and a minimal usage example; directly supports the runnable code examples in this chapter.
