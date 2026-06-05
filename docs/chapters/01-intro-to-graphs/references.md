# References: Introduction to Graphs and Networks

1. [Graph theory](https://en.wikipedia.org/wiki/Graph_theory) - Wikipedia - Comprehensive overview of graph theory covering the formal definitions of graphs, subgraphs, paths, connectivity, and adjacency representations used throughout this chapter. The history section traces the field from Euler's Königsberg bridge problem to modern applications.

2. [Small-world network](https://en.wikipedia.org/wiki/Small-world_network) - Wikipedia - Explains the small-world property—high clustering combined with short average path lengths—and covers the Watts-Strogatz model. Includes empirical examples from social, biological, and technological networks.

3. [Scale-free network](https://en.wikipedia.org/wiki/Scale-free_network) - Wikipedia - Describes networks whose degree distribution follows a power law, discusses the Barabási-Albert preferential attachment model, and surveys real-world examples including the WWW and protein interaction networks.

4. Networks, Crowds, and Markets: Reasoning about a Highly Connected World - David Easley and Jon Kleinberg - Cambridge University Press - An undergraduate-accessible text that builds from basic graph theory to game theory and network effects. Chapters 2–4 give an excellent treatment of connectivity, paths, and structural properties with economic and social motivations. Freely available at the authors' website.

5. Network Science - Albert-László Barabási - Cambridge University Press - A visually rich, interactive textbook covering random graphs, scale-free networks, the Barabási-Albert model, and community structure. Chapters 1–4 map directly to this chapter's content; the online edition at networksciencebook.com includes interactive figures and data sets.

6. [Erdős-Rényi (1960) — On the Evolution of Random Graphs](https://arxiv.org/abs/cond-mat/0205405) - arXiv (Barabási lecture notes covering the model) - The Erdős-Rényi random graph model is the canonical baseline for understanding when giant connected components emerge and what "typical" graph structure looks like. Understanding it makes the non-randomness of real-world networks (clustering, hubs) immediately apparent.

7. [Watts & Strogatz (1998) — Collective dynamics of small-world networks](https://arxiv.org/abs/cond-mat/9811033) - arXiv - The landmark paper introducing the small-world interpolation model between regular lattices and random graphs. Reading it alongside the Karate Club visualization shows concretely how high clustering and low path length coexist in real networks.

8. [PyTorch Geometric — Introduction by Example](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html) - PyTorch Geometric Docs - Shows how graphs are represented as `Data` objects with `edge_index`, `x`, and `y` tensors—the concrete code counterpart to the adjacency matrix formalism introduced in this chapter. The Karate Club example appears directly in the tutorial.

9. [Stanford CS224W: Machine Learning with Graphs — Lecture 1 Slides](http://web.stanford.edu/class/cs224w/) - Stanford University - The opening lecture covers graph structure, why graphs matter for ML, and the taxonomy of graph tasks (node, edge, and graph-level). Freely available slides and notes complement this chapter's conceptual framing.

10. [Barabási & Albert (1999) — Emergence of Scaling in Random Networks](https://arxiv.org/abs/cond-mat/9910332) - arXiv - The original preferential attachment paper demonstrating that growth plus "rich-get-richer" attachment produces power-law degree distributions. Essential reading for understanding why hubs dominate real-world networks and why uniform random graph models fall short.
