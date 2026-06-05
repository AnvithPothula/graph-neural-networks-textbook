# References: Temporal and Dynamic Graphs

1. [Temporal network](https://en.wikipedia.org/wiki/Temporal_network) - Wikipedia - Covers the formal definition of temporal networks, including time-varying edges, snapshot representations, and key metrics like temporal degree and reachability. Provides grounding for the snapshot-vs.-event-stream distinction central to Chapter 22.

2. [Dynamic network analysis](https://en.wikipedia.org/wiki/Dynamic_network_analysis) - Wikipedia - Surveys methods for analyzing networks that change over time, bridging social network analysis and computational approaches. Contextualizes why static GNN methods fall short for evolving graphs.

3. [Graph neural network](https://en.wikipedia.org/wiki/Graph_neural_network) - Wikipedia - Broad overview of GNN architectures including sections on spectral and spatial methods; useful background for understanding how temporal extensions like TGN and TGAT build on static GNN foundations.

4. Graph Representation Learning - William L. Hamilton - Morgan & Claypool (Synthesis Lectures on AI and Machine Learning), 2020 - Chapter 6 addresses dynamic and temporal graphs, discussing snapshot-based methods and the challenges of capturing long-range temporal dependencies; rigorous mathematical treatment of message passing that underpins TGAT and TGN.

5. Deep Learning on Graphs - Yao Ma and Jiliang Tang - Cambridge University Press, 2021 - Dedicated chapter on dynamic graph neural networks covering both discrete-time and continuous-time formulations; connects temporal GNNs to broader topics of graph classification, link prediction, and scalability.

6. [Temporal Graph Networks for Deep Learning on Dynamic Graphs (Rossi et al., 2020)](https://arxiv.org/abs/2006.10637) - arXiv - The original TGN paper introducing the modular memory module, message aggregator, and time encoder; essential reading for understanding per-node memory and the encoder–decoder framework for temporal link prediction.

7. [Inductive Representation Learning on Temporal Graphs (Xu et al., 2020)](https://arxiv.org/abs/2002.07962) - arXiv - Introduces TGAT, the attention-based temporal graph network that uses time2vec encodings and multi-head attention over temporal neighborhoods; establishes the inductive setting where new nodes are handled without retraining.

8. [Learning to Simulate Complex Physics with Graph Networks (Sanchez-Gonzalez et al., 2020)](https://arxiv.org/abs/2002.09405) - arXiv - Demonstrates continuous-time graph networks for physical simulation with dynamic edge connectivity; illustrates how temporal message passing generalizes beyond social/citation networks to physics-based dynamic graphs.

9. [PyTorch Geometric Temporal Documentation](https://pytorch-geometric-temporal.readthedocs.io/en/latest/) - PyTorch Geometric Temporal (ReadTheDocs) - Official library documentation covering snapshot-based (DCRNN, EvolveGCN) and event-based (TGAT, TGN) temporal GNN implementations; includes runnable examples and benchmark comparisons directly relevant to Chapter 22 code exercises.

10. [Inductive Representation Learning on Large Graphs (Hamilton et al., 2017)](https://arxiv.org/abs/1706.02216) - arXiv - Introduces GraphSAGE, the static inductive framework that TGAT and TGN extend to the temporal domain; understanding neighborhood sampling and aggregation in GraphSAGE is prerequisite to following TGN's message-passing design.
