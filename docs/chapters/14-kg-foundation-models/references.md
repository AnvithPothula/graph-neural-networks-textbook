# References: Knowledge Graph Foundation Models

1. [Knowledge Graph](https://en.wikipedia.org/wiki/Knowledge_graph) - Wikipedia - Overview of knowledge graphs as structured representations of facts, covering history from Google's Knowledge Graph (2012) through current large-scale deployments; provides the definitional grounding for what "knowledge graph foundation models" are generalizing over.

2. [Link Prediction](https://en.wikipedia.org/wiki/Link_prediction) - Wikipedia - Describes the link prediction task in networks, including transductive versus inductive settings; directly relevant to understanding why zero-shot link prediction across unseen knowledge graphs is a meaningful capability.

3. [Inductive Transfer Learning](https://en.wikipedia.org/wiki/Transfer_learning) - Wikipedia - Explains transfer learning paradigms, including the distinction between transductive and inductive transfer; grounds the conceptual leap from graph-specific trained models to universal pre-trained models like ULTRA.

4. Hamilton, W. L. — *Graph Representation Learning* — Morgan & Claypool, 2020 — The standard graduate-level textbook on graph ML; Chapter 7 covers knowledge graph embeddings and sets up the transductive framing that inductive and universal models (ULTRA, InGram) are designed to overcome.

5. Hogan, A., Blomqvist, E., Cochez, M., et al. — *Knowledge Graphs* — Synthesis Lectures on Data, Semantics, and Knowledge, Morgan & Claypool, 2021 — Comprehensive treatment of knowledge graph construction, representation, and reasoning; Chapter 6 on embeddings and Chapter 7 on rule mining provide context for why purely embedding-based approaches have scalability and generalization limits.

6. [ULTRA: Towards Foundation Models for Knowledge Graph Reasoning](https://arxiv.org/abs/2310.04562) - arXiv - Galkin et al. (2023) introduce ULTRA, showing that a single pre-trained model with double equivariance achieves competitive zero-shot link prediction across 43 knowledge graphs; the core reference for this chapter's universal graph model section.

7. [NBFNet: Neural Bellman-Ford Networks for Link Prediction](https://arxiv.org/abs/2106.06935) - arXiv - Zhu et al. (2021) present NBFNet, the path-based GNN backbone used inside ULTRA; understanding NBFNet's generalized Bellman-Ford message passing is essential for understanding how ULTRA achieves relation-conditioned entity scoring.

8. [InGram: Inductive Knowledge Graph Embedding via Relation Graphs](https://arxiv.org/abs/2305.02994) - arXiv - Lee et al. (2023) propose building a relation-level graph whose edges encode co-occurrence and inverse patterns, enabling embedding induction for both unseen entities and unseen relations; the key reference for the relation graph induction section.

9. [ULTRA on Papers With Code — Knowledge Graph Completion Benchmarks](https://paperswithcode.com/paper/ultra-towards-foundation-models-for) - Papers With Code - Aggregates reported results for ULTRA on standard inductive KG benchmarks (FB15k-237, WN18RR, NELL-995 inductive splits); useful for locating updated leaderboard comparisons and reproduced numbers from the community.

10. [PyTorch Geometric: Knowledge Graph Datasets and Link Prediction Utilities](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) - PyTorch Geometric Docs - Documents the KG dataset classes (`FB15k237`, `WordNet18RR`, `OGB LinkPropPred`) and link prediction loaders used to run ULTRA, NBFNet, and InGram experiments; the practical reference for reproducing this chapter's code examples.
