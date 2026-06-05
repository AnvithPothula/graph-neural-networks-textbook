# References: Relational Deep Learning

1. [Relational database](https://en.wikipedia.org/wiki/Relational_database) - Wikipedia - Covers the foundational data model underlying relational deep learning: tables, primary keys, foreign keys, and join semantics. Essential background for understanding why the relational-to-graph construction preserves the full information content of a normalized schema.

2. [Graph neural network](https://en.wikipedia.org/wiki/Graph_neural_network) - Wikipedia - Survey-level overview of GNN architectures, message passing, and applications. Provides the graph-learning context needed to understand why GNNs are the natural computational model for relational graphs constructed from database schemas.

3. [Feature engineering](https://en.wikipedia.org/wiki/Feature_engineering) - Wikipedia - Explains the traditional approach to extracting predictive signals from raw data, including from relational databases via aggregation and join-based features. Contrasting this manual pipeline with learned relational representations clarifies the practical motivation for Relational Deep Learning.

4. Deep Learning - Ian Goodfellow, Yoshua Bengio, Aaron Courville - MIT Press - The standard reference for neural network foundations, covering embeddings, optimization, and representation learning. The chapters on structured prediction and sequence models provide the conceptual grounding for understanding how deep models can be extended from flat feature vectors to relational graph structures.

5. Designing Machine Learning Systems - Chip Huyen - O'Reilly Media - Covers practical ML system design including data pipelines, feature stores, and training-serving skew. Particularly relevant to relational deep learning for its treatment of temporal data splits, entity-level label leakage, and the operational complexity of deploying models that depend on live relational database state.

6. [Relational Deep Learning: Graph Representation Learning on Relational Databases (arXiv:2312.04615)](https://arxiv.org/abs/2312.04615) - arXiv - The foundational paper by Fey et al. (2024) establishing the formal relational-to-graph construction, defining the temporal leakage problem, and introducing the RelGNN architecture. The primary technical reference for this chapter.

7. [RelBench: A Benchmark for Deep Learning on Relational Databases (arXiv:2407.20060)](https://arxiv.org/abs/2407.20060) - arXiv - Robinson et al. (2024) document the RelBench datasets, evaluation protocol, and baseline results. Contains the schema diagrams and temporal split definitions needed to reproduce benchmark experiments.

8. [PyTorch Frame Documentation — Relational Deep Learning](https://pytorch-frame.readthedocs.io/en/latest/) - PyTorch Frame / PyG - Official documentation for PyTorch Frame, the tabular deep learning library used for encoding heterogeneous column types in the RDL pipeline. Covers the StatType system, column encoders, and integration with PyTorch Geometric's heterogeneous graph APIs.

9. [Papers With Code — Relational Deep Learning](https://paperswithcode.com/task/relational-deep-learning) - Papers With Code - Tracks state-of-the-art results on RelBench tasks with links to papers and code repositories. Useful for comparing RelGNN variants, tabular baselines, and hybrid approaches as the field evolves beyond the original benchmark publication.

10. [Stanford CS224W: Machine Learning with Graphs — Lecture Notes on Heterogeneous Graphs](https://web.stanford.edu/class/cs224w/) - Stanford University - The publicly available lecture slides and notes from Stanford's graph ML course cover heterogeneous GNNs, relational reasoning, and knowledge graphs. The heterogeneous graph lectures provide complementary treatment of the multi-relational setting that underlies the relational graph construction in RDL.
