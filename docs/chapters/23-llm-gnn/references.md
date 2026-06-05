# References: LLMs and GNNs: Text-Attributed Graphs

1. [BERT (language model)](https://en.wikipedia.org/wiki/BERT_(language_model)) - Wikipedia - Covers the architecture and pre-training objectives of BERT, the transformer-based encoder that underpins most LLM-as-encoder approaches for text-attributed graphs. Explains masked language modeling and next-sentence prediction that give BERT its rich node-feature representations.

2. [Graph neural network](https://en.wikipedia.org/wiki/Graph_neural_network) - Wikipedia - Authoritative overview of the GNN family — message passing, aggregation, and readout — providing the structural backbone that LLM-encoded features are fed into. The article traces the line from spectral methods to spatial GNNs such as GraphSAGE and GIN that appear throughout Chapter 23.

3. [Transformer (deep learning architecture)](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) - Wikipedia - Explains self-attention and positional encoding, the mechanisms that make large language models capable of producing contextualized node embeddings from raw text. Understanding transformers is prerequisite to understanding why LLM encoders outperform bag-of-words features on text-attributed benchmarks.

4. *Graph Representation Learning* - William L. Hamilton - Morgan & Claypool (Synthesis Lectures on AI and Machine Learning), 2020 - Foundational graduate-level treatment of node embeddings, GNN expressiveness, and downstream tasks. Chapter 3 on GNN architectures and Chapter 7 on graph classification set the GNN side of the LLM+GNN equation in rigorous mathematical form.

5. *Natural Language Processing with Transformers* - Lewis Tunstall, Leandro von Werra, Thomas Wolf - O'Reilly Media, 2022 - Practical and theoretical coverage of the Hugging Face transformer ecosystem, covering fine-tuning BERT-family models, tokenization, and embedding extraction. Directly applicable to the LLM-as-encoder pipeline described in TAPE and similar text-attributed graph methods.

6. [TAPE: Text-Attributed Graph Embedding via LLM Explanations](https://arxiv.org/abs/2310.09872) - arXiv - He et al. (2023) introduce TAPE, which uses an LLM to generate natural-language explanations and predictions for each node, then distills those into a GNN. This paper is the canonical reference for the LLM-as-predictor integration strategy and demonstrates strong results on ogbn-arxiv.

7. [One for All: Towards Training One Graph Model for All Classification Tasks](https://arxiv.org/abs/2310.00149) - arXiv - Zhao et al. (2023) present OFA, unifying node, link, and graph classification across heterogeneous datasets via a single prompted GNN trained on text-attributed graphs. The paper introduces the "nodes of interest" prompting paradigm and is the primary reference for cross-dataset graph foundation models in this chapter.

8. [Explanations as Features: LLM-Based Features for Text-Attributed Graphs](https://arxiv.org/abs/2305.19523) - arXiv - Chen et al. (2023) study how LLM-generated text features compare with raw text features across multiple GNN architectures. The systematic ablation clarifies when freezing the LLM encoder is sufficient versus when joint fine-tuning is needed, making it essential reading for practitioners building text-attributed graph pipelines.

9. [PyTorch Geometric — Text-Attributed Graph Datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) - PyTorch Geometric Docs - Official documentation for TAGDataset and related loaders in PyG, covering ogbn-arxiv, Amazon co-purchase, and other text-attributed benchmarks used throughout Chapter 23. Includes API examples for attaching raw text features to node data objects before passing them through an LLM encoder.

10. [Papers With Code — Text-Attributed Graph Benchmark (ogbn-arxiv)](https://paperswithcode.com/sota/node-classification-on-ogbn-arxiv) - Papers With Code - Live leaderboard tracking state-of-the-art node classification accuracy on ogbn-arxiv, the primary benchmark for LLM+GNN methods. Useful for comparing TAPE, OFA, and joint fine-tuning approaches against concurrent and post-publication baselines with reproducible code links.
