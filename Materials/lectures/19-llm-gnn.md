# Lecture 19: LLMs + GNNs, Agents + Graphs, and Conclusion

## Overview

The frontier of graph ML: combining large language models with GNNs for text-attributed graphs, and using graphs to structure agent workflows. The lecture concludes with an overview of the GNN design space and open problems.

**Slides (2025, LLM+GNN):** http://web.stanford.edu/class/cs224w/slides/Lecture16.pdf
**Slides (2025, Agents+Graphs):** http://web.stanford.edu/class/cs224w/slides/2025-cs224w-lecture.pdf
**Slides (2025, Conclusion):** http://web.stanford.edu/class/cs224w/slides/19-conclusion.pdf
**KG Foundation Models:** http://web.stanford.edu/class/cs224w/slides/15-KGFoundationModels.pdf

---

## Key Concepts

### LLMs + GNNs: Text-Attributed Graphs
- Many real-world graphs have rich text on nodes/edges (academic papers, products, code)
- **Approach 1: LLM as feature extractor** — use BERT/GPT embeddings as node features for GNN
- **Approach 2: LLM for graph reasoning** — prompt LLMs to reason over graph-structured inputs
- **Approach 3: Joint LLM-GNN** — co-train LLM and GNN end-to-end

### Graph Instruction Tuning
- Fine-tune LLMs with graph-related instruction-following tasks
- Tasks: node classification, link prediction, subgraph description
- **InstructGLM:** LLM instruction tuning for graph learning
- **GraphGPT:** aligns graph encoder with LLM via graph instruction tuning

### LLM as Graph Reasoner
- Zero-shot: describe graph structure in text, ask LLM to classify/reason
- Chain-of-thought: ask LLM to "walk" through the graph step by step
- **Challenge:** LLMs struggle with structural reasoning beyond ~few hops; context length limits

### Graph Foundation Models
- **Goal:** a single pre-trained model that transfers to any graph task
- **Challenges:**
  - Different graphs have different node/edge feature spaces
  - Different tasks require different heads
- **PRODIGY:** in-context learning with prompt graphs
- **OFA (One for All):** align all graph modalities into shared text space via LLM

### KG Foundation Models (from Lecture 15)

**Inductive KG Reasoning:**
- Classic KG embeddings (TransE, RotatE) are *transductive* — can't handle new entities
- **Inductive setting:** test entities/relations not seen during training

**ULTRA (Universal Transferable Reasoning on Graphs):**
- Foundation model for KG reasoning
- Key insight: use *relation graph* (graph of relations, not entities) — allows transfer across KGs
- Pre-trained on multiple KGs; zero-shot generalizes to new KGs
- Achieves SOTA on 57 different KG benchmarks without fine-tuning

**NBFNet for Inductive Reasoning:**
- Neural Bellman-Ford: path reasoning that generalizes to new entities
- Uses structural features (relative positions) rather than entity IDs

### Agents + Graphs
- Graphs structure multi-agent workflows (task dependencies, tool chains)
- Knowledge graphs provide structured world knowledge for agents
- Tool graphs: agents as nodes, tool-call sequences as paths
- Memory as graphs: agent memory represented as entity-relation graphs

### Conclusion: The GNN Design Space

Five key dimensions:
1. **Message/aggregation:** mean, sum, max, attention
2. **Update function:** MLP depth, residual connections
3. **Layer depth:** 2-3 optimal for most tasks
4. **Pre/post-processing:** initial MLP, final MLP
5. **Training:** batch size, dropout, learning rate

**Open problems:**
- Scaling laws for GNNs (do they scale like LLMs?)
- Long-range dependencies in graphs
- GNNs for combinatorial optimization
- Foundation models for molecules and proteins
- Causality and counterfactuals on graphs
- Privacy in graph ML (re-identification from embeddings)

## Key Papers
- [ULTRA (Galkin et al., 2023)](https://arxiv.org/abs/2310.04562)
- [PRODIGY (Huang et al., 2023)](https://arxiv.org/abs/2305.12600)
- [InGram: Inductive KG Embedding (Lee et al., 2023)](https://arxiv.org/abs/2305.19987)
- [Double Equivariance for Inductive KG (Guo et al., 2023)](https://arxiv.org/abs/2302.01313)
- [Relational Transformer (2025)](https://arxiv.org/abs/2510.06377)
- [Design Space for GNNs (You et al., 2020)](https://arxiv.org/abs/2011.08843)

## Textbook Chapter Notes

Maps to **Chapter 19: LLMs, Agents, and the Future of Graph ML**.
- Surveys emerging directions without going deep into any single method
- Key open research questions as discussion prompts
- Conclusion table: what each GNN component adds and when to use it
