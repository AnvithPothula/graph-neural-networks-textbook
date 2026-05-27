# Lecture 18: Deep Generative Models for Graphs

## Overview

Can we generate new, valid graphs? This is critical for drug discovery (generate new molecules), materials science (generate crystal structures), and network simulation. Graph generation is fundamentally different from image/text generation due to discrete, variable-size, and permutation-invariant structure.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/18-deep-generation.pdf
**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/15-deep-generation.pdf

---

## Key Concepts

### Challenges of Graph Generation
- **Discrete structure:** nodes and edges are discrete (unlike pixel values)
- **Variable size:** graphs have different numbers of nodes
- **Permutation invariance:** same graph can have many adjacency matrix representations
- **Validity constraints:** molecular graphs must satisfy valence rules, etc.

### Problem Formulation
Given a set of observed graphs G = {G_1, ..., G_n}, learn p_θ(G) such that:
1. Generated samples G ~ p_θ(G) look like real graphs
2. Novel graphs can be sampled
3. (Conditional generation) generate graphs satisfying desired properties

### GraphRNN (You et al., 2018)
- **Key idea:** generate a graph as a sequence of nodes + edges
- **Two-level RNN:**
  1. Graph-level RNN: generates one node at a time, tracks graph state
  2. Edge-level RNN: for each new node, decides which previous nodes to connect to
- BFS node ordering reduces edge sequences to tractable lengths
- Trained with teacher forcing (MLE); sampled with autoregressive decoding
- **Limitation:** large memory footprint; doesn't capture global structure well

### GCPN (Graph Convolutional Policy Network, You et al., 2018)
- **Goal-directed molecule generation** (optimize molecular properties)
- **Framework:** RL + GNN + chemistry rules
  - **State:** current partial molecular graph
  - **Action:** add a new atom/bond or terminate
  - **Policy:** GCN-based policy network
  - **Reward:** drug-likeness score + validity + property optimization
- Generates molecules with desired properties (drug-likeness, solubility, etc.)

### Variational Graph Autoencoder (VGAE, Kipf & Welling 2016)
- **Encoder:** GCN → mean and variance of latent code
- **Decoder:** inner product of latent codes → edge probability
- Trained with ELBO: reconstruction + KL divergence
- Good for small/medium graphs; limited scalability

### Diffusion-Based Graph Generation (Recent)
- Apply score-based / diffusion models to graphs
- **DiGress (2022):** discrete diffusion over node/edge types
  - Forward process: gradually add noise to graph (add random edges, change node types)
  - Reverse process: learn to denoise (recover original graph)
  - Equivariant transformer backbone
- **GDSS:** uses SDEs for joint node/edge generation
- State-of-the-art on molecule generation benchmarks

### Evaluation Metrics
- **Validity:** fraction of generated graphs that obey domain rules (e.g., molecular valence)
- **Uniqueness:** fraction of unique generated graphs
- **Novelty:** fraction not in training set
- **FCD (Fréchet ChemNet Distance):** measures distribution distance in chemical property space
- **Graph statistics:** degree distribution, clustering coefficient, orbit counts

## Key Papers
- [GraphRNN (You et al., 2018)](https://arxiv.org/pdf/1802.08773.pdf)
- [GCPN (You et al., 2018)](https://arxiv.org/pdf/1806.02473.pdf)
- [VGAE (Kipf & Welling, 2016)](https://arxiv.org/abs/1611.07308)
- [DiGress (Vignac et al., 2022)](https://arxiv.org/abs/2209.14734)
- [GDSS (Jo et al., 2022)](https://arxiv.org/abs/2209.14734)

## Textbook Chapter Notes

Maps to **Chapter 18: Deep Generative Models for Graphs**.
- Case study: drug discovery pipeline with GCPN
- MicroSim: visualize GraphRNN generating a small graph node by node
