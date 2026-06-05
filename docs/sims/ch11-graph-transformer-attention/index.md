---
title: Graph Transformer Attention Heatmap
description: Interactive p5.js MicroSim for graph transformer attention heatmap.
image: /sims/ch11-graph-transformer-attention/ch11-graph-transformer-attention.png
og:image: /sims/ch11-graph-transformer-attention/ch11-graph-transformer-attention.png
twitter:image: /sims/ch11-graph-transformer-attention/ch11-graph-transformer-attention.png
social:
   cards: false
quality_score: 0
---

# Graph Transformer Attention Heatmap

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the Graph Transformer Attention Heatmap MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Standard GNNs restrict attention to immediate neighbors. Graph transformers like GPS (General Powerful Scalable GNN) pair a local message-passing module with a global attention module, letting every node directly attend to every other node in a single layer.

This MicroSim shows a molecular graph with an attention heatmap overlay. Click a query node and see, for both MPNN-local and transformer-global attention, which atoms receive the highest attention weights. As GPS layers increase, global attention reaches farther across the molecule.

**Learning objective (Bloom's Analyze (Level 4)):** Compare strictly-local MPNN attention with global multi-head self-attention on a molecular graph, and watch how GPS layers let attention spread from local to long-range.

## How to Use

1. **Select a query node** — click any atom in the molecular graph.
2. **Toggle mode** — switch between "MPNN (local)" and "Transformer (global)" attention.
3. **Layer depth slider** — increase GPS layers (1–4) to see how global attention coverage expands.
4. **Heatmap** — edge color and width encode attention weight from the query to each other atom.
5. **Head selector** — view each of the 4 attention heads separately.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch11-graph-transformer-attention/main.html"
        height="522"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
20–30 minutes

### Prerequisites
GCN and GAT (Chapters 6–7). Self-attention in transformers (basic familiarity). Over-squashing (Chapter 8).

### Activities

1. On the molecular graph, pick a terminal atom (degree 1). Under MPNN with 2 layers, which atoms can it see? Under global attention, how many can it attend to directly?
2. At GPS depth 1, compare the local (MPNN) and global (attention) attention patterns for the same query node. Which atoms differ most?
3. Explain how Graphormer encodes shortest-path distance as a bias in the attention score.

### Assessment Question
Describe the GPS architecture: what are the local and global modules, and how are their outputs combined? Explain why this hybrid outperforms pure MPNN or pure transformer on graph tasks.

## References

1. Rampášek et al. (2022). Recipe for a General, Powerful, Scalable Graph Transformer. NeurIPS.
2. Ying et al. (2021). Do Transformers Really Perform Badly for Graph Representation? (Graphormer). NeurIPS.

---
*Part of Chapter 11: Graph Transformers. Return to the [chapter page](../../chapters/11-graph-transformers/) or browse all [MicroSims](../index.md).*