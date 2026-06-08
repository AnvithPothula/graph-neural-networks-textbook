---
title: Graph Contrastive Learning — Two-View Pipeline
description: Interactive p5.js MicroSim for graph contrastive learning — two-view pipeline.
image: /sims/ch24-graph-contrastive-learning/ch24-graph-contrastive-learning.png
og:image: /sims/ch24-graph-contrastive-learning/ch24-graph-contrastive-learning.png
twitter:image: /sims/ch24-graph-contrastive-learning/ch24-graph-contrastive-learning.png
social:
   cards: false
quality_score: 0
---

# Graph Contrastive Learning — Two-View Pipeline

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the Graph Contrastive Learning — Two-View Pipeline MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Graph contrastive learning (GCL) creates two augmented views of the same graph (e.g., by randomly dropping edges and features) and trains a GNN encoder to produce similar embeddings for the two views of the same node while pushing apart embeddings of different nodes.

This MicroSim shows the two-view pipeline in action. A small graph appears on the left; two augmented views are derived by randomly masking edges and node features. The GNN encodes each view, and the contrastive loss between the resulting embedding pairs is shown. Increasing augmentation strength makes the views more different and raises the NT-Xent loss.

**Learning objective (Bloom's Understand (Level 2)):** See how two augmented views of a graph produce two node embeddings each, and how the NT-Xent loss pulls matching views together while pushing apart all other pairs.

## How to Use

1. **Generate views** — click "Augment" to create two stochastic views of the graph.
2. **Adjust augmentation** — the "edge drop %" and "feature mask %" sliders control augmentation strength.
3. **Run GNN** — click "Encode" to compute embeddings for both views.
4. **Read the loss** — the contrastive loss (NT-Xent) is shown, decomposed into positive pair similarity and negative pair similarity.
5. **Temperature slider** — adjust τ to see how sharper temperatures affect the gradient signal.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch24-graph-contrastive-learning/main.html"
        height="522"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
10–15 minutes

### Prerequisites
GNN basics (Chapter 6). Cosine similarity (Chapter 0). Self-supervised learning concept.

### Activities

1. Set edge drop to 0% (no augmentation). The two views are identical. What is the NT-Xent loss? What problem does this create for training?
2. Increase edge drop to 50%. How does the positive-pair similarity change? Does the loss increase or decrease?
3. For the same node, compare its embedding from view 1 and view 2. Are they close? What does "close" mean in cosine similarity?

### Assessment Question
Explain what representation collapse is in contrastive learning and how it arises. List two architectural or training techniques that prevent collapse.

## References

1. You et al. (2020). Graph Contrastive Learning with Augmentations. NeurIPS.
2. Zhu et al. (2021). Graph Contrastive Learning with Adaptive Augmentation. WWW.

---
*Part of Chapter 24: Advanced GNN Topics: In-Context Learning and Uncertainty. Return to the [chapter page](../../chapters/24-advanced-gnn-topics/index.md) or browse all [MicroSims](../index.md).*