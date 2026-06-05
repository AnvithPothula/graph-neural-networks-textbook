---
title: User-Item Graph with Multi-Hop Propagation
description: Interactive p5.js MicroSim for user-item graph with multi-hop propagation.
image: /sims/ch16-lightgcn-explorer/ch16-lightgcn-explorer.png
og:image: /sims/ch16-lightgcn-explorer/ch16-lightgcn-explorer.png
twitter:image: /sims/ch16-lightgcn-explorer/ch16-lightgcn-explorer.png
social:
   cards: false
quality_score: 0
---

# User-Item Graph with Multi-Hop Propagation

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the User-Item Graph with Multi-Hop Propagation MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

LightGCN strips GCN down to its core: pure neighbor aggregation on a bipartite user-item graph, without nonlinearity or feature transformation. Each layer expands the receptive field by one hop — after two layers a user's representation incorporates items' users' items.

This MicroSim shows a small user-item bipartite graph. Click a user node to see its 1-hop, 2-hop, and 3-hop collaborative-filtering neighborhoods. A recommendation panel lists the top-5 items ranked by the dot-product score between the user embedding and item embeddings.

**Learning objective (Bloom's Apply (Level 3)):** Explore how multi-hop paths through a user-item bipartite graph generate collaborative-filtering signals, and see how more LightGCN layers expand the receptive field of recommendation.

## How to Use

1. **Select a user** — click any user node (circular, left side).
2. **Set LightGCN layers** — the slider (1–3) controls how many aggregation layers are applied.
3. **Read the neighborhood** — nodes highlighted in each layer color show the expanding receptive field.
4. **View recommendations** — the right panel shows top-5 item recommendations for the selected user, ranked by embedding similarity.
5. **Add interaction** — click any user-item pair to add or remove a rating edge, then observe how recommendations change.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch16-lightgcn-explorer/main.html"
        height="522"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
15–20 minutes

### Prerequisites
Bipartite graphs. Collaborative filtering basics. GCN aggregation (Chapter 6).

### Activities

1. Select a user with few interactions (low degree). Increase layers from 1 to 3. Does the recommendation list improve (diversify)?
2. Add an interaction between user A and item X. Identify which other users' recommendations are affected and explain the propagation path.
3. Compare LightGCN (no weights, no nonlinearity) to full GCN on this task. What does LightGCN gain by removing these components?

### Assessment Question
Write the LightGCN aggregation rule: \(e_u^{(k)} = \sum_{i \in \mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}} \cdot e_i^{(k-1)}\). Explain how the final embedding is a weighted sum across layers and why layer weighting helps.

## References

1. He et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR.
2. Wang et al. (2019). Neural Graph Collaborative Filtering. SIGIR.

---
*Part of Chapter 16: GNNs for Recommender Systems. Return to the [chapter page](../../chapters/16-recommender-systems/) or browse all [MicroSims](../index.md).*