---
title: SPMiner Order Embedding Space
description: Interactive p5.js MicroSim for spminer order embedding space.
image: /sims/ch19-spminer-embedding/ch19-spminer-embedding.png
og:image: /sims/ch19-spminer-embedding/ch19-spminer-embedding.png
twitter:image: /sims/ch19-spminer-embedding/ch19-spminer-embedding.png
social:
   cards: false
quality_score: 0
---

# SPMiner Order Embedding Space

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the SPMiner Order Embedding Space MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

SPMiner embeds both queries (small subgraph patterns) and graph neighborhoods into the same space using an order embedding: neighborhood \(G\) contains query \(Q\) if and only if \(\text{emb}(Q) \leq \text{emb}(G)\) coordinate-wise (each dimension of \(\text{emb}(G) \geq \text{emb}(Q)\)). This turns subgraph search into a dominance query in embedding space.

This MicroSim shows a 2D projection of the order embedding space. Query patterns are plotted as points. The shaded region above and to the right of a query point represents all neighborhoods that contain it. Use the controls to explore how frequency estimation is performed by counting dominated points.

**Learning objective (Bloom's Understand (Level 2)):** See how order embeddings encode the subgraph partial order geometrically: a neighborhood contains the query if and only if its embedding coordinate-wise dominates the query.

## How to Use

1. **Plot a query** — click "Add Query" and draw or select a small subgraph pattern.
2. **Read the embedding** — the query's 2D embedding point appears; its "domination region" is shaded.
3. **Count contained neighborhoods** — the count of neighborhood embeddings in the domination region updates.
4. **Move the query** — drag the query point to see how frequency estimates change.
5. **Overlay multiple queries** — add multiple query patterns and see how their regions overlap.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch19-spminer-embedding/main.html"
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
Subgraph isomorphism. Order embeddings and partial orders. Vector operations.

### Activities

1. Place a triangle query and a single-edge query. Which has a larger domination region? Why?
2. Move the triangle query point closer to the origin. Does the frequency estimate increase or decrease? Explain.
3. Explain in your own words why coordinate-wise dominance captures the subgraph containment partial order.

### Assessment Question
Define order embeddings formally and prove that they satisfy the containment property: \(G_1 \subseteq G_2\) implies \(\text{emb}(G_1) \leq \text{emb}(G_2)\) coordinate-wise. What loss function does SPMiner use to learn these embeddings?

## References

1. Rex Ying et al. (2020). Frequent Subgraph Mining by Walking in Order Embedding Space. NeurIPS.
2. Vendrov et al. (2016). Order-Embeddings of Images and Language. ICLR.

---
*Part of Chapter 19: Frequent Subgraph Mining. Return to the [chapter page](../../chapters/19-subgraph-mining/index.md) or browse all [MicroSims](../index.md).*
