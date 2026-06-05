---
title: GAT Attention Weight Visualizer
description: Interactive p5.js MicroSim for gat attention weight visualizer.
image: /sims/ch07-gat-attention-weights/ch07-gat-attention-weights.png
og:image: /sims/ch07-gat-attention-weights/ch07-gat-attention-weights.png
twitter:image: /sims/ch07-gat-attention-weights/ch07-gat-attention-weights.png
social:
   cards: false
quality_score: 0
---

# GAT Attention Weight Visualizer

<iframe src="main.html" height="502" width="100%" scrolling="no"></iframe>

[Run the GAT Attention Weight Visualizer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Graph Attention Networks (GAT) replace fixed normalized aggregation with learned, data-dependent attention coefficients. Each head independently learns a scoring function that weighs neighbors by relevance, then softmax normalizes the scores. Multiple heads run in parallel, each potentially attending to different aspects.

This MicroSim pre-loads synthetic attention weights for a small graph. Click any node to see its four attention heads: thick, dark edges carry high weight; thin, pale edges carry little. A bar chart below shows the softmax distribution for each head.

**Learning objective (Bloom's Apply (Level 3)):** Click a node and see attention coefficients over its neighbors as edge width and color, across four independent attention heads, connecting the softmax definition to visual intuition.

## How to Use

1. **Select a node** — click any node to see its neighbor attention weights highlighted as edge widths.
2. **Switch heads** — use the head selector (1–4) to view each attention head independently.
3. **Read the bar chart** — the attention weight distribution for the selected node appears as a bar chart below.
4. **Hover an edge** — see the exact softmax weight for that (node, head) pair.
5. **Compare all heads** — click "All Heads" to overlay all four heads simultaneously as colored edges.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch07-gat-attention-weights/main.html"
        height="502"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
15–20 minutes

### Prerequisites
Softmax function (Chapter 0). Message passing framework and GCN (Chapter 6). Basic neural network layers.

### Activities

1. Select the hub node. Do all four heads agree on which neighbor to attend to most? Or does each head specialize?
2. Find a node whose attention weights are nearly uniform across neighbors (each weight \(\approx 1/\text{degree}\)). Compare to GCN's normalized aggregation — what is the difference?
3. Write the GAT attention scoring formula \(e(i,j) = a(\mathbf{W} \cdot \mathbf{h}_i \| \mathbf{W} \cdot \mathbf{h}_j)\) and explain how softmax is applied.

### Assessment Question
Why does GAT use multiple heads? What is the theoretical advantage of attending to neighbors differently depending on features, compared to GCN's fixed \(\frac{1}{\sqrt{d_i \cdot d_j}}\) weights?

## References

1. Veličković et al. (2018). Graph Attention Networks. ICLR.
2. Brody et al. (2022). How Attentive are Graph Attention Networks? ICLR.

---
*Part of Chapter 7: GNN Design Space: GraphSAGE and GAT. Return to the [chapter page](../../chapters/07-gnn-design-space/) or browse all [MicroSims](../index.md).*
