---
title: GNN Design Space Interactive Comparison
description: Interactive p5.js MicroSim for gnn design space interactive comparison.
image: /sims/ch07-gnn-design-comparison/ch07-gnn-design-comparison.png
og:image: /sims/ch07-gnn-design-comparison/ch07-gnn-design-comparison.png
twitter:image: /sims/ch07-gnn-design-comparison/ch07-gnn-design-comparison.png
social:
   cards: false
quality_score: 0
---

# GNN Design Space Interactive Comparison

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the GNN Design Space Interactive Comparison MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Not every GNN architecture performs best on every dataset. GCN excels on homophilic citation graphs; GraphSAGE generalizes better to unseen nodes; GAT adapts attention to feature-rich settings. This MicroSim presents a dashboard of benchmark results across Cora, Citeseer, ogbn-arxiv, and PPI, letting you explore which model wins under which conditions.

Bar charts compare node classification accuracy, training time, and inductive generalization across architectures. Use filters to isolate specific datasets or metrics and form hypotheses about when to use each model.

**Learning objective (Bloom's Evaluate (Level 5)):** Compare GCN, GraphSAGE, and GAT across datasets and evaluation metrics to develop judgment for selecting an appropriate GNN architecture.

## How to Use

1. **Select a dataset** — choose from the dropdown (Cora, Citeseer, ogbn-arxiv, PPI).
2. **Choose a metric** — toggle between test accuracy, validation loss, and training time.
3. **Read the bars** — each bar represents one architecture. Hover for exact numbers and confidence intervals.
4. **Apply filters** — filter by transductive vs. inductive setting.
5. **Notes panel** — click any bar to read a short interpretation of why that architecture performs the way it does on that dataset.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch07-gnn-design-comparison/main.html"
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
GCN (Chapter 6). GraphSAGE and GAT (Chapter 7). Understanding of inductive vs. transductive learning.

### Activities

1. On Cora, GAT and GCN both perform well. What property of Cora (homophily, node features, small size) might explain this?
2. On PPI (a protein-protein interaction graph), GAT with multi-head attention substantially outperforms GCN. Why might adaptive neighbor weighting be especially valuable in this domain?
3. Identify one scenario (dataset + metric) where GraphSAGE is clearly the best choice and justify why.

### Assessment Question
Compare GCN, GraphSAGE, and GAT along three axes: inductive capability, computational complexity per layer, and sensitivity to neighbor ordering. Produce a 3×3 comparison table.

## References

1. Hu et al. (2020). Open Graph Benchmark: Datasets for Machine Learning on Graphs. NeurIPS.
2. Hamilton et al. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.

---
*Part of Chapter 7: GNN Design Space: GraphSAGE and GAT. Return to the [chapter page](../../chapters/07-gnn-design-space/) or browse all [MicroSims](../index.md).*