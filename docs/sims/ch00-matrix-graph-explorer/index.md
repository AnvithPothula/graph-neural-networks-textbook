---
title: Matrix × Graph Explorer
description: Interactive p5.js MicroSim for matrix × graph explorer.
image: /sims/ch00-matrix-graph-explorer/ch00-matrix-graph-explorer.png
og:image: /sims/ch00-matrix-graph-explorer/ch00-matrix-graph-explorer.png
twitter:image: /sims/ch00-matrix-graph-explorer/ch00-matrix-graph-explorer.png
social:
   cards: false
quality_score: 0
---

# Matrix × Graph Explorer

<iframe src="main.html" height="572" width="100%" scrolling="no"></iframe>

[Run the Matrix × Graph Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

This MicroSim visualizes the matrix multiplication \(A \cdot X\) — the core operation at the heart of every graph convolutional layer. You see a small graph on the right and its adjacency matrix on the left. Click any matrix cell to toggle an edge; the resulting feature aggregation updates instantly.

Three normalization modes show how raw summation (\(A \cdot X\)), degree-normalized averaging (\(D^{-1}A \cdot X\)), and symmetric normalization (\(D^{-1/2}AD^{-1/2} \cdot X\)) treat high-degree and low-degree nodes differently. A color-coded breakdown on each node shows what fraction of the aggregated signal came from each neighbor.

**Learning objective (Bloom's Understand (Level 2)):** Observe how raw, mean, and symmetric normalizations of the adjacency matrix produce different aggregated feature vectors, building intuition for why GCN uses \(D^{-1/2}AD^{-1/2}\) normalization.

## How to Use

1. **Toggle edges** — click a cell in the left adjacency matrix to add or remove an edge between two nodes.
2. **Select a node** — click any node in the graph to highlight which neighbors contribute to its aggregated feature.
3. **Switch normalization** — use the radio buttons to compare Raw (\(A \cdot X\)), Mean (\(D^{-1}A \cdot X\)), and Symmetric (\(D^{-1/2}AD^{-1/2} \cdot X\)) aggregation.
4. **Read the info panel** — the panel on the right shows the numeric feature values before and after aggregation for the selected node.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch00-matrix-graph-explorer/main.html"
        height="572"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
10–15 minutes

### Prerequisites
Basic linear algebra (matrix multiplication, row operations). Familiarity with what a graph is (nodes and edges).

### Activities

1. Add a node with degree 5 and one with degree 1. Switch between Raw and Mean normalization and observe how high-degree nodes dominate under raw but are normalized under mean.
2. Reproduce the GCN aggregation step: switch to Symmetric normalization and verify it matches \(D^{-1/2}AD^{-1/2} \cdot X\).
3. Remove edges until the graph has an isolated node. Observe what each normalization produces for that node.

### Assessment Question
Explain in your own words why symmetric normalization prevents high-degree hubs from dominating feature aggregation. Describe what happens to the diagonal (self-loop) term under each normalization.

## References

1. Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
2. Hamilton (2020). Graph Representation Learning. Synthesis Lectures on AI and ML.

---
*Part of Chapter 0: Math and Programming Prerequisites. Return to the [chapter page](../../chapters/00-math-prerequisites/) or browse all [MicroSims](../index.md).*
