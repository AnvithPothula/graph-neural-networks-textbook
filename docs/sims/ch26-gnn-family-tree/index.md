---
title: GNN Architecture Family Tree
description: Interactive p5.js MicroSim for gnn architecture family tree.
image: /sims/ch26-gnn-family-tree/ch26-gnn-family-tree.png
og:image: /sims/ch26-gnn-family-tree/ch26-gnn-family-tree.png
twitter:image: /sims/ch26-gnn-family-tree/ch26-gnn-family-tree.png
social:
   cards: false
quality_score: 0
---

# GNN Architecture Family Tree

<iframe src="main.html" height="602" width="100%" scrolling="no"></iframe>

[Run the GNN Architecture Family Tree MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

This interactive family tree maps 30+ GNN architectures onto a design-space taxonomy. Four primary axes organize the tree: aggregation (sum vs. mean vs. max vs. attention), scope (local message-passing vs. global transformer), task (node/edge/graph-level), and training regime (supervised vs. self-supervised vs. inductive).

Nodes are architectures; directed edges are "extends" or "inspired by" relationships. Click any architecture node to see its key design decisions and the chapter where it is covered in this textbook.

**Learning objective (Bloom's Analyze (Level 4)):** Locate any major GNN architecture within the design-space taxonomy, trace its lineage to foundational predecessors, and find the textbook chapter that covers it.

## How to Use

1. **Click an architecture** — a card shows the key design decisions (aggregation, normalization, positional encoding, etc.) and chapter link.
2. **Trace lineage** — follow edges backward to foundational models (GCN, GraphSAGE) or forward to advanced variants.
3. **Filter by axis** — use the axis filters (aggregation, task, regime) to highlight a subset of architectures.
4. **Search** — type a model name in the search box to jump to it.
5. **Chapter links** — each node card has a link to the relevant chapter.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch26-gnn-family-tree/main.html"
        height="602"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
20–30 minutes

### Prerequisites
All chapters 6–25. This MicroSim is best used after completing the full textbook as a synthesis tool.

### Activities

1. Trace the lineage from GCN (Chapter 6) to Graphormer (Chapter 11). List every "extends" edge on the path.
2. Find three architectures that share the same aggregation function (sum + MLP) but differ in scope or task.
3. Place one novel architecture you have read about (outside this textbook) on the family tree. Which existing nodes would have edges to it?

### Assessment Question
Describe the GNN design space along four dimensions: aggregation function, scope, positional encoding, and training regime. For each dimension, list two contrasting choices and give one architecture example for each.

## References

1. You et al. (2020). Design Space for Graph Neural Networks. NeurIPS.
2. Bronstein et al. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.

---
*Part of Chapter 26: Conclusion: The GNN Design Space and Open Problems. Return to the [chapter page](../../chapters/26-conclusion/index.md) or browse all [MicroSims](../index.md).*