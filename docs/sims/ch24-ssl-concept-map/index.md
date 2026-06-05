---
title: DGI vs. Contrastive Learning — Concept Map
description: Interactive p5.js MicroSim for dgi vs. contrastive learning — concept map.
image: /sims/ch24-ssl-concept-map/ch24-ssl-concept-map.png
og:image: /sims/ch24-ssl-concept-map/ch24-ssl-concept-map.png
twitter:image: /sims/ch24-ssl-concept-map/ch24-ssl-concept-map.png
social:
   cards: false
quality_score: 0
---

# DGI vs. Contrastive Learning — Concept Map

<iframe src="main.html" height="482" width="100%" scrolling="no"></iframe>

[Run the DGI vs. Contrastive Learning — Concept Map MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Deep Graph Infomax (DGI) and graph contrastive learning (GCL) are both self-supervised graph learning methods, but they approach the objective differently. DGI maximizes mutual information between node embeddings and a global graph summary; GCL maximizes agreement between augmented views.

This concept map arranges the three SSL properties (mutual information, alignment, uniformity) as central nodes and connects each method to the properties it satisfies. Click any node or edge to read an explanation of the connection.

**Learning objective (Bloom's Analyze (Level 4)):** Compare DGI and graph contrastive learning by exploring how each method satisfies the three core SSL properties — mutual information maximization, uniformity, and alignment — via an interactive concept map.

## How to Use

1. **Click a method node** — DGI or GCL — to read a summary of the approach.
2. **Click a property node** — mutual information, alignment, or uniformity — to read its definition.
3. **Click an edge** — to read how the connected method satisfies (or partially satisfies) that property.
4. **Toggle connections** — hide/show DGI or GCL connections to compare coverage.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch24-ssl-concept-map/main.html"
        height="482"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
20–30 minutes

### Prerequisites
Graph contrastive learning basics. DGI conceptual overview (Chapter 24). Mutual information at an intuitive level.

### Activities

1. DGI does not directly enforce uniformity. Which node in the concept map represents this gap? What downstream effect can this have on embedding quality?
2. GCL enforces both alignment and uniformity but requires many negative pairs. Click the "NT-Xent loss" edge to see why large batch sizes are needed.
3. Add a third method node (e.g., BGRL, a bootstrapping SSL method that uses no negatives) and sketch which properties it would connect to.

### Assessment Question
Compare DGI and a two-view GCL method along the three SSL properties. For each (method, property) pair, state whether the property is directly enforced, indirectly satisfied, or not addressed.

## References

1. Veličković et al. (2019). Deep Graph Infomax. ICLR.
2. You et al. (2020). Graph Contrastive Learning with Augmentations. NeurIPS.

---
*Part of Chapter 24: Advanced GNN Topics: In-Context Learning and Uncertainty. Return to the [chapter page](../../chapters/24-advanced-gnn-topics/) or browse all [MicroSims](../index.md).*