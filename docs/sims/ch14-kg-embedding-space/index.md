---
title: Structural KG Embedding Transfer (ULTRA-style)
description: Interactive p5.js MicroSim for structural kg embedding transfer (ultra-style).
image: /sims/ch14-kg-embedding-space/ch14-kg-embedding-space.png
og:image: /sims/ch14-kg-embedding-space/ch14-kg-embedding-space.png
twitter:image: /sims/ch14-kg-embedding-space/ch14-kg-embedding-space.png
social:
   cards: false
quality_score: 0
---

# Structural KG Embedding Transfer (ULTRA-style)

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the Structural KG Embedding Transfer (ULTRA-style) MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Foundation models like ULTRA learn KG representations that transfer across graphs by encoding structural role rather than entity identity. Two entities with the same relational pattern in different KGs receive the same embedding regardless of their names or the specific graph.

This MicroSim shows two small KGs side by side. Click "Shuffle IDs" to relabel entities — the embeddings stay the same, demonstrating that the representation is identity-independent. Click "Perturb Structure" to change connectivity — now embeddings change, showing that structure is what drives the representation.

**Learning objective (Bloom's Analyze (Level 4)):** See how structurally equivalent entities in two different knowledge graphs receive the same embedding, because the representation encodes relational role rather than identity.

## How to Use

1. **View the two KGs** — two knowledge graphs are displayed side by side, each with their entity embeddings shown as 2D scatter plots.
2. **Shuffle IDs** — relabels all entities randomly. Observe that embeddings are unchanged.
3. **Perturb structure** — adds or removes a relation edge in one KG. Observe which embeddings shift.
4. **Identify structural equivalents** — click any entity to highlight its counterpart in the other KG.
5. **Read similarity scores** — the matrix panel shows pairwise cosine similarity between entities across KGs.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch14-kg-embedding-space/main.html"
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
Knowledge graph embeddings (Chapter 12). Inductive reasoning. Graph isomorphism.

### Activities

1. Shuffle IDs five times. Confirm that the pairwise similarity matrix is unchanged each time.
2. Add an edge between two entities. Which entities' embeddings change? Only the directly affected ones, or also their neighbors?
3. Find a pair of entities (one per KG) with cosine similarity \(> 0.9\). Verify they have the same structural role (same relation type distribution).

### Assessment Question
Explain the key design principle that makes ULTRA transferable: why must relation representations be computed relative to each other rather than as absolute parameters?

## References

1. Galkin et al. (2023). Towards Foundation Models for Knowledge Graph Reasoning. ICLR 2024.
2. Zhu et al. (2021). Neural Bellman-Ford Networks. NeurIPS.

---
*Part of Chapter 14: Knowledge Graph Foundation Models. Return to the [chapter page](../../chapters/14-kg-foundation-models/) or browse all [MicroSims](../index.md).*