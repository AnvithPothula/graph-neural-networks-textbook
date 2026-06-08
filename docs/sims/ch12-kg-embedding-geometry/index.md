---
title: TransE Embedding Geometry
description: Interactive p5.js MicroSim for transe embedding geometry.
image: /sims/ch12-kg-embedding-geometry/ch12-kg-embedding-geometry.png
og:image: /sims/ch12-kg-embedding-geometry/ch12-kg-embedding-geometry.png
twitter:image: /sims/ch12-kg-embedding-geometry/ch12-kg-embedding-geometry.png
social:
   cards: false
quality_score: 0
---

# TransE Embedding Geometry

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the TransE Embedding Geometry MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

TransE models each relation as a translation in vector space: the tail entity \(t\) should be close to head \(h\) displaced by relation vector \(r\). Training minimizes \(\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|\) for positive triples while maximizing the distance for corrupted ones.

This MicroSim animates the 2D embedding space. Entity points and relation arrows update as you step through training. A live loss display shows positive and negative pair distances. A "symmetric relation" preset shows the France–borders–Germany example where the relation vector must equal zero to satisfy both \((\text{France}, \text{borders}, \text{Germany})\) and \((\text{Germany}, \text{borders}, \text{France})\).

**Learning objective (Bloom's Apply (Level 3)):** Watch TransE training drive valid triples toward \(\mathbf{h} + \mathbf{r} \approx \mathbf{t}\) and see why a symmetric relation forces its vector toward zero — exposing TransE's fundamental limitation with symmetry.

## How to Use

1. **Select a preset** — choose "simple transitive" or "symmetric (fails)" to load example triples.
2. **Step training** — click "Train Step" to run one gradient update and watch the embeddings shift.
3. **Read the loss** — the loss panel shows the margin loss for positive and negative triples.
4. **Toggle relation** — click any displayed relation arrow to highlight the triples it is used in.
5. **View failure** — in the symmetric preset, observe that the relation vector collapses toward zero.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch12-kg-embedding-geometry/main.html"
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
Knowledge graph basics — entity, relation, triple (Chapter 12 intro). Vector operations. Distance functions.

### Activities

1. Train on the composition example: "fatherOf" composed with "fatherOf" should approximate "grandfatherOf". Do the learned relation vectors have this property?
2. Load the symmetric preset. After 50 training steps, measure the magnitude of the "borders" relation vector \(\mathbf{r}\). Compare to an asymmetric relation.
3. Explain why ComplEx and RotatE overcome the symmetry limitation using complex-valued embeddings.

### Assessment Question
Define the TransE scoring function \(\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|\). List the four relation patterns (symmetry, antisymmetry, inversion, composition) and state which TransE can and cannot model, with justification.

## References

1. Bordes et al. (2013). Translating Embeddings for Modeling Multi-relational Data. NeurIPS.
2. Sun et al. (2019). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. ICLR.

---
*Part of Chapter 12: Knowledge Graph Embeddings. Return to the [chapter page](../../chapters/12-knowledge-graph-embeddings/index.md) or browse all [MicroSims](../index.md).*