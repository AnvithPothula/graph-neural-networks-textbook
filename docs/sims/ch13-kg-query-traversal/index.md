---
title: Query2Box Multi-Hop Traversal
description: Interactive p5.js MicroSim for query2box multi-hop traversal.
image: /sims/ch13-kg-query-traversal/ch13-kg-query-traversal.png
og:image: /sims/ch13-kg-query-traversal/ch13-kg-query-traversal.png
twitter:image: /sims/ch13-kg-query-traversal/ch13-kg-query-traversal.png
social:
   cards: false
quality_score: 0
---

# Query2Box Multi-Hop Traversal

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the Query2Box Multi-Hop Traversal MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Query2Box represents a query's answer set as an axis-aligned hyperrectangle (box) in embedding space. Entities inside the box are candidate answers. Projection operators shift and resize the box along each relation step; intersection operators narrow it to the region satisfying all conditions.

This MicroSim shows a 2D KG embedding space. A query panel on the left lets you compose \(1p\) (single-hop), \(2p\) (two-hop chain), and \(2i\) (two-chain intersection) queries. As you step through the query, boxes animate in the embedding space and entities inside the final box are highlighted as answers.

**Learning objective (Bloom's Analyze (Level 4)):** Watch Query2Box project and intersect axis-aligned boxes to resolve \(1p\), \(2p\), and \(2i\) queries over a toy knowledge graph. Each projection shifts and widens the box; intersection narrows to the overlap.

## How to Use

1. **Choose a query type** — \(1p\), \(2p\), or \(2i\) from the dropdown.
2. **Select a relation** — choose the relation for each hop (e.g., "bornIn", "locatedIn").
3. **Step through** — click "Next" to apply each projection or intersection operation.
4. **Read the answer** — entities inside the final box are the candidate answers.
5. **Switch anchors** — change the anchor entity (\(r_1\) or \(r_2\)) to see how the boxes shift.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch13-kg-query-traversal/main.html"
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
Knowledge graph triples and completion (Chapter 12). Set intersection. Embedding space basics.

### Activities

1. Compose a \(2p\) query: (Einstein) → bornIn → ? → locatedIn → ?. Step through and identify which entities fall in the final box.
2. Construct a \(2i\) query. Before applying intersection, observe how large the two individual query boxes are. After intersection, how much did the answer set shrink?
3. Explain why boxes are a better representation than points for multi-hop queries with uncertain or multiple answers.

### Assessment Question
Derive the projection operator p(q, r) for Query2Box: given a box q = (center, offset), write the update rule for applying relation r. Why must the offset grow monotonically during projection?

## References

1. Ren et al. (2020). Query2box: Reasoning over Knowledge Graphs in Vector Space Using Box Embeddings. ICLR.
2. Hamilton et al. (2018). Embedding Methods for Link Prediction. Knowledge Graph Handbook.

---
*Part of Chapter 13: Reasoning over Knowledge Graphs. Return to the [chapter page](../../chapters/13-kg-reasoning/index.md) or browse all [MicroSims](../index.md).*