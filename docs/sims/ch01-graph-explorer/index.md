---
title: Graph Property Explorer
description: Interactive p5.js MicroSim for graph property explorer.
image: /sims/ch01-graph-explorer/ch01-graph-explorer.png
og:image: /sims/ch01-graph-explorer/ch01-graph-explorer.png
twitter:image: /sims/ch01-graph-explorer/ch01-graph-explorer.png
social:
   cards: false
quality_score: 0
---

# Graph Property Explorer

<iframe src="main.html" height="572" width="100%" scrolling="no"></iframe>

[Run the Graph Property Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

This MicroSim is a live graph editor that computes six structural properties as you build. Add nodes by clicking empty space, connect them by clicking two nodes in sequence, and delete elements by right-clicking. Every change instantly updates the property panel on the right.

The histogram on the right shows the evolving degree distribution as you add and remove edges, making it easy to see the difference between a random graph, a star, and a path at a glance.

**Learning objective (Bloom's Apply (Level 3)):** Construct graphs interactively and observe how global properties — degree distribution, average path length, clustering coefficient, connected components, and bipartiteness — respond to structural edits in real time.

## How to Use

1. **Add a node** — click on any empty area of the canvas.
2. **Add or remove an edge** — click one node, then click another; the edge toggles.
3. **Delete a node or edge** — right-click any node to delete it (and its edges).
4. **Move nodes** — drag any node to reposition it.
5. **Read properties** — the right panel updates live with degree distribution, average path length, clustering coefficient, number of components, and bipartiteness.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch01-graph-explorer/main.html"
        height="572"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
15–20 minutes

### Prerequisites
No prerequisites — this is an introductory simulation. Familiarity with basic graph vocabulary (node, edge, degree) is helpful.

### Activities

1. Build a path graph (nodes in a line). Record the average path length and clustering coefficient. Then add the shortcut edges of a small-world network and observe what changes.
2. Build a star graph (one hub connected to many leaves). Observe the degree distribution. Compare to a ring graph of the same size.
3. Try to build a bipartite graph (two groups with edges only between groups). Notice when the bipartite indicator turns off.

### Assessment Question
State the formula for the local clustering coefficient and verify it manually for one node in your graph. Explain why a path graph has clustering coefficient zero.

## References

1. Zachary (1977). An information flow model for conflict and fission in small groups. Journal of Anthropological Research.
2. Newman (2010). Networks: An Introduction. Oxford University Press.

---
*Part of Chapter 1: Introduction to Graphs and Networks. Return to the [chapter page](../../chapters/01-intro-to-graphs/) or browse all [MicroSims](../index.md).*