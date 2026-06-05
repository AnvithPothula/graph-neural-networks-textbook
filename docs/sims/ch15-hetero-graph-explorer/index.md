---
title: Typed Node and Edge Explorer
description: Interactive p5.js MicroSim for typed node and edge explorer.
image: /sims/ch15-hetero-graph-explorer/ch15-hetero-graph-explorer.png
og:image: /sims/ch15-hetero-graph-explorer/ch15-hetero-graph-explorer.png
twitter:image: /sims/ch15-hetero-graph-explorer/ch15-hetero-graph-explorer.png
social:
   cards: false
quality_score: 0
---

# Typed Node and Edge Explorer

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the Typed Node and Edge Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Heterogeneous graphs encode multiple entity and relation types. An academic graph contains author nodes, paper nodes, and venue nodes, connected by "writes", "cites", and "published_in" edges. Which information a node can access depends critically on which edge types are active.

This MicroSim lets you toggle individual node and edge types on and off and watch the graph restructure. A meta-path panel lets you specify a sequence of edge types (e.g., Author→writes→Paper→cites→Paper) and highlights all nodes reachable through that meta-path from a selected source.

**Learning objective (Bloom's Apply (Level 3)):** Toggle node and edge types on and off and trace meta-path neighborhoods to see how typed structure shapes information flow in a heterogeneous academic graph.

## How to Use

1. **Toggle types** — click the type chips in the legend to show/hide node or edge types.
2. **Select a node** — click any node to see its type-specific neighborhood.
3. **Define a meta-path** — click the "Meta-Path" button and select a sequence of edge types.
4. **Trace the path** — all nodes reachable through the meta-path from the selected source are highlighted.
5. **Key panel** — the legend on the right identifies node and edge types with color codes.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch15-hetero-graph-explorer/main.html"
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
Heterogeneous graph definition. Meta-paths. Basic graph types (Chapter 1).

### Activities

1. Toggle off all edge types except "cites". The graph reduces to a homogeneous citation network. What graph properties change?
2. Trace the Author–writes–Paper–cites–Paper meta-path from one author. How does this capture "papers related to your work"?
3. Define a two-hop meta-path Author–writes–Paper–writes–Author. What does this meta-path compute about two authors?

### Assessment Question
Define a meta-path formally. Explain how HAN (Hierarchical Attention Network) learns attention weights over different meta-paths and why this is more expressive than a single meta-path.

## References

1. Wang et al. (2019). Heterogeneous Graph Attention Network. WWW.
2. Sun & Han (2012). Mining Heterogeneous Information Networks. Synthesis Lectures.

---
*Part of Chapter 15: Heterogeneous Graphs. Return to the [chapter page](../../chapters/15-heterogeneous-graphs/) or browse all [MicroSims](../index.md).*