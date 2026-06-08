---
title: GCN Message Passing Visualizer
description: Interactive p5.js MicroSim for gcn message passing visualizer.
image: /sims/ch06-gcn-message-passing/ch06-gcn-message-passing.png
og:image: /sims/ch06-gcn-message-passing/ch06-gcn-message-passing.png
twitter:image: /sims/ch06-gcn-message-passing/ch06-gcn-message-passing.png
social:
   cards: false
quality_score: 0
---

# GCN Message Passing Visualizer

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the GCN Message Passing Visualizer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

This MicroSim animates mean aggregation (no learned weights, as in the conceptual GCN step) on the Karate Club graph. Click a node to designate it as the focal node. Animated arrows show neighbor messages flowing inward. After one layer, the node's color blends toward the average of its neighbors. Add a second and third layer to see the receptive field expand outward.

The info panel shows the numeric embedding before and after aggregation for the selected node, making the weighted average concrete.

**Learning objective (Bloom's Understand (Level 2)):** Watch a GCN layer aggregate neighbor representations into a node's new embedding and observe how the receptive field expands with each additional layer.

## How to Use

1. **Select a focal node** — click any node on the Karate Club graph.
2. **Choose layer depth** — the slider sets how many GCN layers to simulate (1–3).
3. **Animate** — click "Propagate" to watch messages animate inward for each layer.
4. **Read embeddings** — the right panel shows the feature vector before and after aggregation at each layer.
5. **Reset** — returns all nodes to their initial features.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch06-gcn-message-passing/main.html"
        height="522"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
10–15 minutes

### Prerequisites
Message passing framework (Chapter 6 lecture). Adjacency matrix and degree matrix (Chapter 0). Mean/sum aggregation.

### Activities

1. Select the highest-degree node and run 3 layers. How many unique nodes influenced its final embedding?
2. Pick two structurally symmetric nodes in the Karate Club (e.g., peripheral nodes with the same degree). Run 2 layers. Are their embeddings identical? Why?
3. Sketch the computation tree for a depth-3 GCN centered on one node.

### Assessment Question
Write out the GCN update rule for one layer: \(h_v^{(k)} = \sigma(W \cdot \text{MEAN}(\{h_u^{(k-1)} : u \in \mathcal{N}(v) \cup \{v\}\}))\). Explain each term.

## References

1. Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
2. Hamilton (2020). Graph Representation Learning. Synthesis Lectures on AI and ML.

---
*Part of Chapter 6: GNN Foundations: Message Passing and GCN. Return to the [chapter page](../../chapters/06-gnn-foundations/index.md) or browse all [MicroSims](../index.md).*
