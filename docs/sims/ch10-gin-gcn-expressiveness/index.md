---
title: GIN vs. GCN Expressiveness Demonstrator
description: Interactive p5.js MicroSim for gin vs. gcn expressiveness demonstrator.
image: /sims/ch10-gin-gcn-expressiveness/ch10-gin-gcn-expressiveness.png
og:image: /sims/ch10-gin-gcn-expressiveness/ch10-gin-gcn-expressiveness.png
twitter:image: /sims/ch10-gin-gcn-expressiveness/ch10-gin-gcn-expressiveness.png
social:
   cards: false
quality_score: 0
---

# GIN vs. GCN Expressiveness Demonstrator

<iframe src="main.html" height="542" width="100%" scrolling="no"></iframe>

[Run the GIN vs. GCN Expressiveness Demonstrator MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

GIN replaces mean aggregation with sum aggregation followed by an MLP, making it as powerful as the 1-WL test. But even GIN has a ceiling: pairs of non-isomorphic graphs that WL cannot distinguish will fool GIN too.

This MicroSim loads the canonical "mean vs. sum" example. A slider controls aggregation layers \(K\). On the left, mean-aggregation embeddings collapse after \(K=1\); on the right, sum-aggregation embeddings diverge. A 3-regular pair at the bottom demonstrates that even GIN cannot separate them, no matter how many layers.

**Learning objective (Bloom's Analyze (Level 4)):** Show why GIN (sum aggregation) is strictly more expressive than GCN (mean aggregation). With uniform features, mean collapses every node to the same embedding; sum preserves structure — yet both fail on 3-regular pairs.

## How to Use

1. **Select an example** — choose "Mean fails" or "Both fail (3-regular)" from the dropdown.
2. **Adjust \(K\)** — the aggregation layers slider controls how many message-passing rounds to apply.
3. **Read the embedding table** — the table on the right shows each node's embedding vector after \(K\) layers. Identical rows mean the model assigns the same representation to different nodes.
4. **Toggle aggregator** — switch between Mean and Sum to compare.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch10-gin-gcn-expressiveness/main.html"
        height="542"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
20–30 minutes

### Prerequisites
GCN (Chapter 6). WL test and GNN expressiveness (Chapter 9). Injective functions.

### Activities

1. Load the "mean fails" example, set \(K=1\). Find two structurally different nodes that get the same mean-aggregation embedding. Show that sum distinguishes them.
2. Load the 3-regular example. Confirm that sum aggregation gives identical embeddings to all nodes regardless of \(K\). Explain why.
3. Describe what architectural change (beyond sum aggregation) would be needed to distinguish the 3-regular pair.

### Assessment Question
Formally state the GIN theorem: under what conditions on the aggregation function \(\psi\) is \(h_v = \psi(h_v, \{h_u : u \in \mathcal{N}(v)\})\) as powerful as the 1-WL test? Prove the sum-MLP construction achieves this.

## References

1. Xu et al. (2019). How Powerful are Graph Neural Networks? ICLR.
2. Morris et al. (2019). Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks. AAAI.

---
*Part of Chapter 10: Designing Powerful Encoders: GIN and Beyond. Return to the [chapter page](../../chapters/10-powerful-gnns/index.md) or browse all [MicroSims](../index.md).*