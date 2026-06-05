---
title: WL Refinement and GIN Motivation
description: Interactive p5.js MicroSim for wl refinement and gin motivation.
image: /sims/ch09-wl-refinement/ch09-wl-refinement.png
og:image: /sims/ch09-wl-refinement/ch09-wl-refinement.png
twitter:image: /sims/ch09-wl-refinement/ch09-wl-refinement.png
social:
   cards: false
quality_score: 0
---

# WL Refinement and GIN Motivation

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the WL Refinement and GIN Motivation MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

This MicroSim focuses on the theoretical underpinning of GNN expressiveness via the Weisfeiler–Lehman (1-WL) test. It shows three cases: (1) graphs easily distinguished in round 1, (2) graphs that need multiple rounds, and (3) regular graph pairs that 1-WL provably cannot distinguish — motivating the design of GIN.

Color histograms update after each round \(r\), and a "WL result" indicator shows whether the two graphs \(G_1\) and \(G_2\) are separated. When 1-WL fails, the panel highlights why a sum aggregator with an injective update (GIN) is needed.

**Learning objective (Bloom's Apply (Level 3)):** Run 1-WL color refinement on graph pairs the test can and cannot distinguish, with per-graph color histograms making the power and limits of neighborhood aggregation concrete.

## How to Use

1. **Pick a graph pair** — three presets available: easy, moderate, and WL-hard (regular graphs).
2. **Step** — run one WL refinement round at a time.
3. **Read the histograms** — matching histograms mean 1-WL cannot distinguish the pair.
4. **Check the indicator** — the "SEPARATED / NOT SEPARATED" status updates after each round.
5. **View the GIN fix** — toggle "Show GIN" to overlay how a GIN with sum aggregation would distinguish the hard pair.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch09-wl-refinement/main.html"
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
WL test basics (Chapter 2). Message passing (Chapter 6). Aggregation functions (sum, mean, max).

### Activities

1. Verify that mean aggregation conflates two different-size neighborhoods. Construct a minimal example (two graphs \(G_1\), \(G_2\)) that mean-aggregation GCN cannot distinguish but 1-WL can.
2. Show that sum aggregation is injective over the multisets of neighbor colors \(\{c_u : u \in \mathcal{N}(v)\}\) in the examples provided.
3. The WL-hard regular graph pair has the same color histogram at every round \(r\). Draw the two graphs and identify which substructure differs.

### Assessment Question
State and prove the Xu et al. (2019) theorem: a GNN is at most as powerful as the 1-WL test if and only if its aggregation function is not injective over multisets \(\{\!\{c_u : u \in \mathcal{N}(v)\}\!\}\).

## References

1. Xu et al. (2019). How Powerful are Graph Neural Networks? ICLR.
2. Weisfeiler & Lehman (1968). A reduction of a graph to a canonical form.

---
*Part of Chapter 9: Theory of GNNs: Expressiveness and the WL Test. Return to the [chapter page](../../chapters/09-gnn-theory/) or browse all [MicroSims](../index.md).*
