---
title: WL Color Refinement Simulator
description: Interactive p5.js MicroSim for wl color refinement simulator.
image: /sims/ch02-wl-refinement/ch02-wl-refinement.png
og:image: /sims/ch02-wl-refinement/ch02-wl-refinement.png
twitter:image: /sims/ch02-wl-refinement/ch02-wl-refinement.png
social:
   cards: false
quality_score: 0
---

# WL Color Refinement Simulator

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the WL Color Refinement Simulator MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The Weisfeiler–Lehman (WL) test iteratively relabels each node with a hash of its own color and the multiset of its neighbors' colors. If two graphs ever diverge in their color histograms, they are provably non-isomorphic. If they converge to the same histograms, they may or may not be the same.

This MicroSim shows WL refinement on two graphs side by side. Four preset pairs are available: an easy pair (WL separates them in one round), a triangle vs. path-of-3 (separated in two), and two structurally identical 3-regular graphs that WL cannot distinguish at all. Use the examples to see where the test succeeds and where it hits its theoretical limit.

**Learning objective (Bloom's Analyze (Level 4)):** Run Weisfeiler–Lehman color refinement on graph pairs that 1-WL can and cannot distinguish, building intuition for the expressiveness ceiling of message-passing GNNs.

## How to Use

1. **Choose a graph pair** — click one of the preset buttons at the top to load two comparison graphs.
2. **Step forward** — click "Next Step" to run one WL refinement round. Node colors update to reflect new hashes.
3. **Read the histograms** — the bar charts below each graph show the color distribution. When they match, WL cannot distinguish the graphs.
4. **Reset** — click "Reset" to return to the initial uniform coloring.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch02-wl-refinement/main.html"
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
Understanding of graph isomorphism (Chapter 1). Basic familiarity with hashing as a concept.

### Activities

1. Load the "3-regular indistinguishable" pair. Run WL to convergence. Confirm the color histograms are identical. Try to find a structural difference by inspection.
2. Load the "triangle vs. path" pair. Predict after which round WL will separate them, then verify.
3. Explain why WL refinement is equivalent to one round of a GNN with a sum aggregator and injective update function.

### Assessment Question
What property of a pair of graphs guarantees WL cannot distinguish them? Give a concrete graph pair example and explain why the test fails on it.

## References

1. Weisfeiler & Lehman (1968). A reduction of a graph to a canonical form and an algebra arising during this reduction.
2. Xu et al. (2019). How Powerful are Graph Neural Networks? ICLR.

---
*Part of Chapter 2: Graph Properties and Traditional ML Features. Return to the [chapter page](../../chapters/02-graph-properties-and-features/index.md) or browse all [MicroSims](../index.md).*