---
title: SIGN vs. Neighbor Sampling Architecture
description: Interactive p5.js MicroSim for sign vs. neighbor sampling architecture.
image: /sims/ch20-sign-vs-sampling/ch20-sign-vs-sampling.png
og:image: /sims/ch20-sign-vs-sampling/ch20-sign-vs-sampling.png
twitter:image: /sims/ch20-sign-vs-sampling/ch20-sign-vs-sampling.png
social:
   cards: false
quality_score: 0
---

# SIGN vs. Neighbor Sampling Architecture

<iframe src="main.html" height="572" width="100%" scrolling="no"></iframe>

[Run the SIGN vs. Neighbor Sampling Architecture MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Training GNNs at scale requires either mini-batching with neighbor sampling (GraphSAGE, PinSAGE) or decoupling feature propagation from prediction (SIGN, SGC). Neighbor sampling reduces graph operations per step but still requires them in the training loop and they grow exponentially with depth. SIGN precomputes all \(A^k \cdot X\) matrices once offline; training then involves only an MLP with no graph operations.

This MicroSim shows both architectures side by side. The left panel animates the expanding neighbor-sampling tree; the right panel shows SIGN's offline precomputed matrices flowing into an online MLP. A memory-usage toggle reveals the footprint difference for large fan-out.

**Learning objective (Bloom's Apply (Level 3)):** Compare neighbor sampling's in-loop \(\mathcal{O}(K^L)\) graph operations against SIGN's one-time precomputation of \(A^k \cdot X\) and \(\mathcal{O}(d)\)-per-node online training.

## How to Use

1. **Set fan-out \(K\)** — the first slider sets how many neighbors are sampled per node per layer.
2. **Set depth \(L\)** — the second slider sets how many GNN layers (1–3).
3. **Toggle memory bars** — click "Show Memory Usage" to compare memory footprints.
4. **Read costs** — the bottom of each panel shows the in-loop graph operation count for neighbor sampling and the \(\mathcal{O}(d)\)-per-node cost for SIGN.
5. **Compare** — observe how exponential growth (\(K^L\)) dominates as \(K\) and \(L\) increase.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch20-sign-vs-sampling/main.html"
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
GNN message passing (Chapter 6). Matrix-vector multiplication (Chapter 0). Mini-batch training concepts.

### Activities

1. Set \(K=10\), \(L=3\). Compute the theoretical neighbor sampling cost: \(10^3 = 1000\) node lookups per target. What does SIGN's cost reduce to?
2. With memory bars on, observe how SIGN's precomputed matrices scale with the number of layers \(L\) vs. how sampling scales with \(K\).
3. For ogbn-arxiv (170K nodes, average degree ≈ 13), estimate the cost of neighbor sampling with \(K=15\), \(L=3\). Compare to SIGN.

### Assessment Question
Explain why SIGN can be trained without any graph operations in the training loop. What is the key insight that makes precomputing \(A^k \cdot X\) sufficient? What does SIGN give up compared to a full message-passing GNN?

## References

1. Frasca et al. (2020). SIGN: Scalable Inception Graph Neural Networks. ICML-W.
2. Hamilton et al. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.

---
*Part of Chapter 20: Scaling GNNs to Billion-Node Graphs. Return to the [chapter page](../../chapters/20-scaling-gnns/) or browse all [MicroSims](../index.md).*
