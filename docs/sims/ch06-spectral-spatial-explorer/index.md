---
title: Spectral vs. Spatial GNN Explorer
description: Interactive p5.js MicroSim for spectral vs. spatial gnn explorer.
image: /sims/ch06-spectral-spatial-explorer/ch06-spectral-spatial-explorer.png
og:image: /sims/ch06-spectral-spatial-explorer/ch06-spectral-spatial-explorer.png
twitter:image: /sims/ch06-spectral-spatial-explorer/ch06-spectral-spatial-explorer.png
social:
   cards: false
quality_score: 0
---

# Spectral vs. Spatial GNN Explorer

<iframe src="main.html" height="532" width="100%" scrolling="no"></iframe>

[Run the Spectral vs. Spatial GNN Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

GNNs can be understood from two equivalent angles. Spatially, each layer aggregates a node's neighborhood features — local, hop-by-hop. Spectrally, aggregation is a low-pass filter on the graph Laplacian's eigenvectors, suppressing high-frequency (rapidly oscillating) signals while preserving smooth (community-level) ones.

This MicroSim lets you click individual Laplacian eigenvectors to paint them onto the graph as node colors. Smooth eigenvectors (low eigenvalues) vary slowly across the graph; oscillatory ones (high eigenvalues) alternate sign between neighbors. The side-by-side view shows both representations simultaneously.

**Learning objective (Bloom's Analyze (Level 4)):** Compare the spatial view (\(k\)-hop neighborhood aggregation) with the spectral view (Laplacian eigenvectors and low-pass filtering) and see that both perspectives describe the same underlying computation.

## How to Use

1. **Pick an eigenvector** — click an eigenvalue in the spectrum panel on the left to paint that eigenvector onto the graph.
2. **Run spatial aggregation** — use the "Aggregate" button to run one round of mean aggregation and see how the graph signal smooths.
3. **Compare panels** — the left shows the signal in spectral space; the right shows it spatially on the graph.
4. **Adjust filter** — a low-pass filter slider shows how spectral truncation approximates the spatial aggregation.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch06-spectral-spatial-explorer/main.html"
        height="532"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
20–30 minutes

### Prerequisites
Laplacian matrix (Chapter 0). Eigenvectors and eigenvalues. Message passing overview (Chapter 6).

### Activities

1. Click eigenvector 1 (constant) — it assigns every node the same value. What does applying a GCN layer to this signal produce?
2. Click the highest eigenvector (most oscillatory). Apply one round of aggregation. Does the signal become smoother?
3. Sketch the spectral filter implied by GCN's \(D^{-1/2}AD^{-1/2}\) aggregation step.

### Assessment Question
Explain why the graph Laplacian eigenvectors form a Fourier basis for graph signals. How does spectral graph convolution generalize classical Fourier convolution?

## References

1. Shuman et al. (2013). The emerging field of signal processing on graphs. IEEE SP Magazine.
2. Defferrard et al. (2016). Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. NeurIPS.

---
*Part of Chapter 6: GNN Foundations: Message Passing and GCN. Return to the [chapter page](../../chapters/06-gnn-foundations/) or browse all [MicroSims](../index.md).*