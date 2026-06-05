---
title: Motif Z-Score Explorer
description: Interactive p5.js MicroSim for motif z-score explorer.
image: /sims/ch19-motif-zscore/ch19-motif-zscore.png
og:image: /sims/ch19-motif-zscore/ch19-motif-zscore.png
twitter:image: /sims/ch19-motif-zscore/ch19-motif-zscore.png
social:
   cards: false
quality_score: 0
---

# Motif Z-Score Explorer

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the Motif Z-Score Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

A subgraph pattern becomes a network motif not simply by occurring frequently, but by occurring significantly more often than expected in a random network with the same degree sequence. The Z-score measures this significance: \(Z = (\text{count}_\text{real} - \mu_\text{random}) / \sigma_\text{random}\).

This MicroSim computes Z-scores for wedge, triangle, 4-path, 4-star, and 4-cycle patterns on a network you choose. The bar chart at the bottom shows Z-scores alongside raw counts, making the difference between "common" and "significant" visible.

**Learning objective (Bloom's Analyze (Level 4)):** Distinguish network motifs (statistically over-represented subgraphs, \(Z > 2\)) from anti-motifs (under-represented, \(Z < -2\)) vs. raw frequency, and see how the null model matters.

## How to Use

1. **Choose a network** — load a preset (regulatory network, social network, random graph).
2. **Read the Z-scores** — the bar chart shows Z-scores for each subgraph pattern. Red bars indicate motifs (Z > 2); blue bars indicate anti-motifs (Z < −2).
3. **Compare raw counts** — toggle "Show Raw Counts" to compare Z-scores to absolute frequencies.
4. **Null model** — choose between degree-preserving random rewiring and Erdős–Rényi null models.
5. **Hover a pattern icon** — see the pattern definition and example occurrence in the loaded network.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch19-motif-zscore/main.html"
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
Subgraphs and graphlets (Chapter 19 intro). Z-score and statistical significance. Null models.

### Activities

1. Load the regulatory network. Which pattern has the highest Z-score? Why is this pattern functionally important in gene regulation?
2. Load a random Erdős–Rényi graph. All Z-scores should hover near zero. Verify this.
3. Switch the null model from degree-preserving to Erdős–Rényi on the same network. How do the Z-scores change? Which null model is more conservative?

### Assessment Question
Define a network motif formally using statistical significance. Explain why a feed-forward loop is a motif in transcriptional regulation but not in a social network.

## References

1. Milo et al. (2002). Network Motifs: Simple Building Blocks of Complex Networks. Science.
2. Przulj (2007). Biological network comparison using graphlet degree distribution. Bioinformatics.

---
*Part of Chapter 19: Frequent Subgraph Mining. Return to the [chapter page](../../chapters/19-subgraph-mining/) or browse all [MicroSims](../index.md).*