---
title: Contrastive Loss Surface Explorer
description: Interactive p5.js MicroSim for contrastive loss surface explorer.
image: /sims/ch24-contrastive-loss-explorer/ch24-contrastive-loss-explorer.png
og:image: /sims/ch24-contrastive-loss-explorer/ch24-contrastive-loss-explorer.png
twitter:image: /sims/ch24-contrastive-loss-explorer/ch24-contrastive-loss-explorer.png
social:
   cards: false
quality_score: 0
---

# Contrastive Loss Surface Explorer

<iframe src="main.html" height="482" width="100%" scrolling="no"></iframe>

[Run the Contrastive Loss Surface Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

NT-Xent (Normalized Temperature-scaled Cross-Entropy) is the contrastive loss used in graph SSL. It pulls together two augmented views of the same node (positives) while pushing apart views from different nodes (negatives). Two hyperparameters dominate: augmentation strength (how different the two views are) and temperature τ (how sharply the loss discriminates between positives and negatives).

This MicroSim shows the loss landscape as a 2D heatmap over augmentation strength × temperature. Your current operating point moves as you drag the sliders, and the loss decomposes into alignment (pulling positives together) and uniformity (pushing all embeddings to distribute evenly on the unit sphere).

**Learning objective (Bloom's Apply (Level 3)):** Manipulate augmentation strength and temperature to see how they move the operating point on the NT-Xent loss landscape, building intuition for the alignment-uniformity trade-off.

## How to Use

1. **Adjust augmentation strength** — drag the first slider from "weak" (very similar views) to "strong" (very different views).
2. **Adjust temperature τ** — drag the second slider. Low τ is sharper (hard negatives matter more); high τ is softer.
3. **Read the heatmap** — the operating point moves on the loss surface and the decomposed alignment and uniformity losses update.
4. **View the embedding plot** — the scatter plot shows how embeddings distribute on the unit circle as you change settings.
5. **Find collapse** — move to very weak augmentation + very high temperature to see representation collapse.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch24-contrastive-loss-explorer/main.html"
        height="482"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
15–20 minutes

### Prerequisites
Softmax and cross-entropy (Chapter 0). Graph augmentation concepts. Self-supervised learning basics.

### Activities

1. Set temperature to a very low value (τ = 0.05). How does the loss landscape change? What does this do to gradient signals from hard negatives?
2. Increase augmentation strength gradually. At what point does alignment loss start increasing? Why?
3. Identify the (augmentation, temperature) combination that minimizes total NT-Xent loss. Is this the same as the combination that maximizes downstream accuracy?

### Assessment Question
Write the NT-Xent loss formula for a batch of N positive pairs. Decompose it into alignment and uniformity terms as defined by Wang & Isola (2020). Explain the trade-off between the two.

## References

1. Chen et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). ICML.
2. Wang & Isola (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity. ICML.

---
*Part of Chapter 24: Advanced GNN Topics: In-Context Learning and Uncertainty. Return to the [chapter page](../../chapters/24-advanced-gnn-topics/) or browse all [MicroSims](../index.md).*