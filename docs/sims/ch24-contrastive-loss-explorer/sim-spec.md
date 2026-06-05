<!--
Original MicroSim specification for ch24-contrastive-loss-explorer.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Contrastive Loss Surface Explorer

- sim-id: ch24-contrastive-loss-explorer
- chapter: 24-advanced-gnn-topics
- bloom: Apply
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive visualization of how augmentation strength affects contrastive loss</summary>
Type: MicroSim
**sim-id:** ch24-contrastive-loss-explorer<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Students develop intuition for how augmentation strength and temperature interact to determine the shape of the contrastive loss landscape (Bloom's: Applying — manipulating parameters to observe their effect on the learning objective).

**Canvas:** 780 × 440 px, responsive. Background: #f0f4ff.

**Layout:** Two panels side by side.

Left panel — "Augmented Views" (400 × 400): A small force-directed graph with 8 nodes. Two copies of this graph are shown as overlapping translucent layers. View 1 is blue; View 2 is orange. Dropped edges are drawn as dotted lines; masked features are shown as nodes with a gray center. A slider controls the augmentation strength (0–50%), which simultaneously adjusts edge dropout and feature masking percentages.

Right panel — "Loss Landscape" (360 × 400): A heatmap showing the contrastive loss value as a function of:
- X-axis: cosine similarity between the two views (−1 to 1)
- Y-axis: temperature τ (0.1 to 1.0)
A crosshair cursor marks the current (similarity, τ) point based on the current augmentation settings and temperature slider.

**Controls:**
- "Augmentation Strength" slider: 0–50% (adjusts both edge dropout and feature masking)
- "Temperature τ" slider: 0.1–1.0
- "Batch Size" dropdown: 16 / 64 / 256 (affects the number of negatives in the loss formula)
- "Compute Loss" button: triggers a mock training step animation

**Interactions:**
- Clicking a node in the left panel shows its embedding vector (4D) as a bar chart in a tooltip
- Hovering over the loss heatmap shows the loss value, gradient magnitude, and what the model "would learn" at that point (tooltip: "High τ → soft distribution, slow learning; Low τ → sharp distribution, faster learning but unstable")
- A "Why does this work?" button shows a one-paragraph pop-up explaining the InfoNCE lower bound

**Responsiveness:** panels stack vertically on screens narrower than 650px
</details>
