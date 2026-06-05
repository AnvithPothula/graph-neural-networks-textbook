<!--
Original MicroSim specification for ch24-graph-contrastive-learning.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Graph Contrastive Learning — Two-View Pipeline

- sim-id: ch24-graph-contrastive-learning
- chapter: 24-advanced-gnn-topics
- bloom: Create
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive contrastive learning pipeline with augmentation controls</summary>
Type: MicroSim
**sim-id:** ch24-graph-contrastive-learning<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Students observe how different graph augmentation strategies create two views and how the contrastive loss is computed from their embeddings (Bloom's: Understanding — explaining how augmentation and loss interact to produce invariant representations).

**Canvas:** 900 × 500 px, responsive to window resize. Dark gray background (#1a1a2e) for contrast.

**Layout:** Three columns.

Left column — "Original Graph": A small p5.js force-directed graph with 10 nodes and ~15 edges, colored by node type (two classes: blue and orange). Node labels 1–10.

Middle column — "Two Views" (split into top half and bottom half):
- Top: View 1 — the same graph with a subset of edges drawn in red (indicating dropped edges). A header shows "View 1: Edge Dropout (15%)".
- Bottom: View 2 — the same graph with some node feature bars grayed out (indicating masked features). A header shows "View 2: Feature Masking (20%)".

Right column — "Embedding Space": A 2D scatter plot (t-SNE visualization) showing 10 pairs of points. Each pair (view 1 embedding, view 2 embedding of the same node) is connected by a dashed line. Points from the same node are colored the same; points from different nodes have different colors. The loss value is shown below the plot.

**Controls (bottom panel):**
- "Edge Dropout %" slider: 0–40%, updates View 1 in real time
- "Feature Masking %" slider: 0–40%, updates View 2 in real time
- "Temperature τ" slider: 0.1–1.0, updates the loss display
- "Resample Views" button: generates new random augmentations
- Augmentation type dropdown for View 1: "Edge Dropout | Node Dropout | Feature Masking"
- Augmentation type dropdown for View 2: same options

**Interactions:**
- Clicking a node in the Original Graph highlights its corresponding pair of points in the embedding scatter plot and draws connecting lines
- Hovering over a point in the scatter plot shows: node ID, view number, embedding coordinates, cosine similarity to its pair
- The loss value updates in real time when sliders change
- Hovering over the loss display shows a tooltip explaining the NT-Xent formula in plain language

**Visual style:** blue/orange for node classes, red for dropped edges, gray for masked features, dashed lines for positive pairs in embedding space

**Responsiveness:** below 700px, display in single-column scroll layout
</details>
