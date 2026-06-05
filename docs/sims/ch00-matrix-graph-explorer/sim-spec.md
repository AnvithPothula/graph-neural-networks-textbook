<!--
Original MicroSim specification for ch00-matrix-graph-explorer (Matrix × Graph Explorer).
Excluded from the built site via exclude_docs (**/sim-spec.md).
Kept here so the spec survives after the <details> block is removed from the chapter.
-->

# Spec: Matrix × Graph Explorer

- sim-id: ch00-matrix-graph-explorer
- chapter: 00-math-prerequisites
- bloom: Understand
- library: p5.js

## Original specification block

<details markdown="1">
<summary>Interactive A·X Matrix Multiplication Visualizer</summary>
Type: MicroSim
**sim-id:** ch00-matrix-graph-explorer<br/>
**Library:** p5.js<br/>
**Status:** Built

**Learning objective (Understanding — Bloom's Level 2):** Students observe how different normalizations of the adjacency matrix (raw sum, mean, and symmetric) produce qualitatively different aggregated features, building intuition for why GCN uses symmetric normalization \(\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\) rather than the raw adjacency matrix.

**Canvas:** 900 × 480 px, responsive (resizes on window resize). Left panel (550 px): graph visualization with adjacency matrix overlay. Right panel (350 px): output feature vectors and normalization mode selector.

**Default graph:** 5-node hub-and-spoke graph — one hub node with degree 4 connected to four leaf nodes each with degree 1. This asymmetry makes the normalization difference most visible: the hub dominates in raw-sum mode but is tamed in mean and symmetric modes.

**Visualizations:**
- Graph displayed with nodes as indigo circles; edge lines in gray.
- Node input feature \(x_i\) displayed as a numeric label inside each node (1D scalar, range 0.1–2.0).
- Output feature \((\mathbf{A}\mathbf{x})_i\) (or the normalized variant) displayed as a colored bar alongside each node.
- Bar color intensity encodes output magnitude: white (low) → dark indigo (high).

**Normalization modes (three toggle buttons, only one active at a time):**
1. **Raw** \(\mathbf{A}\mathbf{x}\): Each node receives the sum of neighbor features — hub nodes output large values.
2. **Mean** \(\mathbf{D}^{-1}\mathbf{A}\mathbf{x}\): Each node receives the average of neighbor features — hub nodes tamed.
3. **Symmetric** \(\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\mathbf{x}\): GCN normalization — intermediate weighting based on geometric mean of degrees.

**Interaction — node click:** Clicking a node highlights it in gold, shows its neighbors in yellow, and opens an infobox: "Node 0 (hub): raw output = x₁+x₂+x₃+x₄ = 5.2 → mean = 5.2/4 = 1.30 → symmetric = Σ x_j / √(d_0 · d_j) = 2.60". The infobox updates when the normalization mode changes.

**Interaction — feature slider:** Each node has a small slider (draggable) controlling its input feature \(x_i\) in [0.1, 2.0]. Moving any slider updates all output bars in real time without requiring a full recompute.

**Interaction — graph preset dropdown:** "Hub-Spoke (default)", "Path Graph (n=5)", "Complete Graph K4", "Karate Club Sample (10 nodes)". Selecting a preset replaces the graph, preserves the active normalization mode, and randomizes input features.

**Interaction — adjacency matrix panel:** A small grid below the graph shows the current adjacency matrix as colored squares (gray = 0, indigo = 1). Clicking any off-diagonal cell toggles the corresponding edge.

</details>
