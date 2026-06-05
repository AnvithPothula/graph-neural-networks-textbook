<!--
Original MicroSim specification for ch06-spectral-spatial-explorer (Spectral vs. Spatial GNN Explorer).
Excluded from the built site via exclude_docs (**/sim-spec.md).
-->

# Spec: Spectral vs. Spatial GNN Explorer

- sim-id: ch06-spectral-spatial-explorer
- chapter: 06-gnn-foundations
- bloom: Analyze
- library: p5.js

## Original specification block

<details markdown="1">
<summary>Interactive diagram comparing spectral and spatial views of graph convolution</summary>

**sim-id:** ch06-spectral-spatial-explorer<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Analyzing (Bloom's Level 4) — Students compare the spectral domain (Laplacian eigenvectors) and spatial domain (neighborhood aggregation) views of graph operations, identifying which operations are equivalent and which differ.

**Canvas:** 800×500px, responsive to window resize. Split into two labeled panels side by side:

**Left panel — Spatial Domain (400×500):**
- A 12-node graph drawn with force layout (fixed seed positions for determinism)
- Two seed nodes marked in orange; an interactive depth slider (K = 1, 2, 3) below the panel
- Clicking "Propagate" runs K rounds of mean aggregation from the seed nodes, with concentric highlight rings showing the receptive field expanding hop-by-hop (color: blue→teal→green by distance)
- Node labels show current h_v value (scalar, 2 decimal places) next to each node
- Title: "Spatial: K-hop neighborhood aggregation"

**Right panel — Spectral Domain (400×500):**
- A bar chart of the 12 eigenvectors (sorted by eigenvalue) of the normalized Laplacian
- Each bar colored by the eigenvalue magnitude (low = blue, high = red)
- Clicking a bar highlights the corresponding eigenvector values on the graph in the left panel (node colors = eigenvector component values, diverging colormap)
- A "low-pass filter" toggle: when on, zeros out all eigenvalues above a threshold (slider 0–λ_max), showing the smoothing effect of low-pass spectral filtering on the spatial graph
- Title: "Spectral: Laplacian eigenvectors"

**Interaction bridge:** actions in one panel synchronize to the other:
- Propagating K hops in the spatial view updates the spectral filter band in the right panel to show which frequency components are active
- Clicking a spectral eigenvector in the right panel highlights nodes by component magnitude in the left panel

**Hover tooltips:** every node shows "Node [id] | h_v: [value] | degree: [d]"; every bar shows "Eigenvector [k] | λ_k = [value]"

**Implementation notes:** pre-compute the 12-node graph's Laplacian eigendecomposition (all 12 eigenvalues and eigenvectors) as hardcoded arrays in the JavaScript. Use D3 or p5.js for rendering.

</details>
