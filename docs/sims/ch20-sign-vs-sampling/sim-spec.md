<!--
Original MicroSim specification for ch20-sign-vs-sampling.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: SIGN Architecture vs. Neighbor Sampling Architecture

- sim-id: ch20-sign-vs-sampling
- chapter: 20-scaling-gnns
- bloom: Analyze
- library: p5.js

## Original specification

<details markdown="1">
<summary>SIGN vs. Neighbor Sampling — Side-by-Side Architecture Comparison</summary>
Type: interactive-infographic
**sim-id:** ch20-sign-vs-sampling
**Library:** p5.js
**Status:** Specified

Two-panel interactive diagram (each panel 380×500px, total 760×500px, responsive to window resize) comparing SIGN pre-computation and neighbor-sampling training pipelines side-by-side.

**Left panel — Neighbor Sampling:**
- Show a target node (orange circle) and 2 layers of sampled neighbors (smaller circles)
- Animate arrows showing messages flowing inward from layer-2 neighbors to layer-1 to target
- Label "In-loop graph ops: O(K^L) per step"
- Controls: sliders for K (fan-out, range 2-20) and L (layers, range 1-3)
- As K increases, show more neighbor circles appearing at each hop
- As L increases, add additional hop rings

**Right panel — SIGN Pre-computation:**
- Top section: static "Offline" block showing A^0 X, A^1 X, A^2 X, A^3 X as stacked matrices with colored bars (one color per hop level)
- Bottom section: dynamic "Online Training" block showing a multi-input MLP receiving the 4 pre-computed vectors
- Label "No graph ops in training loop: O(d) per node"
- Arrow from offline block to online block labeled "pre-computed once"
- Controls: slider for K (range 0-4) adding/removing precomputed diffusion levels

**Interactions:**
- Clicking any node circle in the left panel opens a tooltip explaining that node's role (target, 1-hop neighbor, 2-hop neighbor)
- Clicking any matrix block in the right panel opens a tooltip explaining what A^k X computes and its neighborhood interpretation
- Toggle button "Show Memory Usage" adds a colored bar below each panel showing relative memory proportional to subgraph size vs. pre-computed matrix size

**Learning objective (Applying — Bloom's Taxonomy):** Students can manipulate the fan-out K and depth L to observe how sampling complexity scales and compare it with SIGN's constant-per-node pre-computation cost.

Implementation: p5.js with DOM sliders and tooltip divs. Mouse-over on all interactive elements. Responsive via `windowResized()` callback.
</details>
