<!--
Original MicroSim specification for ch02-wl-refinement (WL Color Refinement Simulator).
Excluded from the built site via exclude_docs (**/sim-spec.md).
Kept here so the spec survives after the <details> block is removed from the chapter.
-->

# Spec: WL Color Refinement Simulator

- sim-id: ch02-wl-refinement
- chapter: 02-graph-properties-and-features
- bloom: (see spec body)
- library: p5.js

## Original specification block

<details markdown="1">
<summary>Side-by-Side Weisfeiler-Lehman Color Refinement on Two Graphs</summary>
Type: MicroSim
**sim-id:** ch02-wl-refinement<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective (Analyzing — Bloom's Level 4):** Students observe the WL refinement process on carefully chosen graph pairs — including pairs WL distinguishes and pairs it cannot — developing intuition for which structural differences are captured by neighborhood-aggregation algorithms and which escape detection, directly motivating the expressiveness results in Chapters 9 and 10.

**Canvas:** 900 × 520 px, responsive (resizes on window resize). Two graph panels side by side (each ≈420 × 400 px), separated by an 8 px vertical divider. Control bar along the bottom (80 px height).

**WL algorithm (JavaScript implementation):**
1. **Step 0 (initialization):** Assign label = degree of each node (for unlabeled graphs). Nodes with the same degree get the same initial color.
2. **Step t (refinement):** For each node v, form the string `str(label_v) + ":" + sorted([label_u for u in neighbors(v)]).join(",")`. Hash this string to a compact integer label using a deterministic hash (e.g., FNV-1a or a JavaScript Map from string → sequential integer). Assign the integer as the new label.
3. **Termination:** Repeat up to 6 steps, or until no node's label changes (convergence).

**Color encoding:** A 12-color palette is used. Label integer `k` maps to `palette[k % 12]`. Colors are consistent across both graphs: if node A in graph 1 and node B in graph 2 share the same label, they are drawn in the same color, making correspondence explicit.

**Graph panels:** Each node rendered as a circle (radius 22 px). Current WL label displayed as an integer inside the circle. Edges drawn as gray lines. Panel titles "Graph 1" and "Graph 2" appear at the top of each panel.

**Controls (bottom bar):**
- **Step button:** Advances one WL iteration on both graphs simultaneously. Animates for 700 ms: first highlight each node's neighborhood (yellow border on neighbors), then flash to the new color.
- **Auto / Stop toggle button:** Runs iterations at 1.5 per second until convergence. Label changes between "Auto" and "Stop".
- **Reset button:** Returns both graphs to step 0 (initial degree-based labels), clears the status panel.
- **Graph pair selector (dropdown):**
  - "Distinguishable pair — Triangle vs. Path" (default): 3-cycle vs. 3-path. WL distinguishes after 1 step (degree differences).
  - "WL-indistinguishable pair — 3-regular": Two non-isomorphic 3-regular graphs (6 nodes each) that WL cannot distinguish. All nodes in both graphs receive identical color sequences at every step.
  - "Karate Club sample vs. ER": 10-node Karate Club sample vs. ER graph with the same n and m.
  - "Star K₁₋₅ vs. Path P₆": Distinguishable at step 0 (hub has degree 5, leaves have degree 1; path has nodes of degree 1 and 2).

**Status panel (centered between the two graph panels, below titles):**
- Displays current step: "Step: 0".
- After each step, shows one of: "Distinguishable ✓" (green, if color histograms differ), "Same histogram" (yellow, if histograms match but not converged), or "Converged — Same histogram" (blue, if labels stable and histograms still match).

**Interaction — node click:** Clicking any node opens an infobox overlay near the node: "Node 3 | Step 2 label: 7 | Neighbor labels: {4, 7, 7} | New label input: '7:4,7,7'". Simultaneously highlights the node and its neighbors in both graphs. Clicking elsewhere closes the infobox.

**Interaction — color legend (right of control bar):** A small horizontal strip showing all 12 palette colors with their current label assignments (label integer → color swatch). Updates after each step.

</details>
