<!--
Original MicroSim specification for ch03-pagerank-power-iteration (PageRank Power Iteration Simulator).
Excluded from the built site via exclude_docs (**/sim-spec.md).
Kept so the spec survives after the <details> block is removed from the chapter.
-->

# Spec: PageRank Power Iteration Simulator

- sim-id: ch03-pagerank-power-iteration
- chapter: 03-link-analysis-pagerank
- bloom: Apply
- library: p5.js

## Original specification block

<details markdown="1">
<summary>Interactive PageRank Power Iteration on a Small Graph</summary>
Type: MicroSim
**sim-id:** ch03-pagerank-power-iteration<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective (Applying — Bloom's Level 3):** Students observe how the power iteration algorithm propagates rank through the graph over successive iterations, building intuition for why high-degree, well-connected nodes accumulate disproportionate rank, and how the teleportation parameter controls convergence speed.

**Canvas:** 900 × 520 px, responsive (resizes on window resize). Left panel (600 px): graph visualization. Right panel (300 px): convergence plot + controls.

**Default graph:** 8-node directed graph with a spider trap (nodes 6 and 7 form a two-node mutual loop with no exits), dead end (node 5 has no outgoing edge), and a hub-spoke cluster (node 0 links to nodes 1, 2, 3). This configuration lets students observe the pathological cases alongside normal behavior.

**Node rendering:** Each node is drawn as a circle whose radius is proportional to its current PageRank score (min 12 px, max 40 px). Node color transitions from light blue (low rank) to dark indigo (high rank) using a continuous color scale. Node ID is displayed inside the circle. Edges are drawn as curved arrows; edge color is gray. The node with the current highest rank is highlighted with a golden glow.

**Convergence plot (right panel):** A real-time line chart of \( \|\mathbf{r}^{(t+1)} - \mathbf{r}^{(t)}\|_1 \) on a log-scale y-axis against iteration number on the x-axis. A horizontal dashed line marks the convergence threshold \( 10^{-6} \).

**Controls (right panel beneath convergence plot):**

- **Damping factor slider:** \( d \in [0.5, 1.0] \), default 0.85, step 0.05. Updates label "d = 0.85". Changing \( d \) resets the iteration.
- **Step button:** Advances one power iteration step. Animates rank propagation: for 500 ms, draw arrows proportional to the rank being transferred along each edge.
- **Run / Pause button:** Runs iterations at 2 per second until convergence, then stops. Label changes between "Run" and "Pause".
- **Reset button:** Returns \( \mathbf{r}^{(0)} \) to uniform distribution, clears convergence plot.
- **Graph selector dropdown:** Presets — "Default (spider trap + dead end)", "Karate Club (10-node sample)", "Star graph", "Cycle graph (aperiodic test)". Each preset loads a different adjacency list.

**Interaction — node click:** Clicking a node opens an infobox overlay (positioned near the node) showing: node ID, current PageRank score (4 decimal places), in-degree, out-degree, and a mini-bar chart comparing this node's score to the top-3 nodes. Clicking elsewhere closes the infobox.

**Interaction — iteration counter:** A prominent "Iteration: 0" label at the top of the left panel updates each step. At convergence, the label changes to "Converged at iteration N" in green text.

**Implementation notes:** Maintain the rank vector as a Float64Array of length 8 (or the selected graph size). Matrix-vector multiplication performed in JavaScript. p5.js `draw()` is called only when the simulation state changes (non-animated state: `noLoop()`; animated steps re-enter `loop()` for 500 ms). For the Karate Club preset, use the 10 highest-degree nodes from the full Karate Club graph adjacency list embedded as a hard-coded constant.

</details>
