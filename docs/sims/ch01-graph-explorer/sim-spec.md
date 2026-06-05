<!--
Original MicroSim specification for ch01-graph-explorer (Graph Property Explorer).
Excluded from the built site via exclude_docs (**/sim-spec.md).
Kept here so the spec survives after the <details> block is removed from the chapter.
-->

# Spec: Graph Property Explorer

- sim-id: ch01-graph-explorer
- chapter: 01-intro-to-graphs
- bloom: Apply
- library: p5.js

## Original specification block

<details markdown="1">
<summary>Interactive Graph Builder with Live Property Updates</summary>
Type: MicroSim
**sim-id:** ch01-graph-explorer<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective (Applying — Bloom's Level 3):** Students construct graphs interactively and observe how global properties (degree distribution, average path length, clustering coefficient, connected components) respond to structural changes, developing intuition for the relationship between local edits and emergent network properties.

**Canvas:** 900 × 520 px, responsive (resizes on window resize). Left panel (600 px): interactive graph canvas with physics-based layout (nodes repel each other; edges act as springs). Right panel (300 px): live property readout panel.

**Graph interaction (left panel):**
- **Left-click empty space:** Add a new node at click position. Nodes labeled with sequential integers starting at 0. New node colored light blue.
- **Left-click a node, then left-click another node:** Toggle an edge between the two selected nodes (add if absent, remove if present). First-selected node shown in gold while awaiting the second click.
- **Right-click a node:** Remove the node and all its incident edges immediately.
- **Drag a node:** Reposition it; physics layout updates continuously around the dragged node.

**Live property readout (right panel, ≤100 ms update latency):**
- Node count, edge count
- Average degree (1 decimal place)
- Degree distribution: text-art horizontal bar chart, max width 20 chars, showing P(k) for each observed k
- Average shortest path length (computed with BFS; shows "N/A" if graph disconnected or n > 40)
- Global clustering coefficient (transitivity)
- Number of connected components (sizes of top-3 listed)
- Is bipartite? (Yes / No, using BFS 2-coloring; updates every step)

**Visual encoding:**
- Node color maps degree to a blue gradient: white (degree 0) → light blue (degree 3) → dark indigo (degree 10+)
- Node radius proportional to degree: min 12 px, max 30 px
- Edges are light gray lines; bridge edges (whose removal increases component count) are highlighted orange
- Nodes in different connected components are tinted with distinct background colors (up to 5 distinct tints)

**Interaction — preset selector (dropdown, top of canvas):** "Empty", "Karate Club (10-node sample)", "Complete K5", "Cycle C6", "Star K₁₋₆", "Path P8", "Random ER (n=12, p=0.3)". Selecting a preset replaces the current graph and resets the layout.

**Interaction — node hover tooltip:** Hovering over a node shows a tooltip: "Node 3 | Degree: 4 | Clustering: 0.33 | Eccentricity: 2".

**Interaction — screenshot button:** Saves a PNG of the current graph canvas. Useful for students to document their experiments.

**Implementation notes:** Physics layout uses Hooke's law for edges (spring constant 0.05) and Coulomb repulsion for all node pairs (charge −500). Cap repulsion computation at 30 nearest nodes to maintain 60 fps for n ≤ 50. BFS for path length and bridging uses JavaScript arrays (no external library). Update property panel on every `mouseReleased` event and after each 200 ms physics tick.

</details>
