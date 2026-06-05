<!--
Original MicroSim specification for ch18-girvan-newman.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Girvan-Newman Step-by-Step on the Karate Club Graph

- sim-id: ch18-girvan-newman
- chapter: 18-community-structure
- bloom: Understand
- library: p5.js

## Original specification

<details markdown="1">
<summary>Girvan-Newman algorithm animated on the Karate Club graph</summary>
Type: MicroSim
**sim-id:** ch18-girvan-newman
**Library:** p5.js
**Status:** Specified

**Learning objective:** Understand (Bloom's Level 2) — the student observes how edge betweenness centrality identifies bridge edges and how their successive removal exposes community structure.

**Canvas:** 760 × 480 px, responsive to window resize.

**Layout:**
- Left panel (500 px wide): Karate Club graph with 34 nodes, force-directed layout. Nodes colored by current component membership (start all the same color, split into distinct colors as components separate). Edges colored by betweenness value: gradient from light gray (low betweenness) to vivid red (highest betweenness).
- Right panel (260 px): current iteration number, current modularity Q, list of last 5 removed edges.

**Controls (below the canvas):**
- "Step" button: remove one edge (highest betweenness), recompute betweenness, rerender.
- "Auto Play" button: continuously step every 800ms until graph is fully disconnected.
- "Reset" button: restore original graph.

**Interaction:**
- Hovering any edge shows a tooltip with its betweenness value and rank.
- Hovering any node shows its degree and community ID.

**Behavior:**
- On each step, highlight the to-be-removed edge in thick orange before deleting it (200ms pause).
- After removal, recompute betweenness and update edge colors immediately.
- Track and display modularity Q after each removal (using the formula shown in §18.4).
- When Q is maximized, display a gold banner: "Maximum modularity reached: Q = {value}".

**Visual style:** White background, indigo node fill, node labels showing node ID. Font: sans-serif, 12px.

**Data:** Hardcode the adjacency list of the Karate Club graph (34 nodes, 78 edges).
</details>
