<!--
Original MicroSim specification for ch18-louvain-explorer.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Louvain Two-Phase Iteration Explorer

- sim-id: ch18-louvain-explorer
- chapter: 18-community-structure
- bloom: (see body)
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive visualization of Louvain's two-phase modularity optimization</summary>
Type: MicroSim
**sim-id:** ch18-louvain-explorer
**Library:** p5.js
**Status:** Specified

**Learning objective:** Analyzing (Bloom's Level 4) — the student traces how local node moves aggregate into global community structure across multiple passes, and observes the compression operation that enables scaling.

**Canvas:** 900 × 500 px, responsive to window resize.

**Three-panel layout:**
- Left panel (300 px): Original Karate Club graph. Nodes colored by current community assignment. Highlighted node (if any) shown in gold.
- Center panel (300 px): Current super-graph after the last Phase 2 compression. Nodes are community super-nodes sized proportionally to the number of original nodes they represent. Edge thickness proportional to cross-community edge weight.
- Right panel (300 px): Line chart tracking modularity Q across passes (x-axis: pass number, y-axis: Q value, range [0, 1]). A horizontal dashed line marks "Q = 0.3" as the community detection threshold.

**Controls:**
- "Run Phase 1" button: animates node reassignment — for each node (in order), shows the ΔQ arrow pointing to its new community (300ms per node), then snaps to final state.
- "Run Phase 2" button: animates the compression — communities collapse into super-nodes with a brief animation, then the super-graph appears in center panel.
- "Next Pass" button: triggers both Phase 1 and Phase 2 sequentially.
- Slider: "Resolution γ" from 0.5 to 2.0, step 0.1. Resets and reruns when changed.
- "Reset" button: restore initial state.

**Interaction:**
- Click any node in the left panel to see its ΔQ values for each neighboring community (shown as a tooltip bar chart).
- Click any super-node in the center panel to expand it and see its constituent original nodes highlighted in the left panel.

**Data:** Hardcode the Karate Club graph (34 nodes, 78 edges). Ground-truth faction labels available for comparison.
</details>
