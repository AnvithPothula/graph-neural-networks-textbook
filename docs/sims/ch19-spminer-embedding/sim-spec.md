<!--
Original MicroSim specification for ch19-spminer-embedding.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: SPMiner Order Embedding Space

- sim-id: ch19-spminer-embedding
- chapter: 19-subgraph-mining
- bloom: Understand
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive 2D projection of order embedding space showing query-neighborhood relationships</summary>
Type: MicroSim
**sim-id:** ch19-spminer-embedding
**Library:** p5.js
**Status:** Specified

**Learning objective:** Understanding (Bloom's Level 2) — the student observes how order embeddings encode the subgraph partial order geometrically, and how frequency estimation reduces to counting dominated points in embedding space.

**Canvas:** 900 × 500 px, responsive to window resize.

**Layout:**
- Left panel (200 px): list of 6 predefined query graphs of increasing size (edge, triangle, 4-cycle, diamond, 4-clique, path-of-length-3). Clicking selects the active query.
- Center panel (500 px): 2D scatter plot of order embedding space (embed_dim=2). Each point represents a node neighborhood embedding from the Karate Club graph. Points colored by size of neighborhood (small = blue, large = orange). Active query embedding shown as a large gold star. Points that "dominate" the query (i.e., are coordinate-wise >= query in both dimensions) shown in green (estimated subgraph matches). Non-dominating points shown in gray.
- Right panel (200 px): estimated frequency (green match count / total points), true frequency from VF2 (for reference), and a precision metric.

**Controls:**
- Query selector (left panel): click any of the 6 query graphs.
- Slider: "embedding dimension" (2D only — this sim is fixed at 2D for visualization).
- Slider: "epsilon threshold" (0.001 to 0.1): changes which points count as "matches." Matching points turn green or revert to gray in real time.
- "Show cone" toggle: draws the coordinate-wise dominance cone extending from the query point toward (+∞, +∞).

**Interaction:**
- Hovering a point shows a tooltip: neighborhood size, order violation value, match/no-match status.
- The cone visualization updates in real time as epsilon changes.

**Data:** Hardcode 2D order embeddings for the 34 Karate Club neighborhoods. Pre-compute true VF2 frequencies for the 6 query graphs. Embeddings can be synthetic but must satisfy the order property for the known ground-truth containment relationships.

**Visual style:** White background, indigo axis lines, gold query star, green matches, gray non-matches. Font: sans-serif, 12px.
</details>
