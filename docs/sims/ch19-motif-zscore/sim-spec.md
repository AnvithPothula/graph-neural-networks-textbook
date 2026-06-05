<!--
Original MicroSim specification for ch19-motif-zscore.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Motif Z-Score Explorer

- sim-id: ch19-motif-zscore
- chapter: 19-subgraph-mining
- bloom: (see body)
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive motif significance profile for small connected subgraphs</summary>
Type: MicroSim
**sim-id:** ch19-motif-zscore
**Library:** p5.js
**Status:** Specified

**Learning objective:** Analyzing (Bloom's Level 4) — the student computes and interprets Z-scores for different subgraph patterns across multiple graph types, distinguishing statistical significance from raw frequency.

**Canvas:** 800 × 480 px, responsive to window resize.

**Layout:**
- Left panel (300 px): dropdown to select host graph type (Karate Club, Random Erdős-Rényi, Barabási-Albert scale-free, Regular lattice). Display the selected graph as a force-directed vis-network rendering with 34 nodes. Show node degree distribution as a small bar chart below the graph.
- Right panel (500 px): bar chart of Z-scores for all 13 non-isomorphic connected subgraphs on 3 and 4 nodes. Bars colored green if Z > 2.0 (motif), red if Z < -2.0 (anti-motif), gray otherwise. X-axis labels are small SVG thumbnails of each subgraph pattern.

**Controls:**
- Dropdown: "Host graph type" (Karate Club | Random | Scale-free | Lattice).
- Slider: "Random ensemble size" (100, 500, 1000 samples). Increasing samples narrows confidence intervals.
- "Recompute" button: regenerates the random ensemble and recomputes Z-scores.

**Interaction:**
- Hovering a bar shows a tooltip with the subgraph thumbnail, raw frequency in real graph, mean and std of random ensemble, and Z-score.
- Clicking a bar highlights all instances of that subgraph pattern in the left panel (nodes turn gold, edges turn red).

**Data:** Precompute and hardcode frequencies for the Karate Club graph. Simulate random ensembles using degree-sequence-preserving random rewiring (configuration model). Approximate Z-scores for other graph types using the same rewiring procedure.

**Visual style:** White background, indigo bars default, green/red for significance thresholds. Font: sans-serif, 12px.
</details>
