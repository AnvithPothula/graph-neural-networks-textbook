---
title: Louvain Two-Phase Community Detection Explorer
description: Interactive p5.js MicroSim for louvain two-phase community detection explorer.
image: /sims/ch18-louvain-explorer/ch18-louvain-explorer.png
og:image: /sims/ch18-louvain-explorer/ch18-louvain-explorer.png
twitter:image: /sims/ch18-louvain-explorer/ch18-louvain-explorer.png
social:
   cards: false
quality_score: 0
---

# Louvain Two-Phase Community Detection Explorer

<iframe src="main.html" height="502" width="100%" scrolling="no"></iframe>

[Run the Louvain Two-Phase Community Detection Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The Louvain algorithm finds communities by alternating two phases: in Phase 1, each node greedily moves to the neighboring community that maximizes the modularity gain. In Phase 2, each community is compressed into a single super-node, and Phase 1 repeats on the coarsened graph. This hierarchical coarsening finds a nested community structure.

This MicroSim runs both phases on the Karate Club graph. A modularity bar grows with each accepted move. After Phase 1 converges, the graph collapses to super-nodes for Phase 2.

**Learning objective (Bloom's Analyze (Level 4)):** Trace how local node moves (Phase 1) aggregate into global communities and how compression into super-nodes (Phase 2) enables the algorithm to scale, with live modularity \(Q\) tracking.

## How to Use

1. **Phase 1 — node moves** — click "Node Move" to greedily move one node to the community giving the highest \(\Delta Q\) gain.
2. **Run Phase 1** — click "Run Phase 1" to animate all greedy moves until no improvement remains.
3. **Phase 2 — compress** — click "Compress" to collapse communities to super-nodes.
4. **Read Q** — the modularity panel updates after every accepted move.
5. **Inspect** — hover any node or super-node to see its current community assignment and \(\Delta Q\) for each potential move.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch18-louvain-explorer/main.html"
        height="502"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
20–30 minutes

### Prerequisites
Modularity (Chapter 18 intro). Community structure. Girvan-Newman (same chapter).

### Activities

1. In Phase 1, identify which node move produces the largest single modularity gain.
2. After Phase 2 compression, how many super-nodes remain? How does the super-node graph compare to the original?
3. Compare the Louvain partition to the Girvan-Newman partition on the same graph. Do they agree on community membership?

### Assessment Question
Derive the modularity gain formula \(\Delta Q\) for moving node \(i\) from community \(A\) to community \(B\). Explain why the Louvain algorithm is considered greedy and what local optima it might fall into.

## References

1. Blondel et al. (2008). Fast unfolding of communities in large networks. J. Stat. Mech.
2. Lancichinetti & Fortunato (2009). Community detection algorithms: A comparative analysis. PRE.

---
*Part of Chapter 18: Community Structure in Networks. Return to the [chapter page](../../chapters/18-community-structure/index.md) or browse all [MicroSims](../index.md).*
