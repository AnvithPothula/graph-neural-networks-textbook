---
title: Girvan-Newman Community Detection on Karate Club
description: Interactive p5.js MicroSim for girvan-newman community detection on karate club.
image: /sims/ch18-girvan-newman/ch18-girvan-newman.png
og:image: /sims/ch18-girvan-newman/ch18-girvan-newman.png
twitter:image: /sims/ch18-girvan-newman/ch18-girvan-newman.png
social:
   cards: false
quality_score: 0
---

# Girvan-Newman Community Detection on Karate Club

<iframe src="main.html" height="512" width="100%" scrolling="no"></iframe>

[Run the Girvan-Newman Community Detection on Karate Club MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The Girvan-Newman algorithm identifies community structure by iteratively removing the edge with the highest betweenness centrality — the edge that the most shortest paths pass through. Removing bridge edges splits the graph into increasingly separated communities.

This MicroSim shows the Karate Club graph. At each step, the edge with highest betweenness is highlighted in red before removal. Modularity \(Q\) is plotted after each removal, and its maximum marks the best partition found.

**Learning objective (Bloom's Understand (Level 2)):** Watch the Girvan-Newman algorithm progressively expose community structure by removing the highest-betweenness edge at each step, with live modularity tracking.

## How to Use

1. **Step** — click "Remove Highest Betweenness Edge" to remove one edge and recompute betweenness.
2. **Auto-run** — click "Run to Best \(Q\)" to advance automatically until modularity peaks.
3. **Read \(Q\)** — the modularity chart on the right updates after each removal.
4. **Hover an edge** — see its current betweenness centrality value.
5. **View partition** — community assignments are shown as node colors once Q peaks.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch18-girvan-newman/main.html"
        height="512"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
10–15 minutes

### Prerequisites
Betweenness centrality (Chapter 1). Modularity definition. Connected components.

### Activities

1. Before running the algorithm, identify by eye which edges you think are bridges. Compare to what Girvan-Newman removes first.
2. Run to the \(Q\) maximum. How many communities does the algorithm find? Compare to the known Karate Club ground truth (2 factions).
3. Continue removing edges past the \(Q\) peak. What happens to the communities?

### Assessment Question
Explain why betweenness centrality identifies bridge edges. Derive the computational complexity of Girvan-Newman as O(m²n) where m is the number of edges and n the number of nodes.

## References

1. Girvan & Newman (2002). Community structure in social and biological networks. PNAS.
2. Newman & Girvan (2004). Finding and evaluating community structure in networks. PRE.

---
*Part of Chapter 18: Community Structure in Networks. Return to the [chapter page](../../chapters/18-community-structure/) or browse all [MicroSims](../index.md).*