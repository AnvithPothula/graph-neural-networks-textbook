---
title: PageRank Power Iteration Simulator
description: Interactive p5.js MicroSim for pagerank power iteration simulator.
image: /sims/ch03-pagerank-power-iteration/ch03-pagerank-power-iteration.png
og:image: /sims/ch03-pagerank-power-iteration/ch03-pagerank-power-iteration.png
twitter:image: /sims/ch03-pagerank-power-iteration/ch03-pagerank-power-iteration.png
social:
   cards: false
quality_score: 0
---

# PageRank Power Iteration Simulator

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the PageRank Power Iteration Simulator MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

This MicroSim animates PageRank's power iteration algorithm on a small directed graph. Node size encodes current rank — big nodes are important. At each iteration, rank flows from nodes to their out-neighbors, with a fraction teleporting uniformly to any node.

A live convergence chart at the bottom shows how quickly the scores stabilize. You can adjust the damping factor and watch how values near 1.0 converge slowly (nearly no teleportation) while low values converge fast but assign flatter scores.

**Learning objective (Bloom's Apply (Level 3)):** Watch power iteration propagate rank across a web graph, see why well-connected nodes accumulate rank, and observe how the damping factor controls convergence speed and handles spider traps and dead ends.

## How to Use

1. **Step** — click "Next Iteration" to run one power-iteration step. Node sizes rescale to reflect updated scores.
2. **Auto-run** — click "Run" to animate iterations automatically until convergence.
3. **Damping slider** — drag to change the damping factor \(d\) (default 0.85). Higher \(d\) = more weight on edges; lower \(d\) = more teleportation.
4. **Graph presets** — choose a star, chain, or random web graph to compare convergence behavior.
5. **Hover a node** — see its current PageRank score and incoming/outgoing edge weights.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch03-pagerank-power-iteration/main.html"
        height="522"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
15–20 minutes

### Prerequisites
Matrix-vector multiplication (Chapter 0). Understanding of directed graphs and stationary distributions.

### Activities

1. Start with the star graph. Observe that the central hub accumulates rank because many nodes point to it. How does this change if you flip all the edges?
2. Create a "spider trap" (a clique with no outgoing edges). Run without teleportation (\(d=1\)). Watch rank accumulate and never escape. Then lower \(d\) to 0.85 and see the fix.
3. Find the damping factor \(d\) at which convergence takes fewer than 10 iterations on the default graph.

### Assessment Question
Explain in matrix form why PageRank converges. What property of the transition matrix guarantees a unique stationary distribution? What breaks without teleportation?

## References

1. Page et al. (1999). The PageRank Citation Ranking: Bringing Order to the Web. Technical Report, Stanford.
2. Langville & Meyer (2006). Google's PageRank and Beyond. Princeton University Press.

---
*Part of Chapter 3: Link Analysis and PageRank. Return to the [chapter page](../../chapters/03-link-analysis-pagerank/) or browse all [MicroSims](../index.md).*