---
title: node2vec Biased Random Walk Explorer
description: Interactive p5.js MicroSim for node2vec biased random walk explorer.
image: /sims/ch04-node2vec-walk/ch04-node2vec-walk.png
og:image: /sims/ch04-node2vec-walk/ch04-node2vec-walk.png
twitter:image: /sims/ch04-node2vec-walk/ch04-node2vec-walk.png
social:
   cards: false
quality_score: 0
---

# node2vec Biased Random Walk Explorer

<iframe src="main.html" height="562" width="100%" scrolling="no"></iframe>

[Run the node2vec Biased Random Walk Explorer MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

node2vec extends DeepWalk by introducing a biased random walk that interpolates between two exploration strategies. The return parameter \(p\) controls how likely a walk is to revisit the previous node; the in-out parameter \(q\) controls whether it explores outward (DFS-like, \(q < 1\)) or stays local (BFS-like, \(q > 1\)).

This MicroSim runs live random walks on the Karate Club graph. The walk path is drawn as an animated arrow trail. The heat overlay on nodes darkens as they are visited, making the coverage pattern visual. BFS-heavy walks cluster tightly around the start node; DFS-heavy walks roam across the whole graph.

**Learning objective (Bloom's Apply (Level 3)):** Manipulate the return parameter \(p\) and in-out parameter \(q\) and watch walks shift between BFS-like (local community) and DFS-like (structural role) exploration on the Karate Club graph.

## How to Use

1. **Pick a start node** — click any node on the Karate Club graph to anchor the walk there.
2. **Set p and q** — use the sliders to change the return and in-out parameters.
3. **Run walk** — click "Start Walk" to animate a walk of the configured length.
4. **Compare coverage** — after the walk, the heat map shows which nodes were visited. Toggle between BFS (\(q \gg 1\)), neutral (\(p=q=1\)), and DFS (\(q \ll 1\)) modes.
5. **Reset heat** — click "Clear Heat" to reset visit counts.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch04-node2vec-walk/main.html"
        height="562"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
15–20 minutes

### Prerequisites
Random walks (Chapter 3). Understanding of graph neighborhoods and the concept of structural vs. community similarity.

### Activities

1. Set \(p=1\), \(q=0.1\) (DFS mode). Start from a peripheral node. Does the walk stay near its starting community or wander? Repeat with \(q=10\) (BFS mode).
2. Start five walks from the same hub node with different \((p,q)\) settings. Compare the resulting heat maps.
3. Explain why DFS walks are better for capturing homophily-based similarity while BFS walks capture structural equivalence.

### Assessment Question
The node2vec paper defines the transition probabilities \(\alpha(t, x)\) for moving to node \(x\) given current node \(v\) and previous node \(t\). Write the formula and explain each of the three cases (\(x = t\), \(d(x,t) = 1\), \(d(x,t) = 2\)).

## References

1. Grover & Leskovec (2016). node2vec: Scalable Feature Learning for Networks. KDD.
2. Perozzi et al. (2014). DeepWalk: Online Learning of Social Representations. KDD.

---
*Part of Chapter 4: Node Embeddings: DeepWalk and node2vec. Return to the [chapter page](../../chapters/04-node-embeddings/index.md) or browse all [MicroSims](../index.md).*