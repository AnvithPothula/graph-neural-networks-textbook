---
title: Label Propagation Step-by-Step Simulator
description: Interactive p5.js MicroSim for label propagation step-by-step simulator.
image: /sims/ch05-label-propagation-stepper/ch05-label-propagation-stepper.png
og:image: /sims/ch05-label-propagation-stepper/ch05-label-propagation-stepper.png
twitter:image: /sims/ch05-label-propagation-stepper/ch05-label-propagation-stepper.png
social:
   cards: false
quality_score: 0
---

# Label Propagation Step-by-Step Simulator

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the Label Propagation Step-by-Step Simulator MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Label propagation is a semi-supervised algorithm that assigns each unlabeled node a soft label distribution equal to the weighted average of its neighbors' distributions. It converges to the harmonic solution, where each unlabeled node minimizes the difference from its neighbors.

This MicroSim seeds two labeled factions (the two Karate Club communities) and steps through the propagation iteration by iteration. Nodes are colored as a mixture of the two label colors, making the decision boundary and convergence pattern visible.

**Learning objective (Bloom's Understand (Level 2)):** Watch label information flow outward from two seed nodes through the Karate Club graph, building intuition for the harmonic-averaging property and eventual convergence of label propagation.

## How to Use

1. **Select seeds** — the two faction leaders are pre-seeded with opposite labels (blue vs. red). Click any node to lock its label.
2. **Step** — click "Next Step" to run one propagation round. All unlabeled nodes update simultaneously.
3. **Run to convergence** — click "Run" to animate until the labels stabilize.
4. **Reset** — removes all labels except the original two seeds.
5. **Hover a node** — see its current soft-label distribution as a percentage.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch05-label-propagation-stepper/main.html"
        height="522"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
10–15 minutes

### Prerequisites
Graph basics (Chapter 1). Understanding of weighted average. Helpful: eigenvectors and harmonic functions.

### Activities

1. Step through propagation manually for the first three rounds. Verify that each unlabeled node's new label is the average of its current neighbors' labels.
2. Add a third seed with a neutral (50/50) label. Observe how it changes the final label assignment of nearby nodes.
3. Identify the nodes closest to the decision boundary (\(\approx 50/50\)). Are they the bridge nodes between the two Karate Club factions?

### Assessment Question
Prove that label propagation converges on connected graphs and the fixed point satisfies the harmonic equation \(\Delta f = 0\) on unlabeled nodes.

## References

1. Zhu et al. (2003). Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions. ICML.
2. Zhou et al. (2004). Learning with Local and Global Consistency. NeurIPS.

---
*Part of Chapter 5: Label Propagation and Semi-Supervised Learning. Return to the [chapter page](../../chapters/05-label-propagation/) or browse all [MicroSims](../index.md).*