---
title: Drug Discovery GNN Pipeline
description: Interactive p5.js MicroSim for drug discovery gnn pipeline.
image: /sims/ch21-drug-discovery-pipeline/ch21-drug-discovery-pipeline.png
og:image: /sims/ch21-drug-discovery-pipeline/ch21-drug-discovery-pipeline.png
twitter:image: /sims/ch21-drug-discovery-pipeline/ch21-drug-discovery-pipeline.png
social:
   cards: false
quality_score: 0
---

# Drug Discovery GNN Pipeline

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the Drug Discovery GNN Pipeline MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Modern drug discovery can be structured as three sequential GNN tasks. Stage 1 (Target ID): a GNN performs node classification on a protein-protein interaction (PPI) network to identify high-centrality disease-target proteins. Stage 2 (Hit Generation): a generative GNN like GCPN or DiGress generates candidate molecules conditioned on the identified target binding pocket. Stage 3 (Safety Screening): a GNN link predictor on a drug-drug interaction (DDI) network flags dangerous interactions before clinical testing.

Step through the pipeline to see each stage's visualization and hover elements for more detail.

**Learning objective (Bloom's Understand (Level 2)):** Trace how GNNs contribute across a three-stage drug-discovery workflow: target identification via node classification on a PPI network, candidate generation via GCPN/DiGress, and safety screening via link prediction on a DDI network.

## How to Use

1. **Step through** — click "Step Through Pipeline" to advance through Stages 1, 2, and 3.
2. **Reset** — returns to the pre-pipeline view.
3. **Hover nodes** — in Stage 1, hover red nodes to see disease-pathway probability; in Stage 3, hover red edges to see predicted interaction confidence.
4. **Hover arrows** — click the connectors between stages to see what information passes downstream.
5. **Stage panel** — the text in each stage describes the GNN task, model, and output.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch21-drug-discovery-pipeline/main.html"
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
Node classification, link prediction, graph generation concepts (Chapters 6–8). Molecular graphs (atoms = nodes, bonds = edges).

### Activities

1. In Stage 1, which nodes are identified as disease targets (red)? What makes betweenness centrality a useful feature for target identification?
2. In Stage 2, hover the benzene ring. What property metrics (QED, SA, IC50) are shown? Why are these key molecular optimization objectives?
3. In Stage 3, hover the red edge involving the candidate molecule. What interaction is predicted and with what confidence?

### Assessment Question
For each of the three pipeline stages, identify: (1) the graph input, (2) the GNN task type (node/edge/graph-level), (3) the model architecture used, and (4) what is passed to the next stage.

## References

1. You et al. (2018). Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation. NeurIPS.
2. Vignac et al. (2023). DiGress: Discrete Denoising diffusion for graph generation. ICLR.

---
*Part of Chapter 21: Deep Generative Models for Graphs. Return to the [chapter page](../../chapters/21-generative-models/index.md) or browse all [MicroSims](../index.md).*