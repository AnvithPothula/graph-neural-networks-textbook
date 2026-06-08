---
title: GNN Training Dynamics MicroSim
description: Interactive p5.js MicroSim for gnn training dynamics microsim.
image: /sims/ch08-gnn-training-dynamics/ch08-gnn-training-dynamics.png
og:image: /sims/ch08-gnn-training-dynamics/ch08-gnn-training-dynamics.png
twitter:image: /sims/ch08-gnn-training-dynamics/ch08-gnn-training-dynamics.png
social:
   cards: false
quality_score: 0
---

# GNN Training Dynamics MicroSim

<iframe src="main.html" height="532" width="100%" scrolling="no"></iframe>

[Run the GNN Training Dynamics MicroSim MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

This MicroSim simulates GNN training dynamics on a synthetic node classification task as you vary depth (1–8 layers), dropout rate, and whether residual connections are enabled. Each configuration runs a mini-training loop and plots training loss and validation accuracy curves in real time.

The key phenomenon is over-smoothing: stacking too many layers causes node representations to become indistinguishable, collapsing validation accuracy even as training loss decreases. The gap between the two curves is the over-smoothing signature.

**Learning objective (Bloom's Analyze (Level 4)):** See how training loss and validation accuracy diverge as GCN depth increases, making the over-smoothing feedback loop concrete. Observe how residual connections and dropout mitigate the collapse.

## How to Use

1. **Set depth** — drag the layers slider (1–8) to change network depth.
2. **Toggle residuals** — enable or disable residual (skip) connections.
3. **Set dropout** — adjust the dropout rate applied to each layer's output.
4. **Run training** — click "Train" to animate the loss and accuracy curves over 200 epochs.
5. **Compare runs** — previous runs are overlaid in lighter colors so you can compare configurations.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch08-gnn-training-dynamics/main.html"
        height="532"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
20–30 minutes

### Prerequisites
Gradient descent and backpropagation (Chapter 0). GCN message passing (Chapter 6). Concept of over-smoothing.

### Activities

1. Start with 2 layers (no residuals, no dropout) and train to convergence. Record val accuracy. Increment layers to 4, 6, 8. Plot depth vs. val accuracy.
2. At depth 6, enable residual connections. How much does over-smoothing improve?
3. At depth 6, apply dropout rate 0.5. Compare to the residual-connection fix — which helps more on this task?

### Assessment Question
Prove that repeated application of the GCN propagation rule \(D^{-1/2}AD^{-1/2}\) causes node features to converge. What is the limit each feature converges to?

## References

1. Li et al. (2018). Deeper Insights Into Graph Convolutional Networks for Semi-Supervised Classification. AAAI.
2. Chen et al. (2020). Simple and Deep Graph Convolutional Networks. ICML.

---
*Part of Chapter 8: GNN Training, Augmentation, and Practical Tips. Return to the [chapter page](../../chapters/08-gnn-training/index.md) or browse all [MicroSims](../index.md).*