---
title: LLM+GNN Pipeline — Text-to-Prediction
description: Interactive p5.js MicroSim for llm+gnn pipeline — text-to-prediction.
image: /sims/ch23-llm-gnn-pipeline/ch23-llm-gnn-pipeline.png
og:image: /sims/ch23-llm-gnn-pipeline/ch23-llm-gnn-pipeline.png
twitter:image: /sims/ch23-llm-gnn-pipeline/ch23-llm-gnn-pipeline.png
social:
   cards: false
quality_score: 0
---

# LLM+GNN Pipeline — Text-to-Prediction

<iframe src="main.html" height="502" width="100%" scrolling="no"></iframe>

[Run the LLM+GNN Pipeline — Text-to-Prediction MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

This MicroSim shows the five-stage pipeline that transforms a paper's raw text description into a node classification prediction via an LLM+GNN architecture. Dimension bars at each stage visualize how the representation dimensionality changes: from token IDs (thousands) to LLM embeddings (768-dim) to GNN output (64-dim) to class logits (8-dim).

The pipeline is shown as a sequence of animated blocks. Each stage activates in turn and the representation flows right.

**Learning objective (Bloom's Understand (Level 2)):** Trace a text-described node through the full LLM+GNN pipeline — raw text → tokenize → LLM encode → GNN aggregate → predicted label — with dimension bars at each stage.

## How to Use

1. **Start** — click "Run Pipeline" to start the animation from left to right.
2. **Step** — click "Step" to advance one stage at a time: Text → Tokenize → LLM Encode → GNN Aggregate → Predict.
3. **Hover a stage** — see the intermediate representation shape and a description of what happens at that stage.
4. **Change input** — select a different node from the dropdown to run the pipeline on different text.
5. **Read prediction** — the final stage shows the predicted class with softmax probability.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch23-llm-gnn-pipeline/main.html"
        height="502"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
10–15 minutes

### Prerequisites
Tokenization and embeddings (basic LLM knowledge). GNN aggregation (Chapter 6). Softmax (Chapter 0).

### Activities

1. Which stage reduces dimensionality the most? Which produces the smallest representation?
2. If the LLM embedding is frozen (not fine-tuned), what are the only trainable parameters in the pipeline?
3. Run two papers from different subject areas. After GNN aggregation, do their representations converge or diverge compared to the raw LLM embeddings?

### Assessment Question
For each of the five stages (tokenize, LLM encode, GNN aggregate, readout, classify), state the input and output tensor shapes, assuming a BERT-base encoder, 2-layer GCN with 64 hidden units, and 8 output classes.

## References

1. Zhao et al. (2022). Learning on Large-scale Text-attributed Graphs via Variational Inference. ICLR 2023.
2. He et al. (2023). HARNESSING EXPLANATIONS: LLM-TO-LM INTERPRETER FOR ENHANCED TEXT-ATTRIBUTED GRAPH REPRESENTATION LEARNING (TAPE). ICLR 2024.

---
*Part of Chapter 23: LLMs and GNNs: Text-Attributed Graphs and Joint Training. Return to the [chapter page](../../chapters/23-llm-gnn/index.md) or browse all [MicroSims](../index.md).*