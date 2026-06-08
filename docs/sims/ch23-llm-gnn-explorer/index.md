---
title: LLM+GNN Pipeline Explorer (Full Version)
description: Interactive p5.js MicroSim for llm+gnn pipeline explorer (full version).
image: /sims/ch23-llm-gnn-explorer/ch23-llm-gnn-explorer.png
og:image: /sims/ch23-llm-gnn-explorer/ch23-llm-gnn-explorer.png
twitter:image: /sims/ch23-llm-gnn-explorer/ch23-llm-gnn-explorer.png
social:
   cards: false
quality_score: 0
---

# LLM+GNN Pipeline Explorer (Full Version)

<iframe src="main.html" height="562" width="100%" scrolling="no"></iframe>

[Run the LLM+GNN Pipeline Explorer (Full Version) MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Text-attributed graphs (TAGs) combine node text descriptions with graph structure. There are three ways to integrate LLMs and GNNs: (1) LLM-as-encoder: the LLM produces fixed text embeddings that feed into a GNN; (2) LLM-as-reasoner: the GNN produces structural embeddings that feed into an LLM prompt; (3) joint training: LLM and GNN are fine-tuned together end-to-end.

This MicroSim shows a small TAG (academic citation network). The node panel on the right shows the paper's text. Switch modes to trace how information flows differently, and compare final prediction accuracy for each integration strategy.

**Learning objective (Bloom's Evaluate (Level 5)):** Trace text-attributed features through the LLM+GNN pipeline in three modes — LLM-as-encoder, LLM-as-reasoner, and joint training — and evaluate what each contributes.

## How to Use

1. **Select a node** — click any paper node to view its title and abstract.
2. **Choose a mode** — click "Encoder Mode", "Reasoner Mode", or "Joint Mode" to switch pipelines.
3. **Step through** — click "Step" to animate each stage of the selected pipeline.
4. **Read accuracy** — the prediction panel shows the classification result and confidence.
5. **Compare** — switch between modes after processing the same node to compare outputs.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch23-llm-gnn-explorer/main.html"
        height="562"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
20–30 minutes

### Prerequisites
GNN basics (Chapter 6). LLM embeddings and prompting at a high level. Text-attributed graphs.

### Activities

1. In Encoder Mode, observe that the LLM produces a fixed embedding regardless of graph context. In Joint Mode, the GNN gradient flows back into the LLM. What is the key advantage of joint training?
2. Switch to Reasoner Mode. The prompt injected into the LLM includes neighbors' abstracts. How does this change the LLM's classification behavior?
3. Identify a case where Encoder Mode succeeds but Reasoner Mode fails (or vice versa). What does this reveal about the task?

### Assessment Question
Compare the three LLM+GNN integration modes along three axes: computational cost at inference, ability to leverage graph structure, and ability to fine-tune the text representation. Produce a table.

## References

1. He et al. (2024). G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding. NeurIPS 2024.
2. Chen et al. (2024). Label-free Node Classification on Graphs with Large Language Models (TAPE). ICLR 2024.

---
*Part of Chapter 23: LLMs and GNNs: Text-Attributed Graphs and Joint Training. Return to the [chapter page](../../chapters/23-llm-gnn/index.md) or browse all [MicroSims](../index.md).*