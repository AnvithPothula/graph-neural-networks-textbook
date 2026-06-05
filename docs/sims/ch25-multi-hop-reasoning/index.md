---
title: Multi-Hop KG Reasoning Agent
description: Interactive p5.js MicroSim for multi-hop kg reasoning agent.
image: /sims/ch25-multi-hop-reasoning/ch25-multi-hop-reasoning.png
og:image: /sims/ch25-multi-hop-reasoning/ch25-multi-hop-reasoning.png
twitter:image: /sims/ch25-multi-hop-reasoning/ch25-multi-hop-reasoning.png
social:
   cards: false
quality_score: 0
---

# Multi-Hop KG Reasoning Agent

<iframe src="main.html" height="562" width="100%" scrolling="no"></iframe>

[Run the Multi-Hop KG Reasoning Agent MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

A graph-structured reasoning agent answers multi-hop questions by iteratively retrieving subgraphs, reasoning about which relation to follow next, and updating a beam of candidate paths. Each hop the agent scores candidate next-hop entities using the current query representation and prunes low-scoring branches.

This MicroSim shows a toy KG with a question at the top. The agent's beam is visualized as a set of colored paths growing outward. At each step, paths with low relation-relevance scores are pruned (shown in gray) and only top-k paths survive.

**Learning objective (Bloom's Apply (Level 3)):** Trace an agent's query→retrieve→reason→refine loop across a knowledge graph, watching each hop narrow toward the answer by pruning unlikely relation paths.

## How to Use

1. **Read the query** — the natural-language question is shown at the top (e.g., "Which city is in the country where Einstein was born?").
2. **Step** — click "Next Hop" to advance one reasoning step. The beam of candidate paths grows.
3. **Read scores** — each path shows its cumulative relation-relevance score.
4. **Prune** — after each hop, low-scoring paths are grayed out.
5. **Answer** — after the final hop, the entity with the highest beam score is the predicted answer.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch25-multi-hop-reasoning/main.html"
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
Knowledge graph basics (Chapter 12). Multi-hop queries (Chapter 13). Beam search concept.

### Activities

1. After hop 1, how many paths are in the beam? After pruning, how many survive? At what score threshold were paths pruned?
2. Trace the correct answer path from start to end. At which hop does the correct path first emerge as the top-scored candidate?
3. Change the question to a 3-hop query. How does the beam size scale? What are the accuracy implications of a smaller beam?

### Assessment Question
Describe the ReasonPath / MINERVA agent architecture for multi-hop KG reasoning. Define the policy network, the state space (entity + query embedding), and the action space (outgoing relation edges).

## References

1. Das et al. (2018). Go for a Walk and Arrive at the Answer: Reasoning over Paths in Knowledge Bases (MINERVA). ICLR.
2. Xiong et al. (2017). DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning. EMNLP.

---
*Part of Chapter 25: Agents and Graphs. Return to the [chapter page](../../chapters/25-agents-and-graphs/) or browse all [MicroSims](../index.md).*