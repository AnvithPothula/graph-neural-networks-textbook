---
title: Agent Tool-Use Graph — Interactive Planner
description: Interactive p5.js MicroSim for agent tool-use graph — interactive planner.
image: /sims/ch25-tool-use-graph/ch25-tool-use-graph.png
og:image: /sims/ch25-tool-use-graph/ch25-tool-use-graph.png
twitter:image: /sims/ch25-tool-use-graph/ch25-tool-use-graph.png
social:
   cards: false
quality_score: 0
---

# Agent Tool-Use Graph — Interactive Planner

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the Agent Tool-Use Graph — Interactive Planner MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

An AI agent decomposes a complex task into sub-tasks represented as nodes in a tool-use graph. Directed edges encode dependencies: tool B depends on tool A means B cannot start until A returns a result. Tasks with no dependency between them can run in parallel, saving wall-clock time.

This MicroSim shows a small tool-use DAG (directed acyclic graph) for a research task (e.g., "Write a report on GNNs"). Click "Execute Plan" to animate the agent running the tools. Parallel tasks are shown executing simultaneously in the same time slice; dependent tasks wait.

**Learning objective (Bloom's Understand (Level 2)):** See how a tool-use graph structures an AI agent's plan: dependency edges determine which tools can run in parallel vs. which must wait for predecessors to complete.

## How to Use

1. **Read the plan** — each node is a tool call (web_search, read_paper, summarize, write_report, etc.); edges show dependencies.
2. **Execute** — click "Execute Plan" to animate the agent running the DAG. Parallel tools light up simultaneously.
3. **Edit plan** — drag to add or remove dependency edges. Watch how the execution schedule changes.
4. **Add a tool** — click "Add Tool Node" to insert a new tool and wire its dependencies.
5. **Read the timeline** — the Gantt chart at the bottom shows wall-clock parallelism.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch25-tool-use-graph/main.html"
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
Directed acyclic graphs. Basic understanding of AI agents and tool calling. Topological sort.

### Activities

1. Remove a dependency edge between two tools. Observe how the Gantt chart changes — do those tools now run in the same time slice?
2. Add a new tool that depends on all existing terminal nodes. Where does it appear in the execution timeline?
3. Identify the critical path (longest chain of dependencies) in the default plan. This determines the minimum possible wall-clock time.

### Assessment Question
Explain how topological sort determines the execution order of a tool-use DAG. Define the critical path and prove that reducing any non-critical edge's duration does not decrease total execution time.

## References

1. Schick et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. NeurIPS.
2. Yao et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR.

---
*Part of Chapter 25: Agents and Graphs. Return to the [chapter page](../../chapters/25-agents-and-graphs/index.md) or browse all [MicroSims](../index.md).*