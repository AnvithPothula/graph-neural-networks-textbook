---
title: MicroSims
description: Interactive p5.js simulations — one per chapter — for Graph Neural Networks and Machine Learning with Graphs.
---

# MicroSims

MicroSims are small, self-contained interactive simulations, one for each chapter. Each focuses on a single concept that is difficult to convey with static diagrams. They run directly in your browser via p5.js with no installation required.

## All MicroSims

| # | Chapter | Title | Concept | Status |
|---|---------|-------|---------|--------|
| 0 | Ch 0 | Matrix × Graph Explorer | Visualize **Ax**, **D⁻¹Ax**, and **D⁻½ÂD⁻½x** on a small graph; see why symmetric normalization matters | Built |
| 1 | Ch 1 | Graph Property Explorer | Add/remove nodes and edges; watch degree distribution, clustering coefficient, and bridge detection update live | Planned |
| 2 | Ch 2 | WL Color Refinement | Step through Weisfeiler-Lehman iterations; compare two graphs side-by-side; see when/if they become distinguishable | Planned |
| 3 | Ch 3 | Biased Random Walk | Click a start node; adjust p/q sliders; watch the walk trace DFS-like vs. BFS-like paths; overlay heat maps | Planned |
| 4 | Ch 4 | PageRank Power Iteration | Node size encodes rank; watch scores converge over iterations; damping-factor slider | Planned |
| 5 | Ch 5 | Label Propagation | Click to seed labels; step through rounds; watch labels diffuse across the graph | Planned |
| 6 | Ch 6 | GCN Message Passing | Click a node; animated arrows show messages aggregating from neighbors across 1/2/3 layers | Planned |
| 7 | Ch 7 | GAT Attention Weights | Edge thickness encodes attention score; hover a node to see softmax scores; head-count selector | Planned |
| 8 | Ch 8 | GNN Training Dynamics | Live loss/accuracy curves as epochs advance; toggle train/val/test splits; learning-rate slider | Planned |
| 9 | Ch 9 | Graph Isomorphism Tester | Draw two graphs; WL test reports distinguishable/indistinguishable; shows canonical failure cases | Planned |
| 10 | Ch 10 | GIN vs. GCN Expressiveness | Load 3-regular graph pair; show GCN assigns identical embeddings, GIN distinguishes them | Planned |
| 11 | Ch 11 | Graph Transformer Attention | Attention heatmap overlay on graph; query/key/value visualization per node | Planned |
| 12 | Ch 12 | TransE Geometry | Drag head/relation/tail vectors in 2D; valid triples highlighted when h + r ≈ t | Planned |
| 13 | Ch 13 | Multi-Hop KG Reasoning | Click an entity; watch the reasoning chain extend hop-by-hop with confidence scores | Planned |
| 14 | Ch 14 | KG Embedding Space | 2D projection of entity/relation vectors; click an entity to see its nearest neighbors; zero-shot triple scoring slider | Planned |
| 15 | Ch 15 | Heterogeneous Graph Explorer | Toggle node/edge types on/off; show metapath examples; type-specific attention weights | Planned |
| 16 | Ch 16 | Matrix Factorization for Recommendation | Interactive user-item matrix; drag latent factors; see predicted rating update in real time | Planned |
| 17 | Ch 17 | Table-to-Graph Converter | Input three relational tables (Users, Products, Purchases); watch them transform into a heterogeneous graph | Planned |
| 18 | Ch 18 | Community Detection (Louvain) | Step through Louvain merges; modularity score updates each step | Planned |
| 19 | Ch 19 | Subgraph Pattern Finder | Draw a query subgraph; all isomorphic instances in the host graph highlight; frequency count updates live | Planned |
| 20 | Ch 20 | Neighborhood Sampling | Click a center node; depth/fan-out sliders; see the sampled subgraph highlight and grow | Planned |
| 21 | Ch 21 | GraphRNN Generation | Watch a graph grow node by node; adjacency matrix builds alongside | Planned |
| 22 | Ch 22 | Temporal Graph Evolution | Timeline scrubber; nodes/edges appear and expire; temporal neighborhood shown | Planned |
| 23 | Ch 23 | GraphVAE Latent Space | 2D scatter of graph embeddings; click a point to highlight its graph neighborhood | Planned |
| 24 | Ch 24 | LLM + GNN Pipeline | Step through: raw text → LLM embedding → GNN aggregation → final representation; dimension bars visualize each stage | Planned |
| 25 | Ch 25 | Graph Reasoning Agent | Watch an agent traverse a KG to answer a multi-hop question; shows confidence propagation at each hop | Planned |
| 26 | Ch 26 | GNN Architecture Family Tree | Interactive taxonomy of GNN models; click any node to see key properties; edges show "extends/inspired by" | Planned |

## How MicroSims Are Built

Each MicroSim lives in `sims/<slug>/` with three files:

- `main.html` — self-contained simulation (p5.js CDN, no build step)
- `index.md` — description page with embedded iframe, usage notes, and key insight
- `local.css` *(optional)* — sim-specific styles

Embed in a chapter with:

```html
<iframe src="../../sims/<slug>/main.html" width="100%" height="480px" frameborder="0"></iframe>
```

## Status Key

| Status | Meaning |
|--------|---------|
| **Built** | Simulation is implemented and embedded in the chapter |
| **Planned** | Specification complete; implementation pending |
