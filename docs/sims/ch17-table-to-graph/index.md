---
title: Relational Schema to Heterogeneous Graph
description: Interactive p5.js MicroSim for relational schema to heterogeneous graph.
image: /sims/ch17-table-to-graph/ch17-table-to-graph.png
og:image: /sims/ch17-table-to-graph/ch17-table-to-graph.png
twitter:image: /sims/ch17-table-to-graph/ch17-table-to-graph.png
social:
   cards: false
quality_score: 0
---

# Relational Schema to Heterogeneous Graph

<iframe src="main.html" height="562" width="100%" scrolling="no"></iframe>

[Run the Relational Schema to Heterogeneous Graph MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Relational databases store entities in tables and relationships as foreign keys. Relational Deep Learning (RDL) converts this relational structure directly into a heterogeneous graph: each table row becomes a node of the table's type, and each foreign-key link becomes a typed edge.

This MicroSim shows three tables side by side (Customers, Orders, Products). Click any row to highlight the corresponding node in the graph panel on the right and draw its edges. Hovering over an edge shows which columns produced it.

**Learning objective (Bloom's Understand (Level 2)):** See how foreign keys in a relational database (Customers, Orders, Products) map to typed nodes and typed edges in a heterogeneous graph, making the schema-to-graph transformation concrete.

## How to Use

1. **Read the tables** — Customers, Orders, and Products tables are shown at the top.
2. **Click a table row** — the corresponding node in the graph lights up and its edges are highlighted.
3. **Click a graph node** — the corresponding table row lights up.
4. **Hover an edge** — see which foreign-key column created this edge and what information flows along it.
5. **Schema key** — the color-coded key on the right maps each table to its node color.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch17-table-to-graph/main.html"
        height="562"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
10–15 minutes

### Prerequisites
Relational database basics (primary key, foreign key). Heterogeneous graphs (Chapter 15).

### Activities

1. Identify all edges in the graph that correspond to the "customer_id" foreign key in the Orders table.
2. Add a new row to the Orders table (mentally). How many new nodes and edges would appear in the graph?
3. The Product→Order edge represents "appears in". What does a product node's neighborhood encode about buying patterns?

### Assessment Question
Define the schema graph for the Customers–Orders–Products database. For each table, state its node type and the features that would be encoded on each node. List all edge types and their directions.

## References

1. Fey et al. (2023). Relational Deep Learning: Graph Representation Learning on Relational Databases. NeurIPS 2024.
2. Cvitkovic (2020). Supervised Learning on Relational Databases with Graph Neural Networks.

---
*Part of Chapter 17: Relational Deep Learning. Return to the [chapter page](../../chapters/17-relational-deep-learning/index.md) or browse all [MicroSims](../index.md).*