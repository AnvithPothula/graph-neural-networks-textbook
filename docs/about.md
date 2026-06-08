---
title: "About This Book"
description: "About Graph Neural Networks — its purpose, audience, design, and the author behind it."
keywords:
  - graph neural networks
  - message passing neural networks
  - GCN
  - GAT
  - GIN
  - GraphSAGE
  - graph transformers
  - knowledge graphs
  - heterogeneous graphs
  - PyTorch Geometric
  - open educational resource
  - intelligent textbook
---

# About This Book

<!--
## Publication Metadata

<div class="pub-meta" markdown>

| | |
|---|---|
| **Resource type** | Open Educational Resource (OER) — Interactive Intelligent Textbook |
| **Subject area** | Computer Science &rsaquo; Artificial Intelligence &rsaquo; Machine Learning on Graphs |
| **ACM CCS** | [Computing Methodologies → Machine Learning → Learning on structured data and graphs](https://dl.acm.org/ccs) |
| **Keywords** | graph neural networks, message passing, GCN, GAT, GIN, GraphSAGE, graph transformers, knowledge graphs, heterogeneous graphs, scalable GNNs, PyTorch Geometric, OER |
| **Author** | Anvith Pothula |
| **ORCID** | *[Register and add your ORCID iD](https://orcid.org/register) for academic discoverability* |
| **Published** | 2026 |
| **Last updated** | May 28, 2026 |
| **Version** | 0.1 |
| **Language** | English |
| **Repository** | [github.com/AnvithPothula/graph-neural-networks-textbook](https://github.com/AnvithPothula/graph-neural-networks-textbook) |
| **License** | [CC BY-NC-SA 4.0](license.md) |
| **DOI** | *Not yet registered — [archive a release on Zenodo](https://zenodo.org/account/settings/github/) to mint a citable DOI. Once issued, add `doi:` to the BibTeX entry below.* |

</div>

---
-->

!!! mascot-welcome "Welcome from Sage!"
    <img src="../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving welcome">
    Hi, I'm Sage — a graph node who learns by aggregating wisdom from my neighbors, just like the GNNs we study together. This textbook started as a question: *what would it look like if a student built the GNN course they wished existed?* Anvith spent months building exactly that. Whether you're starting from zero or deepening existing knowledge, you're in the right place. Let's aggregate some knowledge!

## Why This Intelligent Textbook

Graph-structured data is everywhere — social networks, drug interaction databases, knowledge
bases, supply chains, financial transaction networks, and the molecular graphs underlying modern
drug discovery. Graph Neural Networks have emerged as the definitive tool for reasoning over
this class of data, yet they remain poorly covered in most ML curricula: either treated as a
brief aside in a general deep learning course, or fragmented across research papers with no
unified pedagogical thread.

**Graph ML is already operating at planetary scale:**

- **Pinterest's PinSage**, a GNN-based recommender, generates recommendations from a graph
  of **3 billion nodes** and **18 billion edges** — one of the largest deployed GNN systems in
  production, responsible for driving 40% of home-feed impressions.[^1]
- **AlphaFold 2**, which solved the 50-year protein-folding problem and won the 2024 Nobel
  Prize in Chemistry, uses an attention mechanism over residue-pair graphs to achieve atomic
  accuracy — demonstrating that graph-structured reasoning operates at the frontier of
  scientific discovery.[^2]
- The global **knowledge graph / graph analytics market** was valued at **$2.4 billion in 2023**
  and is projected to grow at a **CAGR of 24.3%** through 2030, driven by adoption in
  healthcare, finance, and enterprise AI.[^3]
- The number of papers referencing "graph neural network" on **arXiv grew from under 100
  in 2017 to over 7,500 in 2023** — a 75× increase in six years — making graph ML one of
  the fastest-growing subfields in machine learning.[^4]
- **Drug discovery pipelines** at AstraZeneca, Pfizer, and over 50 biotech startups now use
  GNNs to predict molecular properties, synthesize novel compounds, and repurpose existing
  drugs — with GNN-assisted compounds entering clinical trials as of 2024.[^5]

These numbers represent a field that has already left the lab and is reshaping industries.
Yet most students finish a general ML course without having written a single line of GNN code
or seen a message-passing derivation from first principles.

This textbook closes that gap. It is built on a **learning graph of 300 interconnected
concepts** organized across 15 taxonomy categories, so prerequisites are always introduced
before they are needed. Every chapter leads with a real-world motivating example, builds
intuition before equations, provides full mathematical derivations, and includes complete
runnable **PyTorch Geometric** code. **37 interactive MicroSims** — p5.js simulations
embedded directly in the browser — let you manipulate GNN internals hands-on. The entire
textbook is **open source and free** — no paywalls, no access codes, no expensive editions.

## Audience

This textbook is designed for:

- **Graduate students** in CS, computational biology, data science, or any field with relational data
- **Advanced undergraduates** who have completed an [ML course](https://anvithpothula.github.io/machine-learning-textbook/) and want to go deeper on graphs
- **Practitioners** building recommendation engines, drug discovery pipelines, fraud detection systems, or knowledge bases
- **Researchers** entering graph ML who want a unified, up-to-date reference

## Prerequisites

### Required

- **Programming:** Python proficiency; NumPy/Pandas familiarity
- **Linear algebra:** Matrix multiplication, eigenvalues/eigenvectors, SVD
- **Probability & statistics:** Probability distributions, expectations, Bayes' theorem
- **ML fundamentals:** Supervised/unsupervised learning, gradient descent, backpropagation

### Helpful but taught in Chapter 0

- PyTorch and automatic differentiation
- Graph theory basics (introduced from scratch)
- PyTorch Geometric (PyG)

## What Makes This Textbook Different

- **Self-contained prerequisites** — Chapter 0 covers linear algebra, probability, PyTorch, and PyTorch Geometric from scratch. No assumed background beyond introductory ML.
- **Intuition before formalism** — every concept is explained in plain language before equations appear.
- **Full derivations** — major results are derived, not just stated.
- **Interactive MicroSims** — 37 p5.js simulations let you manipulate parameters and see GNN behavior in real time.
- **Bloom's-aligned exercises** — 12 exercises per chapter spanning all cognitive levels, from recall to open-ended design.
- **Up-to-date benchmarks** — results from 2024–2025, with links to live leaderboards.
- **Broad coverage** — 27 chapters covering both classical graph algorithms and frontier topics like graph foundation models and LLM+GNN integration.

## How to Use This Book

This textbook is designed for self-paced study. The book includes:

- **27 Chapters** spanning graph theory, classical algorithms, node embeddings, GCN/GAT/GIN,
  graph transformers, knowledge graphs, heterogeneous graphs, scalable GNNs, generative
  models, and LLM+GNN integration
- **37 Interactive MicroSims** embedded in chapters — browser-based p5.js simulations you
  can manipulate to explore message passing, attention weights, WL refinement, PageRank
  convergence, and more
- **270 Quiz Questions** across all chapters, distributed over all 6 Bloom's Taxonomy levels
- **Annotated References** per chapter linking to Wikipedia and authoritative primary sources
- **Glossary** with definitions for all 300 key concepts
- **FAQ** with 82 common questions and answers
- **Learning Graph** visualizing 300 concept dependencies so you can plan a non-linear path

**Navigation tips:**

- **Read Chapter 0 first** if you need a refresher on linear algebra, PyTorch, or PyTorch Geometric
- Read chapters in order — concepts are introduced in dependency order per the learning graph
- Use the search bar (top right) to jump to a specific term
- Try the MicroSims as you encounter them — they are the fastest way to build intuition for a new concept
- Check the [Learning Graph](learning-graph/index.md) when you want to see how a concept fits into the larger picture
- Each chapter's "Further Reading" section points to the key papers — follow those after mastering the chapter content

## About the Author

**Anvith Pothula** is a student and independent developer with a deep interest in machine
learning, graph-structured data, and open education. This textbook grew out of a desire to
build the comprehensive, interactive GNN resource that didn't yet exist — one that matches
the depth of graduate coursework while remaining fully accessible to motivated undergraduates
and self-taught practitioners.

The project was built end-to-end using MkDocs Material, p5.js, PyTorch Geometric, and
Claude AI — demonstrating how a single determined builder can produce graduate-level
educational content at no cost to learners.

**Other projects:**

- [Machine Learning Textbook](https://anvithpothula.github.io/machine-learning-textbook/) — a companion interactive textbook covering the ML fundamentals (supervised learning, neural networks, optimization, evaluation) that this book builds on

**Connect:**

- GitHub: [@AnvithPothula](https://github.com/AnvithPothula)
- LinkedIn: [Anvith Pothula](https://www.linkedin.com/in/anvith-pothula/)

## How to Cite This Book

If you reference this textbook in academic work, curriculum proposals, lesson plans, or
other publications, please use one of the following formats.

**APA (7th edition)**

Pothula, A. (2026). *Graph Neural Networks: An Intelligent Interactive Textbook*.
https://AnvithPothula.github.io/graph-neural-networks-textbook/

**Chicago (17th edition)**

Pothula, Anvith. 2026. *Graph Neural Networks: An Intelligent Interactive Textbook*.
https://AnvithPothula.github.io/graph-neural-networks-textbook/.

**MLA (9th edition)**

Pothula, Anvith. *Graph Neural Networks: An Intelligent Interactive Textbook*. 2026,
AnvithPothula.github.io/graph-neural-networks-textbook/.

**BibTeX**

```bibtex
@book{pothula2026gnn,
  title       = {Graph Neural Networks: An Intelligent Interactive Textbook},
  author      = {Pothula, Anvith},
  year        = {2026},
  version     = {0.1},
  url         = {https://AnvithPothula.github.io/graph-neural-networks-textbook/},
  doi         = {},
  keywords    = {graph neural networks, message passing, GCN, GAT, GIN, GraphSAGE,
                 graph transformers, knowledge graphs, PyTorch Geometric},
  license     = {CC BY-NC-SA 4.0},
  note        = {Open-source interactive intelligent textbook; 27 chapters,
                 37 MicroSims, 300 concepts, 301,873 words}
}
```

To cite a specific chapter, append the chapter title and URL — for example:

Pothula, A. (2026). Chapter 1: Introduction to Graphs and Networks. In
*Graph Neural Networks: An Intelligent Interactive Textbook*.
https://AnvithPothula.github.io/graph-neural-networks-textbook/chapters/01-intro-to-graphs/

## License

This work is released under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License (CC BY-NC-SA 4.0)](license.md). You are free to share and adapt the
material for non-commercial purposes as long as you give appropriate credit
and share your adaptations under the same license.

## References

[^1]: Ying, R., et al. (2018). *Graph Convolutional Neural Networks for Web-Scale Recommender
Systems*. KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219890

[^2]: Jumper, J., et al. (2021). *Highly accurate protein structure prediction with AlphaFold*.
Nature, 596, 583–589. https://www.nature.com/articles/s41586-021-03819-2

[^3]: Grand View Research. (2024). *Knowledge Graph Market Size, Share & Trends Analysis
Report*. https://www.grandviewresearch.com/industry-analysis/knowledge-graph-market-report

[^4]: Hu, W., et al. (2020). *Open Graph Benchmark: Datasets for Machine Learning on Graphs*.
NeurIPS 2020. https://arxiv.org/abs/2005.00687

[^5]: Stokes, J. M., et al. (2020). *A Deep Learning Approach to Antibiotic Discovery*.
Cell, 180(4), 688–702. https://doi.org/10.1016/j.cell.2020.01.021
