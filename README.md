# Graph Neural Networks

[![MkDocs](https://img.shields.io/badge/Made%20with-MkDocs-526CFE?logo=materialformkdocs)](https://www.mkdocs.org/)
[![Material for MkDocs](https://img.shields.io/badge/Material%20for%20MkDocs-526CFE?logo=materialformkdocs)](https://squidfunk.github.io/mkdocs-material/)
[![GitHub Pages](https://img.shields.io/badge/View%20on-GitHub%20Pages-blue?logo=github)](https://AnvithPothula.github.io/graph-neural-networks-textbook/)
[![Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-DA7857?logo=anthropic)](https://claude.ai/code)
[![Claude Skills](https://img.shields.io/badge/Uses-Claude%20Skills-DA7857?logo=anthropic)](https://github.com/dmccreary/claude-skills)
[![p5.js](https://img.shields.io/badge/p5.js-ED225D?logo=p5.js&logoColor=white)](https://p5js.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## View the Live Site

Visit the interactive textbook at: [https://AnvithPothula.github.io/graph-neural-networks-textbook/](https://AnvithPothula.github.io/graph-neural-networks-textbook/)

## Overview

This is a comprehensive, interactive intelligent textbook on Graph Neural Networks designed for graduate and advanced undergraduate students in machine learning, data mining, and network science. Built with MkDocs Material, it covers the full arc of the field — from graph theory and classical graph algorithms through modern architectures including graph transformers, knowledge graph models, and LLM+GNN integration.

Every chapter leads with a real-world motivating example, builds intuition before equations, provides full mathematical derivations, and includes complete runnable PyTorch Geometric code. 37 interactive MicroSims — p5.js simulations embedded directly in the browser — let you manipulate GNN internals hands-on: watch message passing animate across layers, tune attention weights in real time, and step through WL color refinement iteration by iteration.

The textbook is structured around a dependency-aware learning graph of 300 concepts, ensuring prerequisite sequencing is respected across all 27 chapters. Quizzes, glossary terms, and FAQs cover every major concept, with exercises distributed across all six levels of Bloom's Taxonomy.

## Site Status and Metrics

| Metric | Count |
|--------|-------|
| Chapters | 27 |
| Concepts in Learning Graph | 300 |
| Glossary Terms | 300 |
| FAQ Questions | 82 |
| Quiz Questions | 270 |
| Interactive MicroSims | 37 |
| Diagrams & Causal Loop Diagrams | 40 |
| LaTeX Equations | 746 |
| Total Words | 301,873 |
| Equivalent Pages | 1,235 |

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Clone the Repository

```bash
git clone https://github.com/AnvithPothula/graph-neural-networks-textbook.git
cd graph-neural-networks-textbook
```

### Install Dependencies

```bash
pip install mkdocs mkdocs-material mkdocs-glightbox
```

### Serve Locally

```bash
mkdocs serve
```

Open your browser to `http://localhost:8000/graph-neural-networks-textbook/`

### Build Static Site

```bash
mkdocs build
```

### Deploy to GitHub Pages

```bash
mkdocs gh-deploy
```

## Repository Structure

```
graph-neural-networks-textbook/
├── docs/                              # MkDocs source content
│   ├── chapters/                      # 27 chapter directories (00–26)
│   │   ├── 00-math-prerequisites/
│   │   │   ├── index.md               # Chapter content (~10,000+ words)
│   │   │   ├── quiz.md                # 10 multiple-choice questions
│   │   │   └── references.md          # 10 curated references
│   │   └── ...                        # Chapters 01–26
│   ├── sims/                          # 37 interactive p5.js MicroSims
│   │   ├── ch06-gcn-message-passing/
│   │   │   ├── main.html              # Standalone simulation
│   │   │   ├── index.md               # Embedded doc page
│   │   │   ├── sketch.js              # p5.js source
│   │   │   └── metadata.json          # Bloom's level, learning objective
│   │   └── ...
│   ├── learning-graph/                # Concept graph data and reports
│   │   ├── learning-graph.csv         # 300-concept dependency DAG
│   │   ├── learning-graph.json        # vis-network format
│   │   ├── concept-list.md
│   │   ├── book-metrics.md            # Aggregated textbook metrics
│   │   ├── diagram-table.md           # All diagrams and MicroSims
│   │   └── quality-metrics.md
│   ├── img/                           # Images and mascot assets
│   │   └── mascot/                    # Sage the Graph Node (7 poses)
│   ├── css/                           # Custom styles
│   ├── glossary.md                    # 300 ISO-compliant definitions
│   └── faq.md                         # 82 frequently asked questions
├── src/
│   └── diagram-reports/               # Diagram audit scripts
├── scripts/                           # Build and metrics scripts
├── plugins/                           # MkDocs hooks (social meta tags)
├── mkdocs.yml                         # Site configuration and nav
└── README.md                          # This file
```

## Reporting Issues

Found a bug, typo, or have a suggestion? Please open an issue:

[GitHub Issues](https://github.com/AnvithPothula/graph-neural-networks-textbook/issues)

When reporting, please include:
- Description of the problem or suggestion
- Chapter and section where it occurs
- Browser and OS (for MicroSim issues)
- Screenshots if applicable

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**You are free to:**
- Share — copy and redistribute the material
- Adapt — remix, transform, and build upon the material

**Under the following terms:**
- **Attribution** — Give appropriate credit with a link to the original
- **NonCommercial** — No commercial use without permission
- **ShareAlike** — Distribute contributions under the same license

See [docs/license.md](docs/license.md) for full details.

## Acknowledgements

This textbook is built on the shoulders of the open source community:

- **[MkDocs](https://www.mkdocs.org/)** — Static site generator optimized for project documentation
- **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)** — Beautiful, feature-rich theme
- **[p5.js](https://p5js.org/)** — Creative coding library powering all MicroSims
- **[vis-network](https://visjs.org/)** — Network visualization for the learning graph viewer
- **[PyTorch Geometric](https://pyg.org/)** — GNN framework used in all code examples
- **[Open Graph Benchmark](https://ogb.stanford.edu/)** — ogbn-arxiv dataset used throughout Part II+
- **[Claude AI](https://claude.ai)** by Anthropic — AI-assisted content generation and tooling
- **[Claude Skills](https://github.com/dmccreary/claude-skills)** — Skill framework by Dan McCreary
- **[GitHub Pages](https://pages.github.com/)** — Free hosting for open source projects

## Contact

**Anvith Pothula**

- GitHub: [@AnvithPothula](https://github.com/AnvithPothula)
- Email: anvithpothula@gmail.com

Questions, corrections, or collaboration opportunities? Open an issue or reach out directly.
