# Learning Graph

The learning graph captures how concepts in the Graph Neural Networks textbook depend on each other. It is a Directed Acyclic Graph (DAG) — every concept has a path back to one or more foundational concepts that have no prerequisites.

The graph covers **282 concepts** across **15 taxonomy categories**:

| Category | Abbreviation | Examples |
|---|---|---|
| Prerequisites | PREREQ | Matrix multiplication, Gradient descent, PyTorch tensor |
| Graph Fundamentals | FOUND | Graph, Node, Edge, Adjacency matrix, Degree distribution |
| Classical Algorithms | ALGO | PageRank, Community detection, Spectral clustering |
| Node Embeddings | EMB | DeepWalk, node2vec, Skip-gram model |
| GNN Architecture | GNN | GCN, GraphSAGE, GAT, GIN, Message passing |
| GNN Theory | THEORY | WL test, Expressiveness, Over-smoothing |
| Graph Transformers | TRANS | Graphormer, GPS, Laplacian positional encoding |
| Knowledge Graphs | KG | TransE, RotatE, ComplEx, Query2Box |
| Heterogeneous Graphs | HETERO | R-GCN, HGT, Meta-path |
| Applications | APP | LightGCN, TGN, Drug discovery, Fraud detection |
| Scalability | SCALE | Neighbor sampling, Cluster-GCN, GraphSAINT |
| Generative Models | GEN | GraphRNN, GCPN, DiGress, VGAE |
| Advanced Topics | ADV | LLM+GNN, Graph foundation models, ULTRA |
| Training & Optimization | TRAIN | DropEdge, PairNorm, Curriculum learning |
| Tools & Frameworks | TOOLS | PyTorch Geometric, NetworkX, OGB, RelBench |

Once the `learning-graph-generator` skill runs, this section will be populated with:

- **Course Description Assessment** — quality review of `course-description.md`
- **Concept Enumeration** — the ~282 concepts the book will cover
- **Concept Taxonomy** — concepts grouped into 15 taxonomy categories
- **Graph Quality Analysis** — DAG structure, foundational and terminal nodes
- **Taxonomy Distribution** — concept count per taxonomy category

Use the `book-installer` skill (option 23) to add an interactive graph viewer once `learning-graph.json` exists.
