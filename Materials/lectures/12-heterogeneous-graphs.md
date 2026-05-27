# Lecture 12: Heterogeneous Graphs

## Overview

Real-world graphs often contain multiple types of nodes and edges (e.g., a drug-protein-disease graph). Heterogeneous GNNs extend the message-passing framework to handle typed nodes and relations.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/09-hetero.pdf

---

## Key Concepts

### What is a Heterogeneous Graph?
- **Node types:** T_V = {type(v)} (e.g., paper, author, venue)
- **Edge types (relations):** T_E = {type(e)} (e.g., writes, cites, publishes_in)
- **Meta-path:** a path through node/edge types (e.g., Author-writes-Paper-cites-Paper)
- Examples: academic networks (DBLP, ACM), biomedical (drug-gene-disease), e-commerce (user-item-category)

### Relational GCN (R-GCN, Schlichtkrull et al., 2018)
- Different weight matrices for each relation type
- **R-GCN update:**
  h_v^(k+1) = σ(Σ_{r ∈ R} Σ_{u ∈ N_r(v)} (1/c_{v,r}) W_r^(k) h_u^(k) + W_0^(k) h_v^(k))
- **Problem:** number of parameters scales with number of relation types
- **Solutions:** basis decomposition or block-diagonal decomposition of W_r

### Heterogeneous Graph Transformer (HGT, Hu et al., 2020)
- Meta-relation-specific attention mechanism
- For each edge (s, r, t): compute attention using source type, relation type, target type
- **3 learnable matrices per relation:** K_τ(s), Q_τ(t), V_τ(s)
- Softmax over all incoming edges of each type
- Captures type-specific semantic interactions

### HAN (Heterogeneous Attention Network)
- Uses **meta-path-based neighbors**
- First-level attention: node-level attention within each meta-path
- Second-level attention: meta-path-level attention across different meta-paths

### RGAT (Relational GAT)
- Combines R-GCN with attention mechanism
- Learns separate attention weights for each relation type

### Practical Considerations
- **Handling isolated node types:** use type-specific projection layers
- **Edge features:** incorporate edge attributes as part of the message
- **Scalability:** full heterogeneous attention is expensive → sample per relation type

## Key Papers
- [R-GCN: Modeling Relational Data with GCNs (Schlichtkrull et al., 2018)](https://arxiv.org/pdf/1703.06103.pdf)
- [HGT: Heterogeneous Graph Transformer (Hu et al., 2020)](https://arxiv.org/pdf/2003.01332.pdf)
- [HAN: Heterogeneous Attention Network (Wang et al., 2019)](https://arxiv.org/abs/1903.07293)

## Textbook Chapter Notes

Maps to **Chapter 12: Heterogeneous Graphs**.
- Real-world example: academic graph (paper, author, venue nodes)
- Code: R-GCN with PyG's `HeteroConv` wrapper
- Dataset: OGB-MAG (Open Academic Graph)

## Code Example

```python
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv

conv = HeteroConv({
    ('paper', 'cites', 'paper'): GCNConv(-1, 64),
    ('author', 'writes', 'paper'): SAGEConv((-1, -1), 64),
    ('paper', 'written_by', 'author'): SAGEConv((-1, -1), 64),
}, aggr='sum')
```
