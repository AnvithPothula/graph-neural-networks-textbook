---
title: "Chapter 15: Heterogeneous Graphs"
description: "Extends message passing to graphs with multiple node and edge types via R-GCN, HGT, and HAN, introducing meta-paths and type-specific projections for richer relational structure."
generated_by: claude skill chapter-content-generator
date: 2026-05-29 23:55:00
version: 0.08
---

# Chapter 15: Heterogeneous Graphs

**Part 4: Graphs in the Wild**

## Summary

Extends message passing to graphs with multiple node and edge types via R-GCN, HGT, and HAN, introducing meta-paths and type-specific projections for richer relational structure.

## Concepts Covered

This chapter covers the following 8 concepts from the learning graph:

1. Heterogeneous GNN
2. R-GCN (Relational GCN)
3. Heterogeneous Graph Transformer (HGT)
4. HAN (Heterogeneous Attention Network)
5. Meta-Path
6. Relation-Specific Weights
7. Basis Decomposition (R-GCN)
8. Type-Specific Projection

## Prerequisites

This chapter builds on:

- [Chapter 6: GNN Foundations: Message Passing and GCN](../06-gnn-foundations/index.md)
- [Chapter 7: GNN Design Space: GraphSAGE and GAT](../07-gnn-design-space/index.md)
- [Chapter 11: Graph Transformers](../11-graph-transformers/index.md)
- [Chapter 12: Knowledge Graph Embeddings](../12-knowledge-graph-embeddings/index.md)

---

!!! mascot-welcome "One Graph, Many Voices"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving welcome">
    Every GNN we have studied so far implicitly assumes that all nodes are the same kind of thing and all edges encode the same kind of relationship. In reality, most large-scale graphs are not homogeneous: they contain multiple semantically distinct node types and edge types whose messages should not be processed identically. A "paper cites paper" edge is not the same relationship as an "author writes paper" edge, and treating them with the same weight matrix discards exactly the type information that makes the graph interesting. This chapter extends message passing to **heterogeneous graphs** — where the node and edge type structure is a first-class citizen of the model — through three architectures (R-GCN, HAN, HGT) that handle typed structure in progressively more sophisticated ways.

---

## 15.1 Motivating Example: The Open Academic Graph

The Open Graph Benchmark's academic graph (OGB-MAG) is a directed heterogeneous graph constructed from the Microsoft Academic Graph, containing four node types and four edge types:

- **Node types:** Paper (736K nodes), Author (1.1M nodes), Field of Study (59K nodes), Institution (8.7K nodes)
- **Edge types:** *writes* (Author → Paper), *cites* (Paper → Paper), *has\_topic* (Paper → Field), *affiliated\_with* (Author → Institution)

The task is to predict each paper's primary field of study (349 classes) using the paper's abstract word2vec embedding, the graph structure, and the typed relational information.

Consider what a standard homogeneous GCN would do when applied to this graph naively. The GCN would aggregate messages from all neighbors regardless of their type: it would mix authors, other papers, and field-of-study nodes in a single aggregation with a single weight matrix. An author contributing a message to a paper it wrote, and another paper contributing a message via citation, would be processed identically. This is a significant loss of information: the "a paper is topically related to X because its author works in field Y" signal is fundamentally different from the "a paper is topically related to X because it cites papers in X directly" signal. Conflating them forces the model to compromise between two structurally different propagation mechanisms.

A **heterogeneous GNN** preserves and exploits this typed structure, using distinct parameter blocks — weight matrices, attention mechanisms, or projections — for each relation type. The three architectures in this chapter take increasingly sophisticated approaches to this problem.

---

## 15.2 Formal Definition: Heterogeneous Graphs

A **heterogeneous graph** (also called a typed graph or multi-relational graph) is a tuple \( G = (\mathcal{V}, \mathcal{E}, \tau, \phi) \) where:

- \( \mathcal{V} \) is the set of nodes, each assigned a type by the mapping \( \tau : \mathcal{V} \to \mathcal{T}_V \) over a finite type set \( \mathcal{T}_V \)
- \( \mathcal{E} \) is the set of directed edges, each assigned a relation type by \( \phi : \mathcal{E} \to \mathcal{T}_E \) over a finite relation type set \( \mathcal{T}_E \)
- Each node \( v \in \mathcal{V} \) carries a feature vector \( \mathbf{x}_v \in \mathbb{R}^{d_{\tau(v)}} \), where the feature dimension \( d_{\tau(v)} \) may differ by node type

A **meta-relation** is a triple \( (\tau(s), \phi(e), \tau(t)) \) specifying the source node type, relation type, and target node type for an edge \( e = (s, t) \). In OGB-MAG there are four meta-relations: (Author, writes, Paper), (Paper, cites, Paper), (Paper, has\_topic, Field), (Author, affiliated\_with, Institution).

The key property distinguishing heterogeneous graphs from knowledge graphs (Chapters 12–14) is that the focus here is on **representation learning** for downstream prediction tasks — node classification, link prediction, graph classification — rather than triple completion. KG embedding methods like TransE and RotatE are optimized specifically for triple scoring; heterogeneous GNNs are designed as general-purpose encoders whose outputs can be fed into any task-specific head.

Several practical differences follow from this focus. Heterogeneous GNNs must handle feature spaces of different dimensions for different node types (authors and papers typically have different feature encodings), must support inductive node classification where test nodes have not been seen during training, and must scale to graphs with millions of nodes — requirements that the transductive, low-dimensional KG embedding approach is not designed to meet.

---

## 15.3 Meta-Paths: Navigating Typed Neighborhoods

Before introducing the GNN architectures, it is worth establishing the **meta-path** abstraction, which underpins the HAN model and provides useful vocabulary for the entire chapter.

A **meta-path** \( \Phi \) is a sequence of node and edge types defining a composite relationship:

\[
\Phi = T_1 \xrightarrow{R_1} T_2 \xrightarrow{R_2} T_3 \cdots \xrightarrow{R_{L-1}} T_L
\]

where each \( T_i \in \mathcal{T}_V \) is a node type and each \( R_i \in \mathcal{T}_E \) is a relation type such that \( (T_i, R_i, T_{i+1}) \) is a valid meta-relation in the graph. A meta-path of length \( L-1 \) defines, for each node of type \( T_1 \), a **meta-path-based neighborhood** consisting of all nodes of type \( T_L \) reachable by following the sequence of relation types \( R_1, R_2, \ldots, R_{L-1} \).

!!! mascot-thinking "Meta-Paths Encode Implicit Semantics"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinking carefully">
    Meta-paths are the heterogeneous graph analogue of multi-hop reasoning in knowledge graphs. Each meta-path captures a distinct semantic relationship that is not directly present as a single edge. Two authors connected by the path Author → writes → Paper ← writes ← Author are **co-authors** — but there is no direct "co-author" edge in the graph. Two papers connected by Paper → has\_topic → Field ← has\_topic ← Paper share a **research topic** without necessarily citing each other. The power of meta-paths is that they reveal latent high-order relationships from sparse typed structure. The weakness is that they require domain knowledge to choose: a researcher must decide which meta-paths are semantically meaningful for the task at hand. Different meta-paths can encode quite different notions of similarity, and choosing the wrong set can hurt performance more than using no meta-paths at all.

In OGB-MAG, three meta-paths are commonly used for paper-level tasks:

| Meta-path | Abbreviation | Semantic meaning |
|---|---|---|
| Paper → cites → Paper | PP | Direct citation |
| Paper ← writes ← Author → writes → Paper | PAP | Co-authored papers |
| Paper → has\_topic → Field ← has\_topic ← Paper | PFP | Same research field |
| Paper ← writes ← Author → affiliated\_with → Institution | PAI | Papers from same institution |

Each meta-path defines a different graph over the same node set, with edges representing the composite relationship. HAN, described in §15.6, applies attention aggregation separately along each meta-path and then learns to combine the resulting representations.

---

## 15.4 R-GCN: Relation-Specific Weight Matrices

Relational GCN (R-GCN, Schlichtkrull et al. 2018) is the most direct generalization of GCN to heterogeneous graphs. The key modification is straightforward: instead of a single shared weight matrix \( \mathbf{W}^{(l)} \) for all neighbor aggregations, R-GCN maintains a separate matrix \( \mathbf{W}_r^{(l)} \) for each relation type \( r \in \mathcal{R} \).

The R-GCN update equation for node \( v \) at layer \( l \) is:

\[
\mathbf{h}_v^{(l+1)} = \sigma\!\left(\mathbf{W}_0^{(l)} \mathbf{h}_v^{(l)} + \sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_r(v)} \frac{1}{|\mathcal{N}_r(v)|} \mathbf{W}_r^{(l)} \mathbf{h}_u^{(l)}\right)
\]

where \( \mathcal{N}_r(v) \) is the set of neighbors of \( v \) connected via relation type \( r \), \( \mathbf{W}_0^{(l)} \) is the self-loop weight matrix, and \( \mathbf{W}_r^{(l)} \in \mathbb{R}^{d \times d} \) is the relation-specific aggregation matrix. The normalization constant \( |\mathcal{N}_r(v)|^{-1} \) prevents high-degree nodes from accumulating unboundedly large representations from a single relation type.

The intuition is identical to message passing in homogeneous GNNs: for each incoming relation type, aggregate the neighbors' messages through a learned linear transformation, then sum across relation types and apply a nonlinearity. The distinction is that each relation type has its own transformation, so "an author writes a paper" and "a paper cites a paper" are processed by different learned linear maps rather than conflated into a single shared map.

### 15.4.1 The Parameter Explosion Problem

R-GCN's relation-specific parameterization has a direct consequence: the number of parameters grows linearly with the number of relation types. With \( |\mathcal{R}| \) relation types and embedding dimension \( d \), the relation weight matrices alone require \( |\mathcal{R}| \times d^2 \) parameters per layer. For OGB-MAG with 4 meta-relations and \( d = 64 \), this is manageable: \( 4 \times 64^2 = 16{,}384 \) parameters per layer. But for knowledge graphs with hundreds of relation types — FB15k-237 has 237 — the count reaches \( 237 \times 64^2 \approx 970{,}000 \) parameters per layer, which is prone to overfitting on sparse relation types where few training examples exist.

---

## 15.5 Basis Decomposition: Controlling the Parameter Budget

R-GCN addresses the parameter explosion with **basis decomposition**: rather than learning each \( \mathbf{W}_r \) independently, it expresses every relation weight matrix as a linear combination of \( B \) shared **basis matrices** \( \{\mathbf{V}_b\}_{b=1}^B \):

\[
\mathbf{W}_r = \sum_{b=1}^{B} a_{r,b}\, \mathbf{V}_b
\]

where:
- \( \mathbf{V}_b \in \mathbb{R}^{d \times d} \) are \( B \) shared basis matrices learned jointly across all relation types
- \( a_{r,b} \in \mathbb{R} \) is a scalar coefficient specific to relation \( r \) and basis \( b \)

This reduces the parameter count from \( |\mathcal{R}| \times d^2 \) to \( B \times d^2 + |\mathcal{R}| \times B \). For \( |\mathcal{R}| = 237 \), \( d = 64 \), \( B = 3 \):

- Full matrices: \( 237 \times 64^2 = 970{,}752 \) parameters
- Basis decomposition: \( 3 \times 64^2 + 237 \times 3 = 12{,}288 + 711 = 12{,}999 \) parameters

A reduction of approximately 74×. Beyond the parameter savings, basis decomposition provides an important **inductive bias**: it forces relation weight matrices to share subspace structure, which acts as a regularizer. Rare relation types — those with few training triples — benefit from the shared basis, because the basis directions are informed by common structural patterns observed across all relation types. Two relation types that often appear in similar graph contexts will learn similar coefficient vectors \( \{a_{r,b}\} \), even if their individual triples are sparse.

The choice of \( B \) controls the trade-off between expressivity and regularization. With \( B = |\mathcal{R}| \) and linearly independent basis matrices, basis decomposition reduces to independent full matrices. With \( B = 1 \), all relation matrices are scalar multiples of the same template, severely restricting expressivity. Typical values are \( B \in [2, 30] \) for knowledge graphs with hundreds of relation types.

!!! mascot-tip "Choosing B in Practice"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage offering practical advice">
    A useful heuristic for selecting the basis count \( B \): start with \( B = \lceil \sqrt{|\mathcal{R}|} \rceil \). For OGB-MAG with 4 relation types, this gives \( B = 2 \) — essentially no decomposition. For FB15k-237 with 237 types, this gives \( B = 16 \). Tune \( B \) via validation loss, watching for the point where increasing \( B \) yields diminishing returns: that plateau indicates the model has captured the rank of the true relation weight matrix subspace. Also monitor per-relation-type validation performance separately — a globally good model may still underfit rare relation types, which often indicates \( B \) is too small relative to the diversity of structural patterns.

---

## 15.6 Type-Specific Projections: Bridging Heterogeneous Feature Spaces

A practical challenge that arises before any heterogeneous GNN layer can run is the **feature dimension mismatch** across node types. In OGB-MAG, papers have 128-dimensional word2vec embeddings, authors have no raw features (represented by a learned embedding vector), institutions have a learned embedding, and fields of study have their own feature encoding. A message-passing layer expecting a single input dimension cannot be applied directly.

The standard solution is to apply a **type-specific linear projection** \( \mathbf{W}_{\tau(v)} \in \mathbb{R}^{d \times d_{\tau(v)}} \) that maps each node's raw features into a shared embedding space of dimension \( d \):

\[
\mathbf{h}_v^{(0)} = \mathbf{W}_{\tau(v)}\, \mathbf{x}_v + \mathbf{b}_{\tau(v)}
\]

where \( \mathbf{W}_{\tau(v)} \) and \( \mathbf{b}_{\tau(v)} \) are learned parameters specific to node type \( \tau(v) \). After this initial projection, all nodes live in the same \( d \)-dimensional space and standard message-passing layers can be applied. In PyTorch Geometric, the linear layer `nn.Linear(-1, d)` with lazy initialization handles this automatically by inferring the input dimension from the first batch of data.

Type-specific projections serve a dual purpose. First, they address the feature dimension mismatch (the practical necessity). Second, they introduce a **semantic alignment**: the projection ideally maps nodes of different types to a shared embedding space where proximity reflects semantic relatedness, not just structural proximity in the graph. A well-trained projection layer aligns the "paper about machine learning" region of the embedding space with the "author who publishes machine learning papers" region, enabling cross-type aggregation to carry meaningful information.

---

## 15.7 HAN: Two-Level Hierarchical Attention

Heterogeneous Attention Network (HAN, Wang et al. 2019) takes a fundamentally different approach from R-GCN. Rather than directly aggregating typed neighbors in a single layer, HAN first **expands** the graph into a collection of meta-path-induced subgraphs and then applies a two-level attention mechanism: one within each meta-path neighborhood and one across meta-paths.

### 15.7.1 Node-Level Attention Within a Meta-Path

For a fixed meta-path \( \Phi \), every node \( v \) has a meta-path-based neighborhood \( \mathcal{N}_\Phi(v) \) consisting of all nodes reachable via \( \Phi \). To aggregate these neighbors, HAN applies a GAT-style attention (Chapter 7) with a meta-path-specific projection matrix \( \mathbf{W}_\Phi \):

**Attention logit:**

\[
e_{vu}^\Phi = \text{LeakyReLU}\!\left(\mathbf{a}_\Phi^T \left[\mathbf{W}_\Phi \mathbf{h}_v \,\|\, \mathbf{W}_\Phi \mathbf{h}_u\right]\right)
\]

**Normalized weight:**

\[
\alpha_{vu}^\Phi = \frac{\exp(e_{vu}^\Phi)}{\displaystyle\sum_{k \in \mathcal{N}_\Phi(v)} \exp(e_{vk}^\Phi)}
\]

**Aggregated representation:**

\[
\mathbf{z}_v^\Phi = \sigma\!\left(\sum_{u \in \mathcal{N}_\Phi(v)} \alpha_{vu}^\Phi\, \mathbf{W}_\Phi \mathbf{h}_u\right)
\]

where \( \mathbf{a}_\Phi \in \mathbb{R}^{2d} \) is a meta-path-specific attention vector. After this step, node \( v \) has one representation \( \mathbf{z}_v^\Phi \) for each meta-path \( \Phi \in \mathcal{P} \).

### 15.7.2 Semantic-Level Attention Across Meta-Paths

The second level of attention determines how much each meta-path contributes to the final representation. HAN learns a shared semantic attention vector \( \mathbf{q} \in \mathbb{R}^d \) and computes the importance of meta-path \( \Phi \) as the average alignment between the semantic vector and the node representations produced by that meta-path:

\[
w_\Phi = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathbf{q}^T \tanh\!\left(\mathbf{W}\, \mathbf{z}_v^\Phi + \mathbf{b}\right)
\]

\[
\beta_\Phi = \frac{\exp(w_\Phi)}{\displaystyle\sum_{\Phi' \in \mathcal{P}} \exp(w_{\Phi'})}
\]

The final node representation is the convex combination of all meta-path representations, weighted by the learned semantic attention scores:

\[
\mathbf{z}_v = \sum_{\Phi \in \mathcal{P}} \beta_\Phi\, \mathbf{z}_v^\Phi
\]

The semantic-level attention is interpretable: after training, \( \beta_\Phi \) reveals which meta-paths are most informative for the task. For paper classification, the PAP (co-authorship) meta-path typically receives high weight because co-authored papers are likely in the same field, while the PFP (same field) meta-path may receive lower weight if field labels are noisy.

HAN's limitation is that it requires **manual meta-path specification**, which demands domain knowledge and does not scale well to graphs with many relation types. It also materializes all meta-path-expanded adjacency matrices, which can be dense and memory-intensive for long meta-paths.

---

## 15.8 HGT: Heterogeneous Graph Transformer

Heterogeneous Graph Transformer (HGT, Hu et al. 2020) eliminates the need for manual meta-path specification by extending the standard multi-head transformer attention to be **type-aware at the level of individual edges**. Rather than specifying which meta-paths to follow, HGT makes attention type-sensitive across all direct (1-hop) edges, allowing multi-hop type-aware propagation to emerge through the stacking of layers.

### 15.8.1 Meta-Relation Parameterization

HGT introduces a separate parameter set for each **meta-relation** \( (\tau(s), \phi(e), \tau(t)) \) — the combination of source node type, edge relation type, and target node type. For an edge \( e = (s, t) \), the attention computation involves three type-specific matrices:

**Key projection** (transforms source node for key computation):

\[
\mathbf{K}(s, e, t) = \mathbf{h}_s\, \mathbf{W}^\text{Key}_{\tau(s), \phi(e), \tau(t)}
\]

**Query projection** (transforms target node for query computation):

\[
\mathbf{Q}(s, e, t) = \mathbf{h}_t\, \mathbf{W}^\text{Query}_{\tau(s), \phi(e), \tau(t)}
\]

**Attention score** (scaled dot-product with a meta-relation-specific prior):

\[
\text{Attn}(s, e, t) = \frac{\mathbf{K}(s, e, t) \cdot \mathbf{Q}(s, e, t)^T}{\sqrt{d / H}} \cdot \mu_{\tau(s), \phi(e), \tau(t)}
\]

where \( H \) is the number of attention heads and \( \mu_{\tau(s), \phi(e), \tau(t)} \in \mathbb{R} \) is a learned scalar **attention prior** that amplifies or attenuates the attention signal for each meta-relation type before the softmax.

**Value projection** (transforms source node message):

\[
\mathbf{V}(s, e, t) = \mathbf{h}_s\, \mathbf{W}^\text{Msg}_{\tau(s), \phi(e), \tau(t)}
\]

### 15.8.2 Aggregation and Update

The updated representation of target node \( t \) at layer \( l+1 \) aggregates messages from all incoming edges, with softmax normalization computed separately for each target node:

\[
\mathbf{h}_t^{(l+1)} = \mathbf{W}_{\tau(t)}^\text{proj}\!\left(\bigoplus_{(s,e,t) \in \mathcal{E}} \text{softmax}_{s}\!\left(\text{Attn}(s,e,t)\right) \cdot \mathbf{V}(s, e, t)\right)
\]

where \( \bigoplus \) denotes concatenation of multi-head outputs and \( \mathbf{W}_{\tau(t)}^\text{proj} \) is a type-specific output projection. The softmax is computed over all source nodes \( s \) sending edges to target \( t \), regardless of their type — HGT does not require separate softmax normalization per relation type, relying instead on the attention prior \( \mu \) to distinguish meta-relations.

!!! mascot-encourage "HGT's Complexity Is Manageable"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage encouraging the reader">
    HGT's parameter count per layer may look intimidating: three matrices per meta-relation (\( \mathbf{W}^\text{Key} \), \( \mathbf{W}^\text{Query} \), \( \mathbf{W}^\text{Msg} \)), each of dimension \( (d/H) \times d \), plus one prior scalar \( \mu \) per meta-relation. For OGB-MAG with 4 meta-relations and \( H = 8 \) heads at \( d = 256 \): \( 4 \times 3 \times 32 \times 256 \approx 98{,}000 \) parameters per layer for the type-specific weights plus \( 4 \) scalars for the priors. This is not large by modern standards. The structured parameterization is exactly what makes HGT tractable: by separating the key, query, and value transformations into type-specific components, the model can represent diverse relation-type semantics without a combinatorial explosion in parameters. Think of \( \mathbf{W}^\text{Key}_{\tau(s),\phi(e),\tau(t)} \) as asking: "given that I am a source of type \( \tau(s) \) sending a message of type \( \phi(e) \) to a target of type \( \tau(t) \), how should I encode my features as a key?" — a focused, interpretable question.

### 15.8.3 Comparison: R-GCN vs. HAN vs. HGT

The three architectures represent distinct design philosophies, which the following table summarizes after the individual discussions above:

| Property | R-GCN | HAN | HGT |
|---|---|---|---|
| Meta-path required? | No | Yes (manual) | No |
| Attention? | No | Two-level | Transformer |
| Parameterization | Per relation \( \mathbf{W}_r \) | Per meta-path | Per meta-relation (K/Q/V) |
| Parameter reduction | Basis decomposition | — | Head sharing |
| Works on direct edges? | Yes | Via meta-path expansion | Yes |
| Inductive? | Yes | Partially | Yes |
| Scalability | High | Medium | Medium |

---

## 15.9 Complete Code: Heterogeneous GNNs in PyTorch Geometric

The following code demonstrates three things: constructing a heterogeneous graph data object for OGB-MAG style data, applying R-GCN-style aggregation via PyG's `HeteroConv`, and applying HGT via PyG's `HGTConv`. Each model is trained for node classification on the paper nodes. The `HeteroConv` wrapper handles the per-relation aggregation and automatically sums contributions from different relation types after applying relation-specific convolutions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, HGTConv, Linear

# ── 1. Build a toy heterogeneous graph (OGB-MAG style) ───────────────────────

def make_toy_hetero_graph(
    n_paper: int = 200,
    n_author: int = 80,
    n_field:  int = 20,
    feat_dim_paper: int = 128,   # paper word2vec features
    feat_dim_field: int = 64,    # field embedding dimension
    num_classes: int = 5,        # paper classification classes
    seed: int = 0,
) -> HeteroData:
    """
    Creates a synthetic heterogeneous graph with node types:
      paper, author, field_of_study
    and meta-relations:
      (author, writes, paper)
      (paper, cites, paper)
      (paper, has_topic, field_of_study)

    Authors have no raw features and will use a learned embedding.
    """
    torch.manual_seed(seed)
    data = HeteroData()

    # Node features (papers and fields have raw features; authors use learned emb)
    data['paper'].x = torch.randn(n_paper, feat_dim_paper)
    data['field_of_study'].x = torch.randn(n_field, feat_dim_field)
    # Authors: num_nodes only — LinearLayer with lazy init will handle feature-free nodes
    data['author'].num_nodes = n_author

    # Paper labels for classification
    data['paper'].y = torch.randint(0, num_classes, (n_paper,))
    # Train/val/test split masks on paper nodes
    perm = torch.randperm(n_paper)
    data['paper'].train_mask = torch.zeros(n_paper, dtype=torch.bool)
    data['paper'].train_mask[perm[:120]] = True
    data['paper'].val_mask = torch.zeros(n_paper, dtype=torch.bool)
    data['paper'].val_mask[perm[120:160]] = True
    data['paper'].test_mask = torch.zeros(n_paper, dtype=torch.bool)
    data['paper'].test_mask[perm[160:]] = True

    # Edges (random for this toy graph)
    def rand_edges(src_n, dst_n, num_edges):
        src = torch.randint(0, src_n, (num_edges,))
        dst = torch.randint(0, dst_n, (num_edges,))
        return torch.stack([src, dst])

    data['author', 'writes', 'paper'].edge_index       = rand_edges(n_author, n_paper, 400)
    data['paper',  'cites',  'paper'].edge_index       = rand_edges(n_paper, n_paper, 600)
    data['paper',  'has_topic', 'field_of_study'].edge_index = rand_edges(n_paper, n_field, 300)

    return data


# ── 2. R-GCN-style model with HeteroConv ────────────────────────────────────

class RGCN(nn.Module):
    """
    Two-layer R-GCN implemented with PyG's HeteroConv.

    HeteroConv applies a separate convolution for each meta-relation and
    sums the results for each target node type. Here we use SAGEConv for
    all meta-relations (it handles bipartite source/target sizes natively).

    Parameters
    ----------
    metadata : tuple
        (node_types, edge_types) from HeteroData.metadata().
    hidden_dim : int
        Embedding dimension for all node types after first layer.
    num_classes : int
        Output classes for paper node classification.
    """
    def __init__(self, metadata, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        node_types, edge_types = metadata

        # Type-specific input projections: map each node type's raw features
        # (or learned embedding) to a shared hidden_dim space
        self.proj = nn.ModuleDict({
            ntype: Linear(-1, hidden_dim) for ntype in node_types
        })

        # First heterogeneous conv layer: one SAGEConv per meta-relation
        self.conv1 = HeteroConv(
            {etype: SAGEConv((-1, -1), hidden_dim) for etype in edge_types},
            aggr='sum'  # sum contributions from all relation types for each target
        )
        # Second layer
        self.conv2 = HeteroConv(
            {etype: SAGEConv((-1, -1), hidden_dim) for etype in edge_types},
            aggr='sum'
        )
        self.classifier = Linear(hidden_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        # Step 1: type-specific projection of raw features to shared space
        # For author nodes with no raw features, x_dict['author'] = zeros
        h = {ntype: F.relu(self.proj[ntype](x)) for ntype, x in x_dict.items()}

        # Step 2: two rounds of heterogeneous message passing
        h = self.conv1(h, edge_index_dict)
        h = {ntype: F.relu(feat) for ntype, feat in h.items()}
        h = self.conv2(h, edge_index_dict)

        # Step 3: classify paper nodes
        return self.classifier(h['paper'])


# ── 3. HGT model ─────────────────────────────────────────────────────────────

class HGT(nn.Module):
    """
    Two-layer HGT with meta-relation-specific K/Q/V attention.

    Parameters
    ----------
    metadata : tuple
        (node_types, edge_types) from HeteroData.metadata().
    hidden_dim : int
        Shared embedding dimension across all node types.
    heads : int
        Number of attention heads per HGT layer.
    num_classes : int
        Output classes for paper classification.
    """
    def __init__(self, metadata, hidden_dim: int = 64,
                 heads: int = 4, num_classes: int = 5):
        super().__init__()
        node_types, _ = metadata

        # Type-specific input projections (same as R-GCN)
        self.proj = nn.ModuleDict({
            ntype: Linear(-1, hidden_dim) for ntype in node_types
        })

        # HGTConv layer 1: learns K, Q, V matrices per meta-relation
        self.hgt1 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=heads)
        # HGTConv layer 2
        self.hgt2 = HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=heads)
        self.classifier = Linear(hidden_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        h = {ntype: F.relu(self.proj[ntype](x)) for ntype, x in x_dict.items()}
        h = self.hgt1(h, edge_index_dict)
        h = {ntype: F.relu(feat) for ntype, feat in h.items()}
        h = self.hgt2(h, edge_index_dict)
        return self.classifier(h['paper'])


# ── 4. Training loop ──────────────────────────────────────────────────────────

def train_hetero(model_name: str = 'rgcn', epochs: int = 50):
    """
    Train R-GCN or HGT on the toy heterogeneous graph.

    model_name : 'rgcn' or 'hgt'
    epochs     : number of training epochs
    """
    data = make_toy_hetero_graph()
    metadata = data.metadata()     # (node_types, edge_types)

    # For feature-free author nodes, create a zero-feature tensor
    if 'author' not in data.x_dict:
        data['author'].x = torch.zeros(data['author'].num_nodes, 1)

    if model_name == 'rgcn':
        model = RGCN(metadata, hidden_dim=64, num_classes=5)
    else:
        model = HGT(metadata, hidden_dim=64, heads=4, num_classes=5)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x_dict, data.edge_index_dict)  # (n_paper, num_classes)
        loss = F.cross_entropy(out[data['paper'].train_mask],
                               data['paper'].y[data['paper'].train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x_dict, data.edge_index_dict)
                pred = out.argmax(dim=-1)
                val_acc = (pred[data['paper'].val_mask] ==
                           data['paper'].y[data['paper'].val_mask]).float().mean()
            print(f"Epoch {epoch:3d} | Loss {loss:.4f} | Val Acc {val_acc:.4f}")

    return model

if __name__ == '__main__':
    print("=== R-GCN ===")
    train_hetero('rgcn', epochs=50)
    print("\n=== HGT ===")
    train_hetero('hgt', epochs=50)
```

Three design decisions in this code are worth noting. First, the `proj` dictionary of `Linear(-1, hidden_dim)` layers performs the **type-specific projections** (§15.6): PyG's `Linear(-1, ...)` uses lazy initialization to infer the actual input dimension on the first forward pass, so papers with 128-dimensional features and fields with 64-dimensional features are handled automatically. Second, `HeteroConv` with `aggr='sum'` implements **R-GCN-style aggregation** (§15.4): it applies a distinct convolution per meta-relation, then sums the resulting messages for each target node. Third, `HGTConv` implements the full **meta-relation K/Q/V attention** (§15.8) transparently from the caller's perspective — the `metadata` argument tells PyG how many meta-relations to parameterize.

---

## 15.10 Benchmark Results: OGB-MAG Node Classification

The following table reports validation accuracy on the ogbn-mag benchmark (paper node classification, 349 classes) for several heterogeneous GNN methods. OGB-MAG uses a temporal split: training papers are from before 2018, validation papers from 2018, test papers from 2019 onward.

| Model | Valid Acc (%) | Meta-path required | Year |
|---|---|---|---|
| MLP (no graph) | 36.6 | — | 2020 |
| R-GCN | 47.4 | No | 2018 |
| HAN (APA + APC) | 39.4 | Yes | 2019 |
| HGT | 49.9 | No | 2020 |
| HGT + NeighborSampler | 50.8 | No | 2020 |
| SeHGNN | 59.7 | Automated | 2023 |
| GLEM (LLM + HGT) | 64.6 | No | 2023 |

Several results deserve commentary. HAN's lower accuracy relative to R-GCN is counterintuitive but explained by its meta-path expansion strategy: the APA meta-path (Author → writes → Paper ← writes ← Author) collapses all papers co-authored with a given author's papers into the neighborhood, which becomes very dense and noisy for prolific authors. HGT avoids this by operating on direct edges only. SeHGNN replaces manual meta-path specification with a learned meta-path selector, recovering HAN's interpretable attention structure while automating the design choice. The GLEM result illustrates an emerging pattern across the textbook: LLM-derived semantic features on top of a structural heterogeneous GNN significantly outperform either component alone.

---

## 15.11 MicroSim: Heterogeneous Graph Explorer

#### Diagram: Typed Node and Edge Explorer


<iframe src="../../sims/ch15-hetero-graph-explorer/main.html" width="100%" height="522px" scrolling="no"></iframe>
[Run Typed Node and Edge Explorer Fullscreen](../../sims/ch15-hetero-graph-explorer/main.html)

---

## 15.12 Common Pitfalls

!!! mascot-warning "Five Traps in Heterogeneous Graph Modeling"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning about common mistakes">
    Heterogeneous GNNs introduce new failure modes beyond standard homogeneous GNN pitfalls. The most damaging are: (1) **ignoring type imbalance** — if 90% of nodes are of one type, the shared embedding space will be dominated by that type's geometry; (2) **aggregating over the wrong normalization** — using the total node degree \( d_v \) rather than the per-relation degree \( |\mathcal{N}_r(v)| \) for normalization in R-GCN dilutes relation-specific signals; (3) **choosing B too small** in basis decomposition and attributing poor performance to the model rather than underfitting; (4) **treating meta-paths as fixed hyperparameters** when the task may reward learning them; and (5) **omitting reverse edges** — if only (author, writes, paper) edges are included but not (paper, written\_by, author), paper-to-author information flow is blocked entirely.

**Symmetric edge types are often missing.** Most real-world graphs are undirected at the semantic level but stored as directed edges. In OGB-MAG, the *writes* relation is directed (author → paper), but papers often need to aggregate author information too. Adding reverse edges — a (paper, written\_by, author) edge type — is almost always beneficial and is standard practice in PyG heterogeneous graph construction.

**Meta-path selection introduces selection bias.** HAN's performance depends critically on which meta-paths are included. A researcher who tries all combinations and selects the best on validation data has effectively used the validation set as a meta-path tuning set, which can lead to overly optimistic reported accuracy. SeHGNN's automated meta-path selection addresses this by jointly learning which meta-paths are useful, but it requires that the meta-path search space be defined in advance.

**Basis decomposition is not always beneficial.** For graphs with few relation types (e.g., OGB-MAG with 4), basis decomposition with a small \( B \) can actually hurt performance by forcing the weight matrices into a too-restrictive subspace. Basis decomposition is most valuable when \( |\mathcal{R}| \gg 10 \) and per-relation training data is sparse.

---

## 15.13 Exercises

The following 12 exercises span all six levels of Bloom's taxonomy.

**Remember**

1. Define "heterogeneous graph" formally, specifying all components of the tuple \( G = (\mathcal{V}, \mathcal{E}, \tau, \phi) \) and what distinguishes it from a homogeneous graph and from a knowledge graph.
2. List the three matrices that HGT computes per meta-relation and explain in one sentence what each one is used for in the attention computation.

**Understand**

3. Explain why the standard GCN aggregation equation \( \mathbf{h}_v^{(l+1)} = \sigma(\sum_{u \in \mathcal{N}(v)} \tilde{A}_{vu} \mathbf{W} \mathbf{h}_u^{(l)}) \) is insufficient for heterogeneous graphs. Use the OGB-MAG example with author and citation edges to illustrate your explanation.
4. Explain in plain language why HAN uses two levels of attention (node-level and semantic-level). What specific limitation of using only node-level attention does the semantic-level attention address?

**Apply**

5. For the OGB-MAG graph, manually enumerate all meta-paths of length 2 that start and end at Paper nodes. For each, write out the type sequence and propose one semantic interpretation.
6. Given a relation weight matrix \( \mathbf{W}_r \in \mathbb{R}^{64 \times 64} \) and a basis set \( \{\mathbf{V}_b\}_{b=1}^3 \) where each \( \mathbf{V}_b \in \mathbb{R}^{64 \times 64} \), compute the total parameter count for (a) independently learned \( \mathbf{W}_r \) across 50 relation types and (b) basis decomposition with \( B = 3 \). Show the calculation and compute the compression ratio.

**Analyze**

7. HAN achieves lower validation accuracy (39.4%) than R-GCN (47.4%) on OGB-MAG despite using more sophisticated two-level attention. Propose at least two structural or implementation explanations for this gap, drawing on specific properties of the OGB-MAG graph and HAN's architecture.
8. Compare R-GCN and HGT on the following four dimensions: (a) sensitivity to the number of relation types, (b) ability to weight incoming messages from different neighbors differently, (c) dependence on manual design choices, and (d) computational cost per layer. Which model would you recommend for a graph with 200 relation types and which for a graph with 4 relation types? Justify your answers.

**Evaluate**

9. A colleague claims: "Since HGT uses transformer-style attention, it should always outperform R-GCN on any heterogeneous graph." Critically evaluate this claim. Identify at least two conditions under which R-GCN would be expected to match or outperform HGT.
10. Evaluate the experimental rigor of the following setup: a researcher trains HAN on a biomedical graph using 5 manually selected meta-paths, selects the best-performing meta-path subset on the validation set, and reports test accuracy. What concerns arise, and what would a more rigorous experimental protocol look like?

**Create**

11. Design a **heterogeneous GNN** for a movie recommendation graph with node types \{User, Movie, Actor, Director, Genre\} and edge types \{watches, acts\_in, directs, belongs\_to\_genre\}. Specify: (a) which meta-relations require separate weight matrices, (b) which meta-paths are semantically meaningful for the recommendation task, (c) whether you would use R-GCN, HAN, or HGT and why, and (d) how you would handle the cold-start problem for new movies with no watches edges.
12. Propose a variant of basis decomposition that takes advantage of **relation type hierarchy**: if some relation types are known to be semantically related (e.g., *cites* and *builds\_on* are both "intellectual lineage" relations), the decomposition should share more basis components between related types than between unrelated types. Formalize the parameterization and describe a training procedure.

---

## 15.14 Further Reading

**Modeling Relational Data with Graph Convolutional Networks (R-GCN)** — Schlichtkrull, Kipf, Bloem, van den Berg, Titov, Welling (2018). The foundational paper for relational GNNs, introducing both the relation-specific weight matrix formulation and the basis and block-diagonal decompositions. The paper applies R-GCN to both entity classification and link prediction on standard KG benchmarks, demonstrating that the decomposition significantly improves generalization on sparse relation types.

**Heterogeneous Graph Transformer (HGT)** — Hu, Dong, Wang, Sun (2020). Introduces the meta-relation-specific K/Q/V attention mechanism and demonstrates HGT on the large-scale OGB-MAG benchmark. Particularly worth reading for §4 (web-scale graph construction) and the discussion of relative positional encodings for time-aware heterogeneous graphs — a direction not covered in this chapter but increasingly relevant for dynamic academic and social graphs.

**Heterogeneous Attention Network (HAN)** — Wang, Ji, Shi, Wang, Ye, Cui, Yu (2019). The first prominent meta-path-based heterogeneous GNN with interpretable two-level attention. The paper's analysis of the learned semantic-level attention weights \( \beta_\Phi \) across different tasks (node classification vs. link prediction) is instructive: the most informative meta-path differs by task even on the same graph.

**OpenHGNN: An Open Source Toolkit for Heterogeneous Graph Neural Networks** — Zhao et al. (2021). A practical implementation toolkit covering 16 heterogeneous GNN models including R-GCN, HAN, HGT, and several later models. Useful for reproducing results and comparing methods on standardized benchmarks beyond OGB-MAG.

**Are We Really Making Much Progress? Revisiting Heterogeneous Graph Learning** — Lv et al. (2021). A critical revisit of heterogeneous GNN benchmarks that shows many prior results are not reproducible due to inconsistent evaluation protocols, data preprocessing choices, and hyperparameter tuning. Essential reading for researchers conducting empirical comparisons in this area — and a reminder that benchmark accuracy improvements do not always reflect genuine architectural advances.

**SeHGNN: Simple and Efficient Heterogeneous Graph Neural Network** — Yang et al. (2023). Demonstrates that combining pre-computed multi-hop meta-path aggregation (similar to SGC's approach for homogeneous graphs) with automated meta-path selection achieves state-of-the-art results on OGB-MAG with a model simpler than HGT, challenging the assumption that greater architectural complexity yields better heterogeneous graph representations.

**Relational Graph Convolutional Networks: A Closer Look** — Vashishth, Sanyal, Nitin, Talukdar (2020). Analyzes R-GCN's behavior on KG completion benchmarks and proposes several improvements including interaction-based message passing and entity-aware gating. Provides useful ablations for understanding which components of R-GCN's design matter most.

---

!!! mascot-celebration "You've Typed the Graph"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating">
    Heterogeneous graphs are ubiquitous: every system that models interactions between multiple types of entities — academic networks, biomedical databases, e-commerce graphs, social platforms — is a heterogeneous graph waiting for typed message passing. The three architectures you have encountered in this chapter (R-GCN, HAN, HGT) represent a progression from the minimal change to GCN needed to handle typed edges, through meta-path-based hierarchical attention, to fully type-aware transformer attention. The conceptual thread running through all of them is the same idea you first encountered in Chapter 6: information propagates through the graph neighborhood, but now that neighborhood is typed, and the propagation parameters must respect those types. In Chapter 16, we will apply a subset of these ideas to recommender systems, where the user-item bipartite graph is the simplest possible heterogeneous graph.

[See Annotated References](./references.md)
