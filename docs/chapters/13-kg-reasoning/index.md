---
title: "Chapter 13: Reasoning over Knowledge Graphs"
description: "Moves from entity embedding to query answering — multi-hop path queries, conjunctive box embeddings (Query2Box), and the Neural Bellman-Ford Network for state-of-the-art relational path reasoning."
generated_by: claude skill chapter-content-generator
date: 2026-05-29 22:05:27
version: 0.08
---

# Chapter 13: Reasoning over Knowledge Graphs

**Part 3: Knowledge Graphs and Reasoning**

## Summary

Moves from entity embedding to query answering: covers query embedding, box embeddings for conjunctive queries (Query2Box), and the Neural Bellman-Ford Network for multi-hop path reasoning, with complete treatment of EPFO query classes and their geometric representations.

## Concepts Covered

This chapter covers the following 5 concepts from the learning graph:

1. Query Embedding
2. Box Embedding (Query2Box)
3. Multi-Hop Query
4. Conjunctive Query
5. Neural Bellman-Ford Net

## Prerequisites

This chapter builds on:

- [Chapter 12: Knowledge Graph Embeddings](../12-knowledge-graph-embeddings/index.md)

---

!!! mascot-welcome "Beyond Triple Completion — Reasoning at Scale"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving welcome">
    Chapter 12 taught you how to score individual triples: given \( (h, r, ?) \), rank all entities by their plausibility as the tail. That is a 1-hop lookup — one relation step from a known entity. But real-world reasoning rarely stops at one hop. A physician asking "Which approved drugs target proteins associated with Alzheimer's?" is asking a multi-hop query across a biomedical knowledge graph, requiring traversal through gene regulation, protein function, drug-target interaction, and drug approval status simultaneously. This chapter develops the geometric and algorithmic machinery to answer such queries efficiently — even when the KG is severely incomplete.

---

## 13.1 Motivating Example: Multi-Hop Biomedical Queries

Alzheimer's disease research illustrates the multi-hop query problem vividly. A biologist searching a biomedical knowledge graph might pose the question: *"Find proteins that are encoded by genes regulated by the transcription factor NFκB AND that are targeted by at least one FDA-approved oncology drug."* Written as a query graph, this request decomposes into two reasoning chains that must be satisfied simultaneously:

\[
\text{Chain 1:}\quad \text{NFkB} \xrightarrow{\textit{regulates}} \text{Gene}_{?} \xrightarrow{\textit{encodes}} \text{Protein}_{?}
\]

\[
\text{Chain 2:}\quad \text{Drug}_{?} \xrightarrow{\textit{targets}} \text{Protein}_{?} \quad\wedge\quad \text{Drug}_{?} \xrightarrow{\textit{approvedFor}} \text{Oncology}
\]

The two chains must agree on the same \( \text{Protein}_? \). This is a **conjunctive query** — a logical AND over multiple path conditions. No single triple-completion lookup can answer it; the system must traverse multiple edges and enforce consistency across chains.

The challenge is compounded by incompleteness: many of the edges in the biomedical KG are missing. Gene-regulation relationships are only partially recorded; drug-target interactions are perhaps 30% complete in published databases. A system that answers only over observed triples will miss most correct answers. The embedding-based approach developed in this chapter predicts answers even when intermediate facts are absent, by reasoning through the geometry of the embedding space rather than requiring explicit graph traversal.

---

## 13.2 Multi-Hop Queries: A Formal Taxonomy

Before developing the machinery, it is worth establishing a precise taxonomy of query types. All queries considered in this chapter are formulated over a set of entities \( \mathcal{E} \), a set of relations \( \mathcal{R} \), and an incomplete set of observed triples \( \mathcal{T} \).

### 13.2.1 Path Queries

The simplest multi-hop extension of link prediction is the **path query**: a chain of relation edges starting from a known anchor entity \( e_s \) and asking for all entities reachable via a specific sequence of relation types \( (r_1, r_2, \ldots, r_L) \):

\[
q = \bigl(e_s,\ (r_1, r_2, \ldots, r_L),\ ?\bigr)
\]

A 1-hop path query (\( L = 1 \)) is exactly the link prediction problem from Chapter 12. A 2-hop path query (\( L = 2 \)) asks: follow relation \( r_1 \) from \( e_s \) to some intermediate entity, then follow \( r_2 \) to reach the answer. For example, \( (\text{Einstein},\ (\textit{bornIn},\, \textit{officialLanguage}),\ ?) \) asks for languages spoken in Einstein's birth country.

### 13.2.2 Conjunctive Queries and the EPFO Class

A **conjunctive query** extends path queries to require simultaneous satisfaction of multiple path conditions, connected by logical AND. The EPFO (Existential Positive First-Order) query class covers all combinations of AND (\( \wedge \)) and OR (\( \vee \)) operations over path conditions, where variables are existentially quantified. Standard query shapes have been assigned short names in the literature:

| Query type | Notation | Description |
|---|---|---|
| 1-hop | **1p** | Single relation edge from one anchor |
| 2-hop chain | **2p** | Two sequential edges from one anchor |
| 3-hop chain | **3p** | Three sequential edges from one anchor |
| 2-chain intersection | **2i** | Two 1-hop chains sharing the answer variable |
| 3-chain intersection | **3i** | Three 1-hop chains sharing the answer variable |
| Path then intersection | **pi** | 2-hop chain AND a 1-hop chain |
| Intersection then path | **ip** | Two 1-hop chains intersect, then one more hop |

The 2i query is perhaps the most important: it asks for entities that satisfy two independent path conditions simultaneously:

\[
q^{\text{2i}} = \bigl\{t \in \mathcal{E} : \exists a_1\ (e_1, r_1, a_1) \wedge (a_1, r_2, t) \wedge \exists a_2\ (e_2, r_3, a_2) \wedge (a_2, r_4, t)\bigr\}
\]

Two independent chains (starting from anchor entities \( e_1 \) and \( e_2 \)) must converge at the same answer entity \( t \). This is the biomedical query from §13.1: the NFκB chain and the drug-target chain must point to the same protein.

### 13.2.3 Why Graph Traversal Fails on Incomplete KGs

A natural approach to multi-hop query answering is pure graph traversal: start at the anchor entity, follow all edges matching \( r_1 \), then from those intermediate nodes follow all edges matching \( r_2 \), and so on. This gives exact answers on a complete KG — but on an incomplete KG, traversal fails if any intermediate edge is missing.

!!! mascot-thinking "Recall Degrades Exponentially with Path Length"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinking about the missing edges problem">
    If a KG is 80% complete (each true edge is observed with probability 0.8, independently), then the probability of an entire L-hop path being traversable is \( 0.8^L \). At \( L = 2 \): 64% recall. At \( L = 3 \): 51%. At \( L = 5 \): 33%. Even a relatively complete KG loses two-thirds of correct answers on 5-hop queries. For biomedical KGs that are 30–50% complete, the situation is far worse. This is why raw graph traversal is fundamentally inadequate for multi-hop reasoning: the expected recall decreases exponentially with path length. Embedding-based query answering circumvents this by reasoning through the smooth geometry of the embedding space, inferring the plausibility of missing intermediate facts rather than requiring their explicit presence.

---

## 13.3 Query Embedding: From Entities to Queries

The **query embedding** paradigm lifts the entity embedding idea from Chapter 12 to the level of entire queries. Instead of learning a vector for each entity, we learn to map each *query* to a geometric object in the embedding space such that true answer entities cluster inside or near that object.

For path queries, the simplest query embedding is chain composition. Using the TransE convention (relation = translation vector), the query embedding for a 2-hop path is:

\[
\mathbf{q}_{(e_s, r_1, r_2)} = \mathbf{e}_s + \mathbf{r}_1 + \mathbf{r}_2
\]

A candidate answer entity \( t \) is scored by its distance to the query point:

\[
f(q, t) = -\|\mathbf{q} - \mathbf{t}\|_2
\]

This approach, introduced as **GQE** (Hamilton et al., 2018), works for path queries but has a critical limitation: there is no natural way to represent logical conjunction between two *point* embeddings. The average of two points gives a point, but the set of entities satisfying a conjunction is generally a *region* of the embedding space. Box embeddings resolve this by replacing points with axis-aligned hyper-rectangles whose size encodes the breadth of valid answers.

---

## 13.4 Box Embeddings: Query2Box

**Query2Box** (Ren et al., 2020) represents each query as a **hyper-rectangle** (axis-aligned box) in \( \mathbb{R}^d \):

\[
\text{Box}(\mathbf{c},\, \mathbf{o}) = \bigl\{\mathbf{x} \in \mathbb{R}^d :\ |x_i - c_i| \leq o_i\ \text{ for all } i \in [d]\bigr\}
\]

where \( \mathbf{c} \in \mathbb{R}^d \) is the center and \( \mathbf{o} \in \mathbb{R}^d_{\geq 0} \) is the offset vector (non-negative half-widths). Entities falling inside the box are predicted answers; entities outside are non-answers. The key intuition: the **offset** encodes query specificity — large offsets indicate many plausible answers (high uncertainty); small offsets indicate high specificity (narrow answer set). Conjunctions shrink the box; following a relation expands it.

### 13.4.1 Projection: Following a Relation

Starting from an anchor entity \( e_s \) (treated as a zero-offset box \( \text{Box}(\mathbf{e}_s, \mathbf{0}) \)), following a relation \( r \) shifts the center and expands the offset via learned parameters:

\[
\text{Proj}\!\left(\text{Box}(\mathbf{c}, \mathbf{o}),\, r\right) = \text{Box}\!\left(\mathbf{c} + \mathbf{r}_c,\ \text{softplus}(\mathbf{o} + \mathbf{r}_o)\right)
\]

where \( \mathbf{r}_c \in \mathbb{R}^d \) is the learned center shift and \( \mathbf{r}_o \in \mathbb{R}^d \) is the learned offset shift for relation \( r \). The softplus function \( \log(1 + e^x) \) ensures the offset remains non-negative while being differentiable everywhere. The offset grows with each hop because traversing a relation from a set of entities reaches a generally broader set of entities — uncertainty accumulates.

For a 2-hop path query \( (e_s, (r_1, r_2), ?) \):

\[
q_\text{box} = \text{Proj}\!\left(\text{Proj}\!\left(\text{Box}(\mathbf{e}_s, \mathbf{0}),\, r_1\right),\, r_2\right)
\]

### 13.4.2 Intersection: Logical AND

The conjunction of multiple boxes represents entities satisfying all conditions simultaneously. Given k query boxes \( \{(\mathbf{c}_1, \mathbf{o}_1), \ldots, (\mathbf{c}_k, \mathbf{o}_k)\} \), the intersection box is:

\[
\mathbf{c}_\text{int} = \sum_{j=1}^{k} w_j\, \mathbf{c}_j, \qquad w_j = \text{softmax}\!\left(f_\text{att}(\mathbf{c}_j)\right)_j
\]

\[
\mathbf{o}_\text{int} = \text{sigmoid}\!\left(f_\text{off}\!\left(\min_j \mathbf{o}_j,\; \bar{\mathbf{o}}\right)\right) \odot \min_j \mathbf{o}_j
\]

where \( f_\text{att} \) and \( f_\text{off} \) are learned MLPs, \( \bar{\mathbf{o}} = \frac{1}{k} \sum_j \mathbf{o}_j \) is the mean offset, and the element-wise minimum \( \min_j \mathbf{o}_j \) enforces that the intersection cannot be larger than the smallest input box — semantically correct because AND is more restrictive than any individual constraint. The sigmoid shrinkage further contracts the offset to reflect the narrowing of the feasible region.

### 13.4.3 Distance Function and Scoring

For a candidate answer entity \( \mathbf{e}_t \) and a query box \( (\mathbf{c}, \mathbf{o}) \), the box distance decomposes into an exterior distance (zero inside, positive outside) and a small interior smoothing term:

\[
d_\text{out}(\mathbf{e}_t, \mathbf{c}, \mathbf{o}) = \left\|\max\!\bigl(|\mathbf{e}_t - \mathbf{c}| - \mathbf{o},\; \mathbf{0}\bigr)\right\|_1
\]

\[
d_\text{in}(\mathbf{e}_t, \mathbf{c}, \mathbf{o}) = \left\|\mathbf{c} - \min\!\bigl(\mathbf{o},\; |\mathbf{e}_t - \mathbf{c}|\bigr)\right\|_1
\]

\[
d_\text{box}(\mathbf{e}_t, \mathbf{c}, \mathbf{o}) = d_\text{out}(\mathbf{e}_t, \mathbf{c}, \mathbf{o}) + \lambda \cdot d_\text{in}(\mathbf{e}_t, \mathbf{c}, \mathbf{o}), \quad \lambda = 0.02
\]

The outside distance is zero when the entity is inside the box; the inside distance provides gradient even for entities that technically satisfy the query, encouraging answers to cluster near the center. The plausibility score is:

\[
f(q, t) = -d_\text{box}(\mathbf{e}_t, \mathbf{c}_q, \mathbf{o}_q)
\]

!!! mascot-tip "Query2Box for Real-Time KG Question Answering"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage giving a tip about Query2Box in production">
    Query2Box's inference cost for a given query is \( O(|\mathcal{E}|) \) — compute the query box once in \( O(L) \) time (L hops), then evaluate \( d_\text{box} \) against all entity embeddings simultaneously via a vectorized operation. For a KG with 1 million entities at embedding dimension 400, this is a single matrix operation taking milliseconds on a GPU. This constant-time-per-entity property makes Query2Box practical for real-time question answering over large KGs — in contrast to graph traversal approaches whose cost grows with the number of valid intermediate paths, which can be exponential in path length.

### 13.4.4 Limitations: Disjunction and Negation

Query2Box handles conjunction (AND) via box intersection, but cannot cleanly represent **disjunction** (OR): the union of two boxes is not a box. The model approximates OR by evaluating each disjunct as a separate query and taking the maximum score across disjuncts — a valid but non-differentiable approximation at the query boundary.

**Negation** (NOT) is fundamentally incompatible with box geometry: the complement of a hyper-rectangle in \( \mathbb{R}^d \) is a non-convex union of \( 2d \) half-spaces, which cannot be compactly represented as another box.

!!! mascot-warning "Query2Box Cannot Handle Negation"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning about negation limitations">
    If your query workload includes NOT operators — "Find proteins associated with Disease X but NOT targeted by Drug Y" — Query2Box will produce incorrect results. The complement of a box has no compact box representation, so applying Query2Box's distance function to a negated condition is geometrically meaningless. The successor model BetaE (Ren & Leskovec, 2020) uses Beta distribution embeddings, whose complement distributions represent negation naturally. Always audit your query workload for negation before committing to Query2Box; for mixed AND/OR/NOT workloads, BetaE is the appropriate baseline.

---

## 13.5 Neural Bellman-Ford Networks

### 13.5.1 Algorithm to Architecture

The **Neural Bellman-Ford Network** (NBFNet, Zhu et al., 2021) approaches KG reasoning from a graph-algorithmic perspective: it generalizes the classical Bellman-Ford shortest-path algorithm to learn relational path reasoning through the graph structure.

In classical Bellman-Ford, given source node \( s \) with initial distance 0, the update rule is:

\[
d^{(t+1)}(v) = \min_{(u, v) \in E}\!\left\{d^{(t)}(u) + w(u,v)\right\}
\]

NBFNet replaces the scalar \( d(v) \in \mathbb{R} \) with a representation vector \( \mathbf{h}_v^{(t)} \in \mathbb{R}^d \), replaces the edge weight \( + w(u,v) \) with a learned relation-conditioned message function, and replaces the min aggregation with sum:

\[
\mathbf{h}_v^{(t+1)} = \sigma\!\left(\sum_{(u, r', v) \in \mathcal{E}} f_\theta\!\left(\mathbf{h}_u^{(t)},\, \mathbf{r}'\right) + \mathbf{h}_v^{(t)}\right)
\]

where the self-loop term \( \mathbf{h}_v^{(t)} \) plays the role of the identity message (preserving prior state across iterations), and \( f_\theta(\mathbf{h}_u, \mathbf{r}') \) is a learned function of neighbor state and edge relation type.

### 13.5.2 Query-Conditioned Initialization

The innovation that makes NBFNet a reasoning system rather than a generic GNN is its **query-conditioned initialization**. For a query \( (e_s, r_q, ?) \):

\[
\mathbf{h}_v^{(0)} = \begin{cases} \mathbf{r}_q & \text{if } v = e_s \\ \mathbf{0} & \text{otherwise} \end{cases}
\]

The source entity receives the query relation embedding as its initial representation; all others start at zero. As Bellman-Ford iterations proceed, the query-conditioned signal propagates from \( e_s \) through the relational graph, with each message operation incorporating the relation types along the traversed path. After \( T \) iterations, \( \mathbf{h}_v^{(T)} \) encodes the query-conditioned reachability of entity \( v \) from \( e_s \) via learned relational paths — integrating information from all paths of length up to \( T \) implicitly.

The final score for candidate answer \( t \) is:

\[
f(e_s, r_q, t) = \text{MLP}_\text{score}\!\left(\mathbf{h}_t^{(T)}\right)
\]

### 13.5.3 Message Function

Zhu et al. (2021) find that a DistMult-inspired query-conditioned message works best:

\[
f_\theta(\mathbf{h}_u^{(t)},\, r') = \mathbf{h}_u^{(t)} \odot \left(W_{r'}\, \mathbf{r}_q\right)
\]

where \( W_{r'} \) is a learned relation-specific matrix and \( \mathbf{r}_q \) is the query relation embedding. This formulation makes the message **query-dependent**: the same graph edge \( (u, r', v) \) contributes differently to different queries, because the message is modulated by the query relation \( r_q \). This is the crucial distinction from a standard heterogeneous GNN: NBFNet messages are not just a function of the edge type but of the edge type *conditioned on the specific query being answered*.

!!! mascot-encourage "NBFNet Is Just Bellman-Ford with Learned Weights"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage offering encouragement about NBFNet">
    Every component of NBFNet has a direct Bellman-Ford analogue: the initialization seeds reasoning from the query source; the iteration propagates query-conditioned information along edges; the message function computes edge-traversal relevance for the specific query; the scoring MLP reads off the final reachability estimate. If you've implemented message-passing GNNs, you've already implemented the hardest part of NBFNet — the only additions are the query-conditioned initialization and the relation-modulated message function. The conceptual leap is not architectural but interpretive: instead of computing node features, you are computing query-specific reachability representations.

### 13.5.4 NBFNet as a GNN

NBFNet is formally a GNN operating on the heterogeneous relational graph \( G = (\mathcal{E}, \mathcal{T}, \mathcal{R}) \). The T Bellman-Ford iterations are T message-passing layers; the edge messages are relation-type-dependent; the initialization is query-dependent rather than feature-dependent. All training techniques from Chapters 7–8 apply directly: mini-batch training by sampling query triples, dropout on message activations, layer normalization between iterations, and early stopping on validation MRR.

The key computational distinction from Query2Box is that NBFNet runs one forward pass *per query*, producing fresh representations for all entities conditioned on that specific query — whereas Query2Box runs one forward pass to train the entity and relation embeddings, then applies fixed embeddings at inference time. This makes NBFNet \( O(|\mathcal{T}| \cdot T) \) per query vs. Query2Box's \( O(|\mathcal{E}| \cdot d) \) per query — NBFNet is cheaper on sparse KGs, more expensive on dense ones.

---

## 13.6 Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Query2Box — Core Geometric Operators
# ══════════════════════════════════════════════════════════════════════════════

class BoxEmbedding:
    """A hyper-rectangle Box(c, o) with center c ∈ R^d and half-widths o ≥ 0."""

    def __init__(self, center: torch.Tensor, offset: torch.Tensor):
        self.center = center    # (..., d)
        self.offset = offset    # (..., d) — all entries ≥ 0

    def distance(self, entity_emb: torch.Tensor, lam: float = 0.02) -> torch.Tensor:
        """d_box = d_out + λ · d_in  (L1 norm over dimensions)."""
        diff  = torch.abs(entity_emb - self.center)               # (..., d)
        d_out = torch.norm(torch.clamp(diff - self.offset, min=0), p=1, dim=-1)
        d_in  = torch.norm(self.center - torch.min(self.offset, diff), p=1, dim=-1)
        return d_out + lam * d_in


class Query2Box(nn.Module):
    """
    Query2Box: Reasoning over KGs in Vector Space using Box Embeddings
    Ren et al., ICLR 2020.

    Entity embeddings: point vectors (zero-offset boxes).
    Relation parameters: center shift r_c, offset shift r_o.
    Supports query types: 1p, 2p, 3p, 2i, 3i, pi, ip.
    """

    def __init__(self, num_entities: int, num_relations: int, dim: int = 400):
        super().__init__()
        self.d = dim

        # Learnable parameters
        self.entity_emb  = nn.Embedding(num_entities,  dim)
        self.rel_center  = nn.Embedding(num_relations, dim)   # center shift per relation
        self.rel_offset  = nn.Embedding(num_relations, dim)   # offset shift per relation

        # Intersection MLPs
        self.inter_att = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.inter_off = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.Sigmoid()
        )

        nn.init.uniform_(self.entity_emb.weight,  -1.0, 1.0)
        nn.init.uniform_(self.rel_center.weight,  -1.0, 1.0)
        nn.init.uniform_(self.rel_offset.weight,   0.0, 1.0)

    # ── Geometric operators ────────────────────────────────────────────────

    def project(self, box: BoxEmbedding, rel: torch.Tensor) -> BoxEmbedding:
        """Follow relation rel from box: shift center, expand offset."""
        rc = self.rel_center(rel)                             # (B, d)
        ro = self.rel_offset(rel)                             # (B, d)
        return BoxEmbedding(
            center=box.center + rc,
            offset=F.softplus(box.offset + ro),              # ensures offset ≥ 0
        )

    def intersect(self, boxes: list) -> BoxEmbedding:
        """Logical AND: attention-weighted center + shrunk offset."""
        centers = torch.stack([b.center for b in boxes], dim=1)   # (B, k, d)
        offsets = torch.stack([b.offset for b in boxes], dim=1)   # (B, k, d)

        # Attention over centers
        attn = F.softmax(self.inter_att(centers), dim=1)           # (B, k, d)
        new_center = (attn * centers).sum(dim=1)                   # (B, d)

        # Offset: sigmoid-shrink of element-wise min
        min_off  = offsets.min(dim=1).values                       # (B, d)
        mean_off = offsets.mean(dim=1)                             # (B, d)
        new_offset = self.inter_off(
            torch.cat([min_off, mean_off], dim=-1)
        ) * min_off                                                # (B, d)

        return BoxEmbedding(new_center, new_offset)

    def anchor_box(self, entity_idx: torch.Tensor) -> BoxEmbedding:
        """Entity → zero-offset box (point embedding)."""
        emb = self.entity_emb(entity_idx)
        return BoxEmbedding(center=emb, offset=torch.zeros_like(emb))

    def score(self, query_box: BoxEmbedding, tail_idx: torch.Tensor) -> torch.Tensor:
        """f(q, t) = -d_box(e_t, query_box)."""
        return -query_box.distance(self.entity_emb(tail_idx))

    # ── Query-type forward methods ─────────────────────────────────────────

    def forward_1p(self, head, rel):
        """1-hop: (head, rel, ?)."""
        return self.project(self.anchor_box(head), rel)

    def forward_2p(self, head, r1, r2):
        """2-hop path: (head, (r1, r2), ?)."""
        return self.project(self.project(self.anchor_box(head), r1), r2)

    def forward_2i(self, h1, r1, h2, r2):
        """2-intersection: two 1-hop chains sharing the answer variable."""
        return self.intersect([
            self.forward_1p(h1, r1),
            self.forward_1p(h2, r2),
        ])

    def forward_pi(self, h1, r1, r2, h2, r3):
        """pi: 2-hop chain AND a 1-hop chain."""
        return self.intersect([
            self.forward_2p(h1, r1, r2),
            self.forward_1p(h2, r3),
        ])


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: NBFNet — Neural Bellman-Ford for KG Link Prediction
# ══════════════════════════════════════════════════════════════════════════════

class NBFNetLayer(MessagePassing):
    """
    One Bellman-Ford iteration:
      message f(h_u, r') = h_u ⊙ W_r' r_q  (DistMult-style, query-conditioned)
      aggregate: sum over incoming edges
      update: GRU recurrent step
    """

    def __init__(self, hidden_dim: int):
        super().__init__(aggr='add')
        self.W_rel = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gru   = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h, edge_index, edge_rel_emb, query_rel_emb):
        agg = self.propagate(
            edge_index, h=h,
            edge_rel_emb=edge_rel_emb,
            query_rel_emb=query_rel_emb,
        )
        return self.gru(agg, h)

    def message(self, h_j, edge_rel_emb, query_rel_emb_j):
        # h_j ⊙ W(r') applied to query relation — ties edge traversal to query
        return h_j * self.W_rel(edge_rel_emb) * query_rel_emb_j


class NBFNet(nn.Module):
    """
    Neural Bellman-Ford Network (Zhu et al., NeurIPS 2021).

    Forward pass for query (src, rel_q, ?):
      1. Initialize h_src = r_q, h_other = 0
      2. Run T Bellman-Ford layers (message-passing with relation-conditioned messages)
      3. Score all entities via MLP(h_v^(T))
    """

    def __init__(
        self,
        num_entities:  int,
        num_relations: int,
        hidden_dim:    int = 64,
        num_layers:    int = 6,
    ):
        super().__init__()
        self.num_entities  = num_entities
        self.hidden_dim    = hidden_dim

        self.relation_emb = nn.Embedding(num_relations, hidden_dim)
        self.layers = nn.ModuleList([
            NBFNetLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_idx, rel_idx, edge_index, edge_type):
        """
        Returns scores: (num_entities,) — higher = more likely answer.
        src_idx: int (query head entity)
        rel_idx: scalar tensor (query relation)
        """
        N = self.num_entities
        device = edge_index.device

        # Relation embeddings for each edge and for the query
        edge_rel_emb  = self.relation_emb(edge_type)          # (E, d)
        q_rel         = self.relation_emb(rel_idx)             # (d,)
        query_rel_all = q_rel.unsqueeze(0).expand(N, -1)       # (N, d)

        # Query-conditioned initialization: source gets query relation embedding
        h = torch.zeros(N, self.hidden_dim, device=device)
        h[src_idx] = q_rel

        # T Bellman-Ford iterations
        for layer in self.layers:
            h = layer(h, edge_index, edge_rel_emb, query_rel_all)

        return self.score_mlp(h).squeeze(-1)                   # (N,)


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Training NBFNet on a Toy KG
# ══════════════════════════════════════════════════════════════════════════════

def train_nbfnet_demo():
    """Minimal training loop demonstrating NBFNet on 6 entities, 3 relations."""
    torch.manual_seed(0)

    # Toy KG: 6 entities, 3 relations
    # (0)-r0->(1)-r1->(4)   [2-hop: 0→1→4]
    # (0)-r0->(2)-r1->(5)   [2-hop: 0→2→5]
    # (3)-r2->(1)
    edges_src  = torch.tensor([0, 0, 1, 2, 3, 1, 4, 5])
    edges_tgt  = torch.tensor([1, 2, 4, 5, 1, 0, 1, 2])
    edge_types = torch.tensor([0, 0, 1, 1, 2, 0, 1, 1])
    edge_index = torch.stack([edges_src, edges_tgt])

    model     = NBFNet(num_entities=6, num_relations=3, hidden_dim=32, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Positive triples to train on
    pos_triples = [(0, 0, 1), (0, 0, 2), (1, 1, 4), (2, 1, 5)]

    for epoch in range(300):
        optimizer.zero_grad()
        loss = torch.tensor(0.0)
        for src, rel, tgt in pos_triples:
            scores = model(src, torch.tensor(rel), edge_index, edge_types)
            labels = torch.zeros(6)
            labels[tgt] = 1.0
            loss = loss + F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    # Evaluate: query (entity 0, relation 0, ?)
    with torch.no_grad():
        scores = model(0, torch.tensor(0), edge_index, edge_types)
        top_k  = torch.topk(scores, 3).indices.tolist()
        print(f"\nQuery (0, r0, ?) — top-3 answers: {top_k}")
        print(f"Expected answers: entities 1 and 2 (both at rank 1 or 2)")


if __name__ == '__main__':
    train_nbfnet_demo()
```

---

## 13.7 Benchmark Results

Multi-hop query answering is evaluated on FB15k-237 using the EPFO query shapes defined in §13.2.2. The table reports MRR (mean reciprocal rank, filtered, higher is better) for each query type.

| Model | 1p | 2p | 3p | 2i | 3i | pi | ip | Year |
|---|---|---|---|---|---|---|---|---|
| GQE (point) | 35.0 | 7.2 | 5.3 | 23.3 | 34.6 | 16.5 | 10.7 | 2018 |
| Query2Box | 40.6 | 14.2 | 11.6 | 29.8 | 42.7 | 22.9 | 16.5 | 2020 |
| BetaE (+negation) | 39.0 | 10.9 | 10.0 | 28.8 | 42.5 | 22.4 | 12.6 | 2020 |
| CQD | 46.7 | 18.1 | 14.0 | 35.3 | 49.8 | 26.0 | 21.7 | 2021 |
| **NBFNet** | **55.0** | — | — | — | — | — | — | 2021 |

Sources: Hamilton et al. (2018), Ren et al. (2020), Arakelyan et al. (2021), Zhu et al. (2021).

Three patterns stand out. First, Query2Box substantially outperforms GQE on all conjunctive query types (2i, 3i, pi, ip), confirming that geometric box regions capture logical AND far more effectively than point averaging. Second, BetaE slightly underperforms Query2Box on 1p and 2p queries despite its theoretical advantage on negation — its probabilistic parameterization adds optimization complexity that hurts on non-negation queries. Third, NBFNet achieves the highest 1p MRR by a substantial margin (55.0 vs. 46.7 for CQD), demonstrating that graph-structure-aware propagation extracts substantially more signal from KG topology than embedding lookup alone. NBFNet's absence from multi-hop columns reflects its quadratic entity-pair scaling for those query types, which remains an open engineering challenge.

---

## 13.8 MicroSim: Query Box Traversal

<iframe src="../../sims/ch13-kg-query-traversal/main.html" width="100%" height="522px" scrolling="no"></iframe>
[Run Query2Box Multi-Hop Traversal Fullscreen](../../sims/ch13-kg-query-traversal/main.html)

---

## 13.9 Common Pitfalls

**Applying Query2Box to negated conditions.** Query2Box's box geometry cannot represent the complement of a box. If your query workload includes NOT operators, apply BetaE or a dedicated negation-capable model; never use Query2Box's distance function on negated conditions, as the geometric semantics break down entirely and the resulting scores are arbitrary.

**Training on only one query type and expecting multi-type generalization.** The intersection operator's MLPs (\( f_\text{att} \) and \( f_\text{off} \)) have no gradient signal unless training examples include conjunctive queries. A model trained solely on 1p queries will have randomly initialized intersection parameters; when queried with 2i or 3i queries at inference, results are meaningless. Always include training examples of every query type you intend to support at inference time.

**Using naive binary cross-entropy without positive reweighting in NBFNet.** A single query \( (e_s, r, ?) \) has one or a few positive answers out of \( |\mathcal{E}| \) entities. Naive BCE with uniform weights produces models that predict near-zero for all entities (a trivially low-loss solution for imbalanced classification). Use a loss that explicitly rewards high scores for positive answers: margin-based losses, focal loss, or positive reweighting with a factor proportional to \( |\mathcal{E}| / n_{\text{pos}} \).

**Conflating query embedding inference cost with NBFNet inference cost.** Query2Box scores all entities against a query box in one vectorized operation — \( O(|\mathcal{E}|) \). NBFNet requires T message-passing iterations over the full KG edge set per query — \( O(T \cdot |\mathcal{T}|) \). For KGs with 100M+ triples, NBFNet's per-query cost becomes substantial. Check your inference latency requirements before choosing NBFNet for production deployment; for latency-sensitive systems, Query2Box's constant-time scoring is often preferred.

**Using the unfiltered evaluation protocol.** Just as in single-triple link prediction (Chapter 12), multi-hop query evaluation must filter out known-correct triples from the candidate ranking. On complex query types (2i, pi, ip), multiple entities can simultaneously satisfy the query conditions; all of them must be filtered before computing the rank of any specific test answer. Evaluating without filtering inflates apparent difficulty and makes models look systematically worse than they are.

---

## 13.10 Exercises

**Remember**

1. Define a 2i (2-chain intersection) query using formal notation. Write out the biomedical example from §13.1 as a 2i query, identifying the anchor entities, relation types, and intermediate variables.

2. State the Query2Box projection operator and explain in one sentence why the offset uses \( \text{softplus} \) rather than a direct sum with the learned offset shift.

**Understand**

3. Derive the expected recall of multi-hop graph traversal on an incomplete KG. If a KG has edge completeness rate \( p \) (each true edge is present independently with probability \( p \)), show that an L-hop path query has expected recall \( p^L \). Compute the expected recall for \( p = 0.7 \) at \( L = 1, 2, 3, 4, 5 \) and explain why this motivates embedding-based query answering.

4. Explain why logical OR cannot be represented by a box intersection in Query2Box, and why logical NOT cannot be represented by any box. Are there other geometric primitives (e.g., ellipsoids, convex hulls, polytopes) that would better support disjunction? What tradeoffs would they introduce?

**Apply**

5. For a 3i query (three 1-hop chains sharing an answer variable), write out the full Query2Box computation using the operators from §13.4. Then implement a `forward_3i(h1, r1, h2, r2, h3, r3)` method for the `Query2Box` class in §13.6, and verify that it produces a box with smaller offset than any single-chain box.

6. Trace through two NBFNet Bellman-Ford iterations for the query \( (\text{entity 0},\ r_0,\ ?) \) on the toy KG in §13.6 (8 edges, 6 entities). Show \( \mathbf{h}^{(0)} \), \( \mathbf{h}^{(1)} \), and \( \mathbf{h}^{(2)} \) for entities 0, 1, and 2. Which entity should have the highest score at \( \mathbf{h}^{(2)} \) and why?

**Analyze**

7. Compare the asymptotic inference complexities of Query2Box, GQE, and NBFNet for answering a 3-hop path query (3p) over a KG with \( |\mathcal{E}| \) entities, \( |\mathcal{T}| \) triples, embedding dimension \( d \), and \( T = 6 \) Bellman-Ford layers. Under what conditions (in terms of the ratio \( |\mathcal{T}| / |\mathcal{E}| \)) is NBFNet cheaper than Query2Box per query?

8. The benchmark table shows CQD achieving higher MRR than Query2Box on all complex query types. CQD decomposes queries into individual triple scores and combines them via continuous t-norm operators (product t-norm for AND, probabilistic sum for OR). Analyze the key architectural difference between CQD and Query2Box: which model has more inductive bias for logical structure, and which is more dependent on the quality of the underlying entity embeddings? Predict which model would generalize better to query types unseen during training.

**Evaluate**

9. A practitioner has a biomedical KG with 500K entities and 50M triples. They need to answer complex conjunctive queries (up to 4 chains, 3 hops each) in under 100ms per query at inference time. Evaluate the feasibility of using (a) Query2Box and (b) NBFNet for this deployment. Estimate the approximate inference time for each approach (provide your assumptions), and recommend a deployment strategy.

10. The NBFNet paper (Zhu et al. 2021) reports that 6 Bellman-Ford iterations (T=6) is sufficient to achieve near-optimal performance on FB15k-237. Evaluate this claim: what theoretical property of Bellman-Ford guarantees convergence in finite iterations on a finite graph, and how does that translate to NBFNet's learned version? Is T=6 likely to be sufficient for a KG with diameter much larger than 6 (e.g., a protein interaction network with diameter 20+)?

**Create**

11. Design a unified architecture, BoxBFNet, that combines Query2Box's box geometry with NBFNet's graph propagation. Specifically: (a) define box-valued node representations \( \text{Box}(\mathbf{c}_v, \mathbf{o}_v) \); (b) define a box-valued message function for edge propagation; (c) define aggregation of multiple incoming box messages using the Query2Box intersection operator; and (d) define the final scoring function for answer entities. Analyze which EPFO query types your architecture would support and what its computational complexity would be.

12. Temporal KG queries add a time dimension: triples \( (h, r, t, \tau) \) are valid only at timestamp \( \tau \), and multi-hop queries may span different time points (e.g., "Who was the president of France in 1990 and also served as a minister before 1985?"). Propose an extension of the Query2Box framework to handle temporal conjunctive queries: define a temporal box \( \text{TBox}(\mathbf{c}, \mathbf{o}, [t_\text{start}, t_\text{end}]) \), specify how projection and intersection handle the time interval, and discuss what new failure modes arise when the time intervals of two chains in a 2i query do not overlap.

---

## 13.11 Further Reading

**Hamilton, W. L., et al. (2018). Embedding Logical Queries on Knowledge Graphs.** *NeurIPS 2018.* Introduced GQE and the canonical EPFO query benchmark (1p, 2p, 3p, 2i, 3i, pi, ip) that all subsequent multi-hop KG reasoning papers use for evaluation. Although GQE is now outperformed on every metric, the benchmark design and evaluation protocol remain the field standard. [arXiv:1806.01445](https://arxiv.org/abs/1806.01445)

**Ren, H., et al. (2020). Query2Box: Reasoning over Knowledge Graphs in Vector Space using Box Embeddings.** *ICLR 2020.* Introduced box embeddings for conjunctive KG query answering. The learnable intersection operator is the key contribution — demonstrating that geometric AND has a natural box-intersection implementation that outperforms point averaging on all conjunctive query types. [arXiv:2002.05969](https://arxiv.org/abs/2002.05969)

**Ren, H., & Leskovec, J. (2020). Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs.** *NeurIPS 2020.* Extended the query embedding paradigm to handle negation using Beta distribution embeddings, where complement distributions represent NOT naturally. Essential reference for applications requiring negation queries. [arXiv:2010.11465](https://arxiv.org/abs/2010.11465)

**Zhu, Z., et al. (2021). Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction.** *NeurIPS 2021.* Introduced NBFNet, achieving new state-of-the-art on FB15k-237 and WN18RR. The key insight — query-conditioned initialization enables Bellman-Ford-style path reasoning through the KG — is theoretically grounded and leads to a remarkably clean and effective implementation. [arXiv:2106.06935](https://arxiv.org/abs/2106.06935)

**Arakelyan, E., et al. (2021). Complex Query Answering with Neural Link Predictors.** *ICLR 2021.* Introduced CQD, which decomposes complex queries into triple scores and combines via continuous t-norm logic. Outperforms Query2Box on most query types while requiring no specialized geometric operators — an important baseline for the box embedding approach. [arXiv:2011.03459](https://arxiv.org/abs/2011.03459)

**Sadeghian, A., et al. (2019). DRUM: End-To-End Differentiable Rule Mining over Knowledge Graphs.** *NeurIPS 2019.* Represents the rule-learning approach to multi-hop reasoning, learning differentiable soft logical rules for KG completion. Complementary to the embedding-based approaches in this chapter, particularly relevant where interpretability of the reasoning chain is required. [arXiv:1909.03834](https://arxiv.org/abs/1909.03834)

**Zhang, Z., et al. (2022). Knowledge Graph Reasoning with Relational Digraph.** *WWW 2022.* Proposes a graph-theoretic formulation of multi-hop KG reasoning using relational directed graphs, unifying path-based and embedding-based approaches under a common framework. Provides complementary theoretical perspective on when and why graph-propagation methods like NBFNet outperform entity-level embedding approaches. [arXiv:2108.06040](https://arxiv.org/abs/2108.06040)

---

!!! mascot-celebration "Chapter 13 Complete — From Triples to Multi-Hop Reasoning"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating Chapter 13 completion">
    You've traversed the full arc from individual triple scoring to complex multi-hop conjunctive reasoning. The key design principles: use box embeddings when you need to answer conjunctive queries at scale with constant per-query inference cost; use NBFNet when single-hop link prediction accuracy is the priority and you can afford per-query graph propagation; use BetaE when your query workload includes negation. Chapter 14 takes one more step — asking whether KG reasoning can be made truly universal, transferring across entirely different knowledge graphs without retraining, via foundation models for KGs.

[See Annotated References](./references.md)
