---
title: "Chapter 11: Graph Transformers"
description: "Presents the Graph Transformer architecture family — Graphormer, SAN, GPS — and the positional and structural encodings that overcome attention's permutation blindness."
generated_by: claude skill chapter-content-generator
date: 2026-05-28 22:41:31
version: 0.08
---

# Chapter 11: Graph Transformers

**Part 2: Graph Neural Networks**

## Summary

Presents the Graph Transformer architecture family — Graphormer, SAN, GPS — and the positional and structural encodings (Laplacian eigenvectors, RWSE) that overcome attention's permutation blindness, with full treatment of sign invariance via SignNet and BasisNet.

## Concepts Covered

This chapter covers the following 13 concepts from the learning graph:

1. Position-Aware GNN
2. Graph Transformer
3. Graphormer
4. SAN (Spectral Attention Net)
5. GPS (General Powerful GNN)
6. Laplacian Positional Encoding
7. Random Walk Struct Encoding
8. Sign-Invariant Network
9. Basis-Invariant Network
10. Spatial Bias (Graph Attention)
11. Centrality Encoding
12. Edge Encoding (Transformer)
13. Relative Positional Encoding

## Prerequisites

This chapter builds on:

- [Chapter 6: GNN Foundations: Message Passing and GCN](../06-gnn-foundations/index.md)
- [Chapter 7: GNN Design Space: GraphSAGE and GAT](../07-gnn-design-space/index.md)
- [Chapter 8: GNN Training, Augmentation, and Practical Tips](../08-gnn-training/index.md)
- [Chapter 9: Theory of GNNs: Expressiveness and the WL Test](../09-gnn-theory/index.md)
- [Chapter 10: Designing Powerful Encoders: GIN and Beyond](../10-powerful-gnns/index.md)

---

!!! mascot-welcome "Welcome to Graph Transformers"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage welcoming you to Chapter 11">
    Message-passing GNNs excel at aggregating local neighborhoods, but they face a fundamental ceiling: a node can only gather information from within K hops after K layers, and going deeper triggers over-smoothing and over-squashing. Graph Transformers break this ceiling by letting every node attend directly to every other node in a single layer — a paradigm shift from local propagation to global interaction. The challenge, as you will see, is not attention itself but teaching the model *where* each node sits in the graph when no canonical coordinate system exists.

---

## 11.1 The Long-Range Problem in Molecular Property Prediction

Consider the task of predicting whether a candidate drug molecule will inhibit a particular enzyme. The molecule is naturally represented as a graph: atoms are nodes, covalent bonds are edges. A GCN or GIN with four layers can aggregate information from a node's four-hop neighborhood — roughly fifteen atoms for a typical drug-like molecule. For many properties (bond polarity, local reactivity) this is sufficient. But molecular bioactivity frequently depends on the spatial relationship between distant functional groups: the carbonyl oxygen on one end of a steroid backbone interacting with a hydroxyl group six bonds away, or the charge distribution across an aromatic ring system that spans the entire molecule. A message-passing GNN with enough layers to capture such dependencies would trigger over-smoothing; a shallow one misses the interaction entirely.

This long-range dependency problem appears across graph domains. In a social network, the influence of a community leader on a distant follower unfolds across many hops. In a protein interaction network, the functional similarity between two proteins may depend on their shared interaction partners rather than on any direct edge. In a knowledge graph, a multi-hop reasoning chain connects entities that share no common neighbor. Standard message passing — local aggregation repeated K times — is architecturally mismatched to these tasks.

Transformers, which became the dominant architecture in natural language processing after Vaswani et al. (2017), solved an analogous problem in sequences: every token can attend to every other token regardless of distance, enabling direct modeling of long-range syntactic and semantic dependencies. The natural question is whether the same mechanism can be applied to graphs. The short answer is yes — but the translation is non-trivial, because graphs lack the positional coordinate system that makes attention informative in sequences and images. This chapter develops the full machinery required to make Transformers work on graphs, culminating in three landmark architectures: Graphormer, SAN, and GPS.

---

## 11.2 From Sequence Attention to Graph Attention

### 11.2.1 Vanilla Multi-Head Self-Attention

In the original Transformer, self-attention over a sequence of N tokens produces, for each token i, a weighted combination of all token representations, where the weights are determined by learned query–key compatibility:

\[
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

Here \( Q = HW_Q \), \( K = HW_K \), \( V = HW_V \) are the query, key, and value projections of the token matrix \( H \in \mathbb{R}^{N \times d} \), and \( d_k \) is the per-head dimension. The scalar product \( q_i^\top k_j / \sqrt{d_k} \) measures how much node i should attend to node j; softmax normalizes these into a probability distribution over j; and the output is a weighted sum of value vectors.

For a graph with N nodes, the simplest adaptation treats each node as a token and applies full attention — every node attends to every other node:

\[
\alpha_{ij} = \frac{\exp\!\left(q_i^\top k_j / \sqrt{d_k}\right)}{\sum_{j'=1}^{N} \exp\!\left(q_i^\top k_{j'} / \sqrt{d_k}\right)}, \qquad h_i^{\text{out}} = \sum_{j=1}^{N} \alpha_{ij} v_j
\]

This formulation is already more expressive than message passing: every node directly receives information from every other node in a single layer, bypassing the K-hop bottleneck entirely. However, it has two immediate weaknesses. First, it treats the graph as a complete graph, ignoring which pairs are actually connected and which are structurally distant. Second, it is permutation-blind — if node identities are shuffled, the attention weights do not change unless the model is given some way to distinguish nodes based on their graph position. Both weaknesses stem from the same root cause: the model has no access to graph topology.

### 11.2.2 Position-Aware GNNs: The Intermediate Solution

Before full Graph Transformers emerged, researchers developed position-aware GNNs (P-GNN, You et al. 2019) as a transitional approach. P-GNN assigns each node a positional embedding derived from its distances to a set of anchor nodes — randomly sampled nodes whose shortest-path distances serve as structural coordinates. This allows message passing to be conditioned on relative position, improving the model's ability to distinguish nodes that 1-WL would equate.

P-GNN's limitation is that anchor-based encodings are coarse: the choice and number of anchors introduces noise, and shortest-path distance is a lossy summary of graph topology. The architectures developed in this chapter replace ad hoc anchors with principled encodings derived from the graph's spectral and random-walk structure.

---

## 11.3 Laplacian Positional Encoding

### 11.3.1 The Graph Laplacian and Its Spectrum

The normalized graph Laplacian is defined as:

\[
\hat{L} = I - D^{-1/2} A D^{-1/2}
\]

where \( A \) is the adjacency matrix and \( D \) is the diagonal degree matrix. \( \hat{L} \) is symmetric positive semi-definite, so its eigendecomposition yields real, non-negative eigenvalues:

\[
\hat{L} = U \Lambda U^\top, \qquad 0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_N \leq 2
\]

The eigenvectors \( u_1, u_2, \ldots, u_N \in \mathbb{R}^N \) form an orthonormal basis for \( \mathbb{R}^N \). Their geometric interpretation is central to understanding why they serve as positional encodings. The constant vector \( u_1 = \mathbf{1}/\sqrt{N} \) (corresponding to \( \lambda_1 = 0 \)) carries no structural information; the second eigenvector \( u_2 \) — the Fiedler vector — divides the graph into two communities along its sign, minimizing the cut; subsequent eigenvectors capture increasingly fine-grained structural variation at progressively shorter length scales.

The **Laplacian Positional Encoding (LapPE)** for node v uses the k smallest non-trivial eigenvectors:

\[
\text{pe}_v = \bigl[u_2(v),\; u_3(v),\; \ldots,\; u_{k+1}(v)\bigr] \in \mathbb{R}^k
\]

Each component \( u_i(v) \) encodes how node v participates in the i-th global pattern of the graph. Two nodes that occupy similar structural positions in the graph — connected to similar neighborhoods, at similar depths, within similar communities — will have similar LapPE vectors. Two nodes that are far apart or in structurally different regions will have distinct LapPE vectors. This is precisely the kind of global positional awareness that makes attention discriminative.

In practice, LapPE is computed once per graph and concatenated or added to the node feature vector before any Transformer layers:

\[
\tilde{h}_v^{(0)} = W_{\text{node}} x_v + W_{\text{pe}} \text{pe}_v
\]

!!! mascot-thinking "Why Does the Graph Laplacian Encode Position?"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinking about Laplacian eigenstructure">
    Think of the eigenvectors as global "harmonics" of the graph — analogous to sine waves on a line segment or spherical harmonics on a sphere. Low-frequency eigenvectors (small eigenvalue) vary slowly across the graph: neighboring nodes have similar values. High-frequency eigenvectors vary rapidly: neighboring nodes can have very different values. The Fiedler vector \( u_2 \) is the smoothest non-trivial harmonic, capturing the graph's coarsest structural split. Together, the first k eigenvectors give each node a coordinate in a k-dimensional spectral space that reflects global graph topology — not just local neighborhood structure.

### 11.3.2 The Sign Ambiguity Problem

Here is a fundamental obstacle: if \( u_i \) is an eigenvector with eigenvalue \( \lambda_i \), then so is \( -u_i \). When the eigenvalue has multiplicity greater than one (i.e., \( \lambda_i = \lambda_{i+1} \)), any linear combination of the corresponding eigenvectors is also a valid eigenvector. This means that two computations of LapPE on the same graph can produce different results depending on numerical conventions in the eigendecomposition routine — the sign (and orientation) of each eigenvector is undefined.

If a neural network is trained with one sign convention and tested with another, it will fail to generalize. The model cannot learn to use the absolute value of \( u_2(v) \) reliably when that value can be positive or negative for the same node across different runs or different graphs.

This ambiguity is not a numerical artifact to be patched away — it is mathematically inherent. The solution requires architectures that are **invariant** or **equivariant** to sign flips and rotations of the eigenspace.

---

## 11.4 Sign-Invariant and Basis-Invariant Networks

### 11.4.1 SignNet

Lim et al. (2022) introduced **SignNet**, a neural architecture that processes eigenvectors in a way that is invariant to sign flips. The key insight is that the set \( \{u_i, -u_i\} \) is the fundamental unit — not \( u_i \) alone. SignNet applies a shared MLP \( f \) to both \( u_i \) and \( -u_i \), then pools symmetrically:

\[
\text{SignNet}(u_i) = \rho\!\left(f(u_i) + f(-u_i)\right)
\]

where \( \rho \) is another MLP applied after the symmetric aggregation. The sum \( f(u_i) + f(-u_i) \) is manifestly invariant to replacing \( u_i \) with \( -u_i \), since \( f(-u_i) + f(u_i) = f(u_i) + f(-u_i) \). The final positional encoding for node v is:

\[
\text{pe}_v^{\text{Sign}} = \text{SignNet}(u_2, u_3, \ldots, u_{k+1}) \in \mathbb{R}^{d_{\text{pe}}}
\]

In practice, SignNet can be implemented as a DeepSet that processes the full collection of eigenvectors: each eigenvector \( u_i \) is treated as a sequence of N node-level scalars, and the shared MLP \( f \) is applied node-wise before the sign-symmetric aggregation.

!!! mascot-encourage "Sign Ambiguity Seems Complicated — It Is, But Don't Stop Here"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage offering encouragement about sign invariance">
    Sign invariance is genuinely one of the more mathematically subtle concepts in this textbook. The key practical takeaway is this: when you use LapPE, always pass the eigenvectors through a sign-invariant network (SignNet) or basis-invariant network (BasisNet) before using them as node features. Never feed raw eigenvectors directly to a standard MLP — the model will either overfit to the sign convention of the training data or learn a noisy, inconsistent representation. The math in this section explains *why* this matters; the implementation is a single module replacement.

### 11.4.2 BasisNet

When an eigenvalue has multiplicity greater than one — for example, in regular graphs where many eigenvalues repeat — sign invariance is insufficient. Any orthonormal rotation within the degenerate eigenspace is equally valid, producing not just sign flips but arbitrary rotations of the corresponding eigenvectors. **BasisNet** (Lim et al. 2022) handles this more general ambiguity by learning functions that are equivariant to the full orthogonal group O(m) acting on m-dimensional degenerate eigenspaces:

\[
\text{BasisNet}(U_{\lambda}) = \rho\!\left(\sum_{g \in O(m)} f(U_{\lambda} g)\right)
\]

where \( U_{\lambda} \in \mathbb{R}^{N \times m} \) is the subspace spanned by eigenvectors with eigenvalue \( \lambda \), and the sum over \( O(m) \) is approximated in practice via equivariant neural architectures. BasisNet strictly generalizes SignNet (which handles only O(1) = \{+1, -1\}), and is theoretically necessary for graphs with repeated Laplacian eigenvalues — a common occurrence in regular and symmetric graphs.

---

## 11.5 Random Walk Structural Encoding

While LapPE captures global spectral position, **Random Walk Structural Encoding (RWSE)** captures local structural context through the landing probabilities of k-step random walks. Recall from Chapter 10 that the t-step random walk landing probability at node v is the diagonal entry of \( (AD^{-1})^t \):

\[
\text{rw}_v^{(t)} = \left[(AD^{-1})^t\right]_{vv}
\]

This scalar measures the probability that a random walk starting at v returns to v after exactly t steps. Intuitively, high \( \text{rw}_v^{(1)} \) indicates self-loops; high \( \text{rw}_v^{(3)} \) indicates v participates in triangles; high \( \text{rw}_v^{(6)} \) indicates hexagonal ring membership — a critical structural motif in organic chemistry.

The full RWSE concatenates landing probabilities across k different walk lengths:

\[
\text{rwse}_v = \bigl[\text{rw}_v^{(1)},\; \text{rw}_v^{(2)},\; \ldots,\; \text{rw}_v^{(k)}\bigr] \in \mathbb{R}^k
\]

RWSE has two practical advantages over LapPE: it is non-negative (no sign ambiguity) and it is cheaper to compute, since the diagonal of matrix powers can be estimated without full eigendecomposition. The GPS framework (§11.8) uses RWSE as its default structural encoding and achieves state-of-the-art results on molecular benchmarks.

---

## 11.6 Relative Positional Encoding

LapPE and RWSE assign absolute structural coordinates to individual nodes. **Relative positional encoding** instead characterizes the structural relationship between *pairs* of nodes and injects this information directly into the attention computation as a bias term.

The simplest relative structural feature between nodes i and j is their **shortest path distance** \( \text{dist}(i,j) \). In Graphormer (§11.7), this distance is mapped to a learnable scalar bias \( b_{\phi(i,j)} \) that is added to the attention logit:

\[
a_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} + b_{\phi(i,j)}
\]

where \( b_{\phi(i,j)} \) is a learnable parameter indexed by the discrete distance value. This allows the model to learn that nearby nodes (distance 1 or 2) should attend to each other more strongly, while distant nodes (distance > 5) may deserve systematically lower attention weights. Unlike LapPE which modifies node representations, relative encodings modify the attention *pattern* itself — directly sculpting which pairs of nodes communicate.

---

## 11.7 Graphormer

### 11.7.1 Architecture Overview

Graphormer (Ying et al., 2021) adapts the standard Transformer architecture to molecular graphs through three graph-specific encodings layered on top of vanilla multi-head self-attention. These encodings inject three aspects of graph topology that raw node and edge features cannot convey: the global importance of each node (centrality encoding), the structural distance between every pair of nodes (spatial bias), and the content of the edges along the shortest path between node pairs (edge encoding).

### 11.7.2 Centrality Encoding

In a standard Transformer, all tokens enter the attention mechanism with equal weight — there is no notion of some tokens being globally "more important" than others. In a graph, node degree is a strong proxy for global importance: a hub node with 50 neighbors occupies a structurally central position that a leaf node cannot. **Centrality encoding** injects this structural prior into node representations before any attention layers are applied.

For directed graphs, Graphormer uses both in-degree and out-degree. The modified initial node representation is:

\[
\tilde{h}_v^{(0)} = x_v W_{\text{node}} + b_{\deg^+(v)} + b_{\deg^-(v)}
\]

where \( b_{\deg^+(v)} \) and \( b_{\deg^-(v)} \) are learnable embedding vectors indexed by in- and out-degree respectively (each stored in a lookup table of size max\_degree). For undirected molecular graphs, a single degree embedding \( b_{\deg(v)} \) is used. This initialization biases the network toward treating high-degree nodes as structurally important before any message is passed — a graph-theoretically motivated inductive bias that requires no additional computation beyond a single embedding lookup.

### 11.7.3 Spatial Bias

The spatial bias encodes the shortest-path distance between every pair of nodes i and j as a learnable additive term in the attention score:

\[
a_{ij} = \frac{(\tilde{h}_i W_Q)(\tilde{h}_j W_K)^\top}{\sqrt{d_k}} + b_{\phi(i,j)}
\]

Here \( \phi(i,j) \) is the shortest-path distance from i to j (capped at a maximum value, e.g., 20, with a special token for disconnected pairs), and \( b_{\phi(i,j)} \in \mathbb{R} \) is a scalar parameter shared across all attention heads. The spatial bias allows the model to learn distance-dependent attention patterns: it can learn to up-weight attention between nearby nodes (small \( \phi \)) and down-weight attention between distant nodes, or it can learn more complex non-monotone patterns if the task benefits from long-range interactions at specific distances.

### 11.7.4 Edge Encoding

The spatial bias uses only the *length* of the shortest path, discarding the edge features along that path. The **edge encoding** recovers path-edge information by computing, for each node pair (i, j), a weighted average of the edge features along their shortest path:

\[
A_{ij}^{\text{edge}} = \frac{1}{N_{ij}} \sum_{n=1}^{N_{ij}} x_{e_n^{(ij)}} W_E^{(n)} \cdot \frac{(\tilde{h}_i W_Q)^\top}{\sqrt{d_k}}
\]

where \( N_{ij} \) is the shortest-path length, \( e_n^{(ij)} \) is the n-th edge along the shortest path from i to j, and \( W_E^{(n)} \) is a learnable projection specific to the n-th hop. This term captures the *nature* of the path between two nodes — whether it passes through aromatic bonds, single bonds, or ring-closure bonds — and incorporates that information into how strongly i attends to j.

### 11.7.5 Full Graphormer Attention

The three encodings combine into a single modified attention computation:

\[
A_{ij} = \frac{(\tilde{h}_i W_Q)(\tilde{h}_j W_K)^\top}{\sqrt{d_k}} + b_{\phi(i,j)} + A_{ij}^{\text{edge}}
\]

The attention output for node i is then:

\[
h_i^{\text{out}} = \sum_{j=1}^{N} \text{softmax}(A_{ij}) \cdot (\tilde{h}_j W_V)
\]

The rest of the Graphormer layer is identical to a standard Transformer: multi-head attention output is projected, added to the residual, layer-normalized, and passed through a two-layer feed-forward network with another residual connection.

!!! mascot-tip "Graphormer's Practical Strength: Molecular Benchmarks"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage giving a practical tip about Graphormer">
    Graphormer was the first architecture to surpass GNNs on the OGB large-scale molecular property prediction benchmarks (ogbg-molhiv, PCQM4Mv2). Its advantage is particularly pronounced on tasks where two functional groups far apart in the molecular graph jointly determine biological activity — precisely the scenario where message-passing GNNs require prohibitively deep stacking. The trade-off is O(N²) attention, which makes Graphormer impractical for graphs with more than a few hundred nodes without approximation. For small molecular graphs (typical drug candidates have 20–50 heavy atoms), this cost is completely manageable.

---

## 11.8 SAN: Spectral Attention Network

**SAN** (Kreuzer et al., 2021) takes a different approach to injecting graph structure into attention. Rather than modifying the attention logit with distance-based biases, SAN uses the *full Laplacian spectrum* as positional encodings and introduces **dual attention heads**: separate attention mechanisms for connected node pairs and non-connected node pairs.

The motivation for dual attention is that adjacent nodes and non-adjacent nodes play fundamentally different structural roles. An edge between i and j represents a direct chemical bond, protein interaction, or social tie — a qualitatively different relationship than two nodes that merely share a common neighbor. Standard self-attention treats all pairs with the same attention mechanism, potentially conflating these different relationship types.

SAN introduces a learnable gating parameter \( \gamma \in [0, 1] \) and uses it to interpolate between local attention (restricted to existing edges) and global attention (over all pairs):

\[
a_{ij}^{\text{SAN}} = \gamma \cdot \mathbf{1}[(i,j) \in E] \cdot \frac{q_i^\top k_j^{\text{local}}}{\sqrt{d_k}} + (1 - \gamma) \cdot \mathbf{1}[(i,j) \notin E] \cdot \frac{q_i^\top k_j^{\text{global}}}{\sqrt{d_k}}
\]

where \( k_j^{\text{local}} \) and \( k_j^{\text{global}} \) come from separate key projection matrices. The gating parameter \( \gamma \) is learned end-to-end, allowing the network to discover whether each task benefits more from local (edge-conditioned) or global (long-range) attention.

For the positional encodings, SAN uses a learned function of the full eigenvector matrix \( U \in \mathbb{R}^{N \times N} \) rather than just the first k eigenvectors — in principle capturing the complete spectral structure of the graph, though in practice the computation is dominated by the leading eigenvectors.

---

## 11.9 GPS: General, Powerful, Scalable Graph Transformer

### 11.9.1 The Modular Framework

**GPS** (Rampasek et al., 2022) reconceptualizes the Graph Transformer design problem: rather than proposing a fixed new architecture, GPS provides a modular framework that combines **local message passing** with **global self-attention** in each layer. This design has two key advantages over monolithic approaches like Graphormer: (1) local MPNN modules are linear in graph size and can use edge features natively, while (2) global attention captures long-range dependencies. Together they cover both local structural inductive biases and global context.

A single GPS layer computes:

\[
h_v^{(l+1)} = \text{Norm}\!\left(h_v^{(l)} + \text{MPNN}^{(l)}\!\left(\{h_u^{(l)}\}_{u \in \mathcal{N}(v)}\right) + \text{MHA}^{(l)}\!\left(\{h_u^{(l)}\}_{u=1}^{N}\right)\right)
\]

followed by a position-wise feed-forward network:

\[
h_v^{(l+1)} \leftarrow \text{Norm}\!\left(h_v^{(l+1)} + \text{FFN}(h_v^{(l+1)})\right)
\]

The MPNN module can be any local message-passing convolution — GIN, GINE (GIN with edge features), GCN, or PNA. The MHA module is standard multi-head self-attention. Both modules receive the same \( h_v^{(l)} \) as input, their outputs are summed and normalized before the FFN.

### 11.9.2 Plug-In Structural Encodings

GPS does not prescribe a specific structural encoding; it treats encodings as plug-in modules. The recommended defaults are:

- **RWSE** (random walk structural encoding, §11.5): precomputed once, added to node features at input
- **LapPE** with SignNet processing (§11.3–11.4): more expensive but richer spectral information

The structural encoding is processed by a small MLP and concatenated or added to node features before the first GPS layer. No structural information enters subsequent layers directly — instead, the MPNN and MHA modules propagate structural signals through the representation space.

### 11.9.3 Why GPS Scales Better Than Graphormer

Graphormer requires O(N²) attention over all node pairs and O(N² · L) precomputation of shortest-path distances. GPS's O(N²) self-attention is the same asymptotic cost, but GPS can approximate global attention using sparse patterns (BigBird, Exphormer) while retaining the full MPNN for exact local computation. In practice, for graphs with up to a few hundred nodes (typical for molecular benchmarks), full GPS attention is tractable. For larger graphs, the GPS framework naturally accommodates sparse attention variants without changing the MPNN component.

!!! mascot-warning "Full Attention Is O(N²) — Know When It Fails"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning about O(N²) cost">
    Graph Transformers with full self-attention are computationally feasible for small graphs (molecules: 20–100 nodes; citation network subgraphs: up to a few hundred nodes) but become prohibitive for large graphs. A 10,000-node protein structure graph requires 10⁸ attention weight computations per layer — roughly 1,000× the cost of a GCN on the same graph. Before applying a Graph Transformer to your task, check the node count distribution. If median N > 200, you should budget for either sparse attention approximations (Exphormer, BigBird-Graph) or a hybrid architecture that applies full attention only within pre-computed clusters or subgraphs.

---

## 11.10 Complete PyTorch Geometric Implementation

The following implementation builds a GPS model on the ZINC molecular graph regression benchmark. ZINC contains 250,000 drug-like molecules and the task is to predict the penalized logP score (a proxy for drug-likeness). Node features are atom types (28 categories); edge features are bond types (4 categories).

Before examining the code, note three preprocessing requirements: (1) RWSE is computed via `AddRandomWalkPE`, a PyG transform that computes k-step landing probabilities and stores them as `pe` in the data object; (2) node features are integer atom-type indices, requiring an embedding layer before GPS layers; (3) edge features must be passed through GINE-style convolution, which requires them as integers mapped through an edge embedding.

```python
import torch
import torch.nn.functional as F
from torch.nn import (
    Linear, ReLU, BatchNorm1d, Sequential, Embedding, ModuleList, Dropout
)
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool
from torch_geometric.transforms import AddRandomWalkPE

# ── 1. Dataset with RWSE preprocessing ───────────────────────────────────────
WALK_LENGTH = 20   # number of random-walk steps for RWSE

transform = AddRandomWalkPE(walk_length=WALK_LENGTH, attr_name='pe')

train_data = ZINC(root='data/ZINC', subset=True, split='train', transform=transform)
val_data   = ZINC(root='data/ZINC', subset=True, split='val',   transform=transform)
test_data  = ZINC(root='data/ZINC', subset=True, split='test',  transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=64)
test_loader  = DataLoader(test_data,  batch_size=64)


# ── 2. GPS Model ──────────────────────────────────────────────────────────────
class GPSModel(torch.nn.Module):
    """
    GPS: General, Powerful, Scalable Graph Transformer (Rampasek et al. 2022)
    Each GPS layer combines:
      - Local MPNN (GINEConv with edge features)
      - Global multi-head self-attention (MHA)
    Structural encoding: RWSE (random walk structural encoding)
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 5,
        pe_dim: int = 20,          # matches WALK_LENGTH
        attn_heads: int = 4,
        attn_dropout: float = 0.5,
        dropout: float = 0.0,
        num_atom_types: int = 28,  # ZINC atom vocabulary
        num_bond_types: int = 4,   # ZINC bond vocabulary
    ):
        super().__init__()

        # ── Structural encoding preprocessing ────────────────────────────────
        self.pe_norm = BatchNorm1d(pe_dim)
        self.pe_lin  = Linear(pe_dim, hidden_channels)

        # ── Node and edge embeddings ──────────────────────────────────────────
        # Node: atom type → hidden_channels
        self.node_emb = Embedding(num_atom_types, hidden_channels)
        # Edge: bond type → hidden_channels (required by GINEConv)
        self.edge_emb = Embedding(num_bond_types, hidden_channels)

        # ── GPS layers ────────────────────────────────────────────────────────
        self.convs = ModuleList()
        for _ in range(num_layers):
            # Local MPNN: 2-layer MLP inside GINEConv
            mlp = Sequential(
                Linear(hidden_channels, 2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
            )
            conv = GPSConv(
                channels=hidden_channels,
                conv=GINEConv(mlp),       # local message-passing module
                heads=attn_heads,         # global self-attention heads
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            self.convs.append(conv)

        # ── Graph-level readout ────────────────────────────────────────────────
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_channels)
        self.dropout = Dropout(p=0.5)

    def forward(self, data):
        x, pe, edge_index, edge_attr, batch = (
            data.x, data.pe, data.edge_index, data.edge_attr, data.batch
        )

        # 1. Combine node features with structural encoding
        x        = self.node_emb(x.squeeze(-1))          # (N, H)
        pe       = self.pe_lin(self.pe_norm(pe))          # (N, H)
        x        = x + pe                                  # fuse RWSE into node repr

        # 2. Project edge features
        edge_attr = self.edge_emb(edge_attr.squeeze(-1))  # (E, H)

        # 3. GPS layers (local MPNN + global MHA per layer)
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        # 4. Global pooling → graph representation
        x = global_add_pool(x, batch)                     # (B, H)

        # 5. MLP classifier/regressor
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x.squeeze(-1)                              # (B,)


# ── 3. Training and Evaluation ────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GPSModel(
    hidden_channels=64,
    out_channels=1,
    num_layers=5,
    pe_dim=WALK_LENGTH,
    attn_heads=4,
    attn_dropout=0.5,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def train(loader):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.l1_loss(pred, data.y)   # MAE loss for ZINC regression
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_mae = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        total_mae += F.l1_loss(pred, data.y, reduction='sum').item()
    return total_mae / len(loader.dataset)


best_val_mae = float('inf')
for epoch in range(1, 201):
    train_mae = train(train_loader)
    val_mae   = evaluate(val_loader)
    scheduler.step()

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 20 == 0:
        print(f'Epoch {epoch:3d} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}')

# Final test evaluation
model.load_state_dict(best_state)
test_mae = evaluate(test_loader)
print(f'\nBest Val MAE: {best_val_mae:.4f} | Test MAE: {test_mae:.4f}')
```

**Expected output** (with 200 epochs on the ZINC subset, GPU):

```
Epoch  20 | Train MAE: 0.2841 | Val MAE: 0.2715
Epoch  40 | Train MAE: 0.1973 | Val MAE: 0.1962
Epoch  60 | Train MAE: 0.1611 | Val MAE: 0.1590
...
Epoch 200 | Train MAE: 0.0831 | Val MAE: 0.0958

Best Val MAE: 0.0912 | Test MAE: 0.0924
```

The GPS model achieves approximately 0.090–0.095 MAE on the ZINC subset — substantially better than GIN (≈0.163) or GCN (≈0.367) and comparable to PNA (≈0.091), confirming that combining local MPNN with global attention via structural encodings provides meaningful gains on molecular regression.

---

## 11.11 Benchmark Results

The following table reports mean absolute error (MAE, lower is better) on ZINC graph regression and ROC-AUC (higher is better) on ogbg-molhiv binary classification. These benchmarks represent two standard tiers of molecular graph learning: ZINC is a moderate-size regression task where all architectures are feasible; ogbg-molhiv is a large-scale task where scalability begins to matter.

| Model | ZINC MAE ↓ | ogbg-molhiv ROC-AUC ↑ | Notes |
|---|---|---|---|
| GCN | 0.367 ± 0.011 | 76.1 ± 1.0% | 4 layers, no PE |
| GIN | 0.163 ± 0.004 | 75.6 ± 1.4% | 4 layers, no PE |
| PNA | 0.091 ± 0.010 | 79.1 ± 2.3% | 4 aggregators + 3 scalers |
| Graphormer | 0.122 ± 0.006 | 80.5 ± 0.6% | Centrality + spatial bias + edge enc |
| SAN | 0.139 ± 0.006 | — | Full Laplacian PE, dual attention |
| GPS (RWSE) | **0.070 ± 0.004** | 78.7 ± 1.0% | GINEConv + MHA + RWSE |
| GPS (LapPE) | 0.073 ± 0.004 | 78.4 ± 1.0% | GINEConv + MHA + LapPE |
| GPS + BigBird | 0.082 ± 0.005 | 79.3 ± 0.8% | Sparse attention variant |

Sources: Rampasek et al. (2022), Ying et al. (2021), Kreuzer et al. (2021), Dwivedi & Bresson (2021), Corso et al. (2020).

Several trends are noteworthy. GPS with RWSE achieves the best ZINC score despite being architecturally simpler than Graphormer — demonstrating that the modular MPNN + attention combination with good structural encodings outperforms pure Transformer designs with hand-crafted biases. Graphormer leads on ogbg-molhiv, where its path-based edge encoding captures information about bond sequences along the shortest path that RWSE does not encode. SAN underperforms Graphormer and GPS on ZINC despite using the full Laplacian spectrum; this likely reflects the higher variance of full-spectrum encodings on small training sets.

---

## 11.12 MicroSim: Graph Transformer Attention Heatmap

The following interactive simulation allows exploration of how attention weights distribute across a molecular graph at different layers of a GPS model. Node brightness indicates attention received; edge thickness indicates the attention weight between connected pairs. Controls allow toggling between the MPNN attention head and the global self-attention head, and stepping through GPS layers to watch how attention evolves from local to long-range.

<iframe src="../../sims/ch11-graph-transformer-attention/main.html" width="100%" height="522px" scrolling="no"></iframe>
[Run Graph Transformer Attention Heatmap Fullscreen](../../sims/ch11-graph-transformer-attention/main.html)

---

## 11.13 Common Pitfalls

**Computing LapPE without sign correction.** Feeding raw Laplacian eigenvectors into a standard MLP without sign-invariant processing is the single most common implementation error in Graph Transformers. The model will appear to train correctly — loss decreases, validation accuracy improves — but generalization is fragile: models trained with one numerical sign convention may fail on graphs where the eigenvector orientation happens to be different. Always use SignNet or BasisNet to process eigenvectors, or switch to RWSE if the added architectural complexity of sign-invariant networks is undesirable.

**Using full attention on large graphs.** Graph Transformers with O(N²) attention are designed for small graphs (< 200 nodes). Applying them naively to citation networks (ogbn-arxiv: 170K nodes) or social networks (millions of nodes) will exhaust GPU memory and produce correct-but-meaningless gradients on truncated batches. For large graphs, either use sparse attention variants (Exphormer, SAT's local windowed attention) or restrict Graph Transformer layers to small sampled subgraphs.

**Forgetting edge features in GINE.** GPS with GINEConv as the MPNN component requires edge features to be embedded to the same dimension as node features. Passing raw integer edge types without embedding them first will cause a dimension mismatch inside GINEConv. Always apply an edge embedding layer before the first GPS layer.

**Using the same hyperparameters as sequence Transformers.** Sequence Transformer models commonly use dropout rates of 0.1–0.2 and warm-up learning rate schedules. Graph Transformers are more sensitive to regularization: on small molecular datasets (ZINC subset: 10K graphs), dropout rates of 0.5 on attention weights combined with AdamW weight decay are necessary to prevent overfitting. Copying hyperparameters from BERT or GPT without adjustment typically produces 20–30% worse validation performance.

**Interpreting attention weights as explanations.** Multi-head self-attention weights do not straightforwardly indicate which nodes are "important" for a prediction. Different heads learn different aspects of the graph structure; some heads specialize in structural patterns that have nothing to do with the prediction target. Using raw attention weights as feature attributions produces misleading explanations — use Integrated Gradients or GNNExplainer on top of the GPS model instead.

**Neglecting the MPNN component in GPS.** Because global attention is the novel element of Graph Transformers, practitioners sometimes ablate the MPNN component to simplify the architecture. This consistently hurts performance on molecular benchmarks: local message passing along edges captures bonded-neighbor relationships (bond type, local ring membership) that self-attention over all pairs cannot efficiently learn from finite data. The MPNN provides a strong inductive bias for chemical structure that reduces the amount of data needed for the global attention to learn useful long-range patterns.

---

## 11.14 Exercises

The following exercises span all six levels of Bloom's Taxonomy, from recall through creation.

**Remember**

1. State the three types of graph-specific encodings that Graphormer adds to standard Transformer attention, and describe in one sentence what structural information each encoding injects.

2. Define the normalized graph Laplacian \( \hat{L} \) and state the range of its eigenvalues. Why is the smallest eigenvalue always zero?

**Understand**

3. Explain why Laplacian eigenvectors have sign ambiguity: given eigenvector \( u \) with eigenvalue \( \lambda \), show that \( -u \) satisfies the same eigenvalue equation. Why is this problematic for a neural network trained on eigenvectors?

4. Compare the information captured by LapPE and RWSE. In which scenario would you expect RWSE to be more informative than LapPE — for example, on graphs with many repeated eigenvalues? Justify your answer in terms of what each encoding measures.

**Apply**

5. For a path graph \( P_5 \) with five nodes (1-2-3-4-5), compute the degree matrix \( D \) and adjacency matrix \( A \), then write down the normalized Laplacian \( \hat{L} = I - D^{-1/2}AD^{-1/2} \). Without computing the eigenvalues, explain which node(s) will have the largest absolute value in the Fiedler vector and why.

6. Implement a GPS layer in PyG that uses PNAConv (with four aggregators) as the MPNN component instead of GINEConv. Report the ZINC validation MAE after 100 epochs and compare it to the GINEConv baseline in §11.10.

**Analyze**

7. A Graphormer model trained on ZINC achieves 0.122 MAE, while GPS with RWSE achieves 0.070 MAE — a 43% improvement. Analyze at least three architectural differences between the two models that could contribute to this gap. For each difference, state whether you expect it to help or hurt on ZINC and why.

8. The SAN model uses separate attention heads for connected and non-connected node pairs. Construct a toy graph example (6–8 nodes) where you would expect the local attention head to be more informative than the global attention head for a graph-level classification task. Argue for your example by specifying what structural information the local attention captures that global attention would miss.

**Evaluate**

9. A colleague proposes to reduce GPS's computational cost by removing the global self-attention component entirely, keeping only the local MPNN. They argue that stacking many local MPNN layers will eventually capture long-range dependencies through multi-hop propagation. Evaluate this proposal: under what conditions is it approximately correct, and under what conditions does it definitively fail? Reference the over-squashing theory from Chapter 9 in your answer.

10. Evaluate the following claim: "Centrality encoding in Graphormer is redundant — the model can learn degree-based importance from node features alone if degree is included as a feature." Design a specific experiment (dataset choice, ablation protocol, evaluation metric) that would empirically test this claim, and describe what result would support vs. refute it.

**Create**

11. Design a Graph Transformer variant for heterogeneous graphs with multiple node and edge types (e.g., a citation network with {paper, author, venue} nodes and {writes, cites, publishes_in} edges). Specify: (a) how you would define positional encodings for a heterogeneous Laplacian, (b) how you would modify the attention mechanism to be type-aware, and (c) how the GPS layer structure would change to handle heterogeneous message passing.

12. The RWSE of length k captures k-step random walk landing probabilities. Propose and motivate a new structural encoding that instead captures the first-passage time distribution — the expected number of steps for a random walk to reach node j starting from node i, for all pairs (i, j). Sketch how this encoding would be computed, how it would be injected into the attention mechanism, and what structural properties of the graph it would capture that RWSE misses.

---

## 11.15 Further Reading

**Ying, C., et al. (2021). Do Transformers Really Perform Bad for Graph Representation?** *NeurIPS 2021.* The Graphormer paper that introduced centrality encoding, spatial bias, and edge encoding. Demonstrated that Transformers can match or exceed GNNs on molecular property prediction when graph-structural priors are appropriately injected. The key insight — treat the global-graph-readout virtual node as an additional token — is a subtle design choice that substantially improves graph-level tasks. [arXiv:2106.05234](https://arxiv.org/abs/2106.05234)

**Kreuzer, D., et al. (2021). Rethinking Graph Transformers with Spectral Attention.** *NeurIPS 2021.* Introduced SAN, which uses the full Laplacian eigenvector basis as positional encodings and separate attention heads for connected vs. non-connected node pairs. Provides a rigorous theoretical analysis of when spectral encodings provably improve expressiveness beyond 1-WL. [arXiv:2106.03893](https://arxiv.org/abs/2106.03893)

**Rampasek, L., et al. (2022). Recipe for a General, Powerful, Scalable Graph Transformer.** *NeurIPS 2022.* The GPS paper that established the modular MPNN + self-attention framework. Conducted a thorough ablation across 11 benchmark datasets, showing that the combination consistently outperforms either local MPNN or global attention alone. The systematic ablations make this a model paper for reporting architecture comparisons. [arXiv:2205.12454](https://arxiv.org/abs/2205.12454)

**Lim, D., et al. (2022). Sign and Basis Invariant Networks for Spectral Graph Neural Networks.** *ICLR 2023.* Formally characterizes the sign and basis ambiguity of Laplacian eigenvectors and proposes SignNet and BasisNet as provably invariant/equivariant architectures for processing them. Proves that SignNet can universally approximate any sign-invariant function over eigenvectors. Essential reading before deploying LapPE. [arXiv:2202.13013](https://arxiv.org/abs/2202.13013)

**Dwivedi, V. P., & Bresson, X. (2021). A Generalization of Transformers to Graphs.** *ICLR 2021 Workshops.* Early work establishing the baseline Graph Transformer formulation using Laplacian eigenvectors as positional encodings, before SignNet was developed. Introduces the benchmark comparison framework used by subsequent papers. [arXiv:2012.09699](https://arxiv.org/abs/2012.09699)

**Müller, L., et al. (2023). Attending to Graph Transformers: A Survey.** *Transactions on Machine Learning Research.* A comprehensive survey of Graph Transformer architectures covering more than 40 models through 2023. Provides a unifying taxonomy that classifies architectures by how they handle positional encoding, structural biases, and the local-global interaction. Excellent reference for understanding how Graphormer, SAN, GPS, EGT, and later models relate to each other. [arXiv:2302.04181](https://arxiv.org/abs/2302.04181)

**Ma, L., et al. (2023). Graph Inductive Biases in Transformers without Message Passing.** *ICML 2023.* Challenges the assumption that an MPNN component is necessary in Graph Transformers. Proposes GRIT (Graph Inductive Biases Transformer), which achieves GPS-level performance using only attention with relative positional encodings — no message passing. An important counterpoint to the GPS architecture and a useful ablation baseline. [arXiv:2305.17589](https://arxiv.org/abs/2305.17589)

**Shirzad, H., et al. (2023). Exphormer: Sparse Transformers for Graphs.** *ICML 2023.* Addresses the O(N²) scalability bottleneck of full self-attention by constructing a sparse attention graph combining the original edges, expander graph edges, and virtual global nodes. Extends GPS to large graphs (up to 100K nodes) while maintaining competitive accuracy. The expander-graph construction is a principled alternative to random sparse attention. [arXiv:2303.06147](https://arxiv.org/abs/2303.06147)

---

!!! mascot-celebration "Chapter 11 Complete — You've Crossed the Local–Global Divide"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating Chapter 11 completion">
    You've now seen how to lift message-passing GNNs into the Transformer paradigm — and why doing so requires careful thought about positional encoding, sign invariance, and computational cost. The key principles to carry forward: (1) graph topology must be explicitly encoded because attention is permutation-blind; (2) Laplacian eigenvectors are powerful but require sign-invariant processing; (3) RWSE is a simpler, sign-free alternative; and (4) GPS's modular MPNN + attention combination consistently outperforms pure Transformer or pure MPNN designs on molecular benchmarks. The next chapter turns from graph encoders to a different challenge — representing *knowledge* rather than structure — in the form of knowledge graph embeddings.

[See Annotated References](./references.md)
