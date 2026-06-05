---
title: "Chapter 7: GNN Design Space: GraphSAGE and GAT"
description: "Explores the GNN design space through GraphSAGE and GAT, covering aggregation choices, skip connections, hierarchical pooling, and the full taxonomy of node, edge, and graph-level prediction tasks."
generated_by: claude skill chapter-content-generator
date: 2026-05-28 21:49:09
version: 0.08
---

# Chapter 7: GNN Design Space: GraphSAGE and GAT

**Part 2: Graph Neural Networks**

## Summary

Explores the GNN design space through GraphSAGE and GAT, covering aggregation choices, skip connections, hierarchical pooling, and the full taxonomy of node, edge, and graph-level prediction tasks.

## Concepts Covered

This chapter covers the following 22 concepts from the learning graph:

1. GraphSAGE
2. Graph Attention Network (GAT)
3. Attention Mechanism (Graph)
4. Multi-Head Attention (Graph)
5. Skip Connection (GNN)
6. Residual Connection (GNN)
7. Jumping Knowledge Network
8. Graph-Level Readout
9. Global Mean Pooling
10. Global Sum Pooling
11. DiffPool
12. MinCutPool
13. Node-Level Task
14. Edge-Level Task
15. Graph-Level Task
16. Link Prediction
17. Node Classification
18. Graph Classification
19. Graph Regression
20. Virtual Node Augmentation
21. Virtual Edge Augmentation
22. DeepSNAP

## Prerequisites

This chapter builds on:

- [Chapter 6: GNN Foundations: Message Passing and GCN](../06-gnn-foundations/index.md)

---

!!! mascot-welcome "Designing the Right GNN"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving welcome">
    Chapter 6 gave you the GCN — a single, principled GNN architecture derived from spectral graph theory. This chapter steps back and asks a harder question: given the enormous space of possible design choices for a GNN (aggregation function, skip connections, pooling strategy, task head), how do you navigate it to build the right model for your specific problem? We will work through GraphSAGE and GAT as two architectures that push different design levers, develop a complete taxonomy of what GNNs can predict, and learn how to read pooled graph-level representations. By the end, you will have a principled decision framework rather than a bag of heuristics.

---

## 7.1 A Taxonomy of GNN Prediction Tasks

Before examining specific architectures, it is worth establishing the full scope of what GNNs can be asked to predict. The chapter outline covers three fundamentally different prediction regimes, and the appropriate model design — particularly the output head — depends critically on which regime you are in.

Three levels of prediction exist on graphs, corresponding to three different scopes of the output:

- **Node-level tasks:** the model predicts a label or continuous value for each node individually. **Node classification** assigns each node to one of \( C \) categorical classes; **node regression** outputs a continuous value. Examples include classifying users in a social network by community membership, or predicting the atomic partial charge of each atom in a molecule.

- **Edge-level tasks:** the model predicts a property of each edge or the presence of edges that do not yet exist. **Link prediction** is the canonical edge-level task: given the current graph, predict which pairs of nodes will form edges in the future (or which edges are missing from an incomplete observation). Examples include recommending new connections in a professional network, predicting drug-drug interactions, or inferring protein-protein binding from partial interaction databases.

- **Graph-level tasks:** the model predicts a single label or value for an entire graph. **Graph classification** assigns the whole graph to a category; **graph regression** outputs a scalar property. Examples include predicting whether a molecule is toxic (binary classification), estimating its binding affinity to a target protein (regression), or classifying a brain connectivity graph by neurological condition.

These three regimes require different output architectures. For node-level tasks, the node representations \( \mathbf{h}_v^{(K)} \) from the final GNN layer are fed directly into a per-node classifier or regression head (typically a linear layer). For edge-level tasks, an edge representation is formed from the endpoint node representations (by concatenation, element-wise product, or dot product) and passed to an edge classifier. For graph-level tasks, all node representations must be aggregated into a single graph-level vector — the **graph-level readout** — before the final prediction head.

---

## 7.2 GraphSAGE: Inductive Neighborhood Aggregation

The **Graph SAmple and aggreGatE** (GraphSAGE) model, introduced by Hamilton, Ying, and Leskovec (2017), addresses a limitation of GCN that becomes critical in production settings: GCN is transductive, requiring the full graph to be present during training. When new nodes are added to the graph after training (new users joining a social network, new papers published in a citation database), GCN must be retrained from scratch.

GraphSAGE resolves this by learning an **inductive** aggregation function: rather than learning a fixed embedding for each node, it learns the function \( f(\mathbf{x}_v, \{\mathbf{x}_u : u \in \mathcal{N}(v)\}) \) that maps a node's features and its neighborhood features to an embedding. This function can be applied to any node — seen during training or not — as long as its neighborhood features are available.

The GraphSAGE layer has two distinguishing features relative to GCN:

**1. Concatenation instead of averaging.** While GCN computes a weighted average of the node's own representation and its neighbors', GraphSAGE explicitly concatenates the node's own representation \( \mathbf{h}_v^{(k-1)} \) with the aggregated neighborhood representation \( \mathbf{h}_{\mathcal{N}(v)}^{(k)} \), then applies a linear transformation:

\[
\mathbf{h}_{\mathcal{N}(v)}^{(k)} = \text{AGGREGATE}_k\!\left(\left\{\mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v)\right\}\right)
\]
\[
\mathbf{h}_v^{(k)} = \sigma\!\left(W^{(k)} \cdot \left[\mathbf{h}_v^{(k-1)} \;\big\|\; \mathbf{h}_{\mathcal{N}(v)}^{(k)}\right]\right)
\]

where \( \| \) denotes vector concatenation. The weight matrix \( W^{(k)} \in \mathbb{R}^{d^{(k+1)} \times 2d^{(k)}} \) has twice as many input columns as GCN's \( W^{(k)} \), because it processes the concatenated vector. This design preserves the distinction between the node's own identity and its neighborhood context — a distinction GCN's averaging operation conflates.

**2. Neighborhood sampling.** On large graphs, computing the exact neighborhood aggregation for every node at every layer is computationally prohibitive: a node with degree 1,000 requires 1,000 message computations per layer, and those 1,000 neighbors each have their own large neighborhoods at the next layer, causing exponential blowup. GraphSAGE addresses this by sampling a fixed-size subset of each node's neighbors at each layer — typically 25 neighbors at layer 1 and 10 at layer 2 — making the per-node compute cost independent of degree.

GraphSAGE supports four aggregator variants, each with a different inductive bias:

- **Mean aggregator:** \( \mathbf{h}_{\mathcal{N}(v)}^{(k)} = \text{mean}(\{\mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v)\}) \) — the simplest variant, equivalent to GCN (without concatenation) when combined with a linear transform.
- **Max-pooling aggregator:** \( \mathbf{h}_{\mathcal{N}(v)}^{(k)} = \max(\{\sigma(W_{\text{pool}}\, \mathbf{h}_u^{(k-1)} + \mathbf{b}) : u \in \mathcal{N}(v)\}) \) — applies a learned linear transformation to each neighbor before taking the element-wise max, allowing the network to learn which features to "look for" before pooling.
- **LSTM aggregator:** applies a Long Short-Term Memory network to the sequence of neighbor embeddings (in a random permutation). This is the most expressive but also the most expensive, and its permutation-sensitivity is a theoretical weakness.
- **Attention aggregator:** generalizes toward GAT (described below), computing a weighted sum using learned attention scores.

---

## 7.3 Graph Attention Networks: Learning Which Neighbors Matter

GCN assigns each neighbor a weight inversely proportional to the geometric mean of the two nodes' degrees — a fixed, structure-determined weighting that ignores the content of the representations. GraphSAGE's mean and max aggregators similarly treat all (sampled) neighbors as equally worthy of consideration. Neither architecture allows the model to learn that, in a citation network, a paper on differential geometry should attend more strongly to other geometry papers than to unrelated papers that happen to cite it.

The **Graph Attention Network** (GAT), introduced by Veličković et al. (2018), resolves this by replacing fixed structural weights with **learned, content-dependent attention coefficients** that are computed from the feature representations of both endpoints of each edge.

### 7.3.1 Attention Score Computation

Before the attention coefficient is computed, both the sending node \( j \) and the receiving node \( i \) undergo a shared linear transformation \( W \in \mathbb{R}^{d' \times d} \) (projecting from dimension \( d \) to \( d' \)). An **attention mechanism** \( a : \mathbb{R}^{d'} \times \mathbb{R}^{d'} \to \mathbb{R} \) then maps the concatenated transformed representations to a scalar attention score:

\[
e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}^T \left[W\mathbf{h}_i \;\big\|\; W\mathbf{h}_j\right]\right)
\]

where \( \mathbf{a} \in \mathbb{R}^{2d'} \) is a learnable attention vector and LeakyReLU (with negative slope 0.2) is used rather than ReLU to prevent dead neurons on negative scores. The score \( e_{ij} \) represents, in unnormalized form, how important node \( j \) is to node \( i \) given their current representations.

The unnormalized scores are normalized across the neighborhood of node \( i \) using softmax:

\[
\alpha_{ij} = \frac{\exp(e_{ij})}{\displaystyle\sum_{k \in \mathcal{N}(i) \cup \{i\}} \exp(e_{ik})}
\]

This normalization ensures that attention coefficients sum to 1 over each node's neighborhood (including the self-loop), making them interpretable as a learned probability distribution over which neighbors to attend to. The final node representation is a weighted sum of transformed neighbor features:

\[
\mathbf{h}_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}\, W\, \mathbf{h}_j\right)
\]

A key architectural property is that this computation is **local**: the attention coefficients for node \( i \) are computed solely from the features of \( i \) and its direct neighbors, requiring no global information. This means GAT retains the \( O(|E|) \) per-layer computational complexity of other message-passing GNNs.

!!! mascot-thinking "Why Attention Is Richer Than Fixed Weights"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinking">
    The attention coefficient \( \alpha_{ij} \) depends on both \( h_i \) and \( h_j \) — the current representations of both endpoints. This means attention is dynamic: as representations evolve across layers, the same edge can receive different weights in layer 1 versus layer 2. A neighbor that is superficially similar to the center node (similar initial features) may receive high attention early on, then low attention later once the center node has aggregated its context and can better distinguish relevant from irrelevant neighbors. GCN's fixed normalization factor \( \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} \) can never adapt this way — it is a property of the graph topology, not the learned representations.

### 7.3.2 Multi-Head Attention

To stabilize training and allow the model to jointly attend to information from multiple representation subspaces, GAT uses **multi-head attention**: \( K \) independent attention mechanisms are run in parallel, each with its own parameters \( (W^{(k)}, \mathbf{a}^{(k)}) \), and their outputs are either concatenated (for intermediate layers) or averaged (for the final layer).

**Concatenation** (intermediate layers): the representations from \( K \) heads are concatenated, producing a \( Kd' \)-dimensional output:

\[
\mathbf{h}_i' = \Big\|_{k=1}^{K} \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)}\, W^{(k)}\, \mathbf{h}_j\right)
\]

**Averaging** (final layer): concatenating at the output layer would produce an overly wide representation; instead, the \( K \) head outputs are averaged:

\[
\mathbf{h}_i' = \sigma\!\left(\frac{1}{K}\sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)}\, W^{(k)}\, \mathbf{h}_j\right)
\]

The original GAT paper uses \( K = 8 \) heads in the first layer and \( K = 1 \) (or averaging \( K = 8 \)) in the output layer.

!!! mascot-warning "Multi-Head Attention Memory Cost"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning">
    Multi-head attention with \( K \) heads and hidden dimension \( d' \) requires storing \( K \) separate weight matrices \( W^{(1)}, \ldots, W^{(K)} \) and attention vectors \( \mathbf{a}^{(1)}, \ldots, \mathbf{a}^{(K)} \), plus the attention score \( \alpha_{ij}^{(k)} \) for every edge and every head. On a graph with 1 million edges and \( K = 8 \) heads, storing float32 attention scores requires \( 1{,}000{,}000 \times 8 \times 4 = 32 \) MB per layer — manageable, but non-trivial on GPU. More critically, the softmax normalization over each node's neighborhood is a sequential reduction that is hard to parallelize on GPU hardware when neighborhoods have highly variable sizes. For large graphs, **GATv2** (Brody et al., 2022) addresses this with a modified attention mechanism that is both more expressive and more GPU-friendly.

#### Diagram: GAT Attention Weight Visualizer


<iframe src="../../sims/ch07-gat-attention-weights/main.html" width="100%" height="502px" scrolling="no"></iframe>
[Run GAT Attention Weight Visualizer Fullscreen](../../sims/ch07-gat-attention-weights/main.html)

---

## 7.4 Skip Connections and Jumping Knowledge

The over-smoothing problem established in Chapter 6 limits most GNN architectures to 2–3 layers. Two complementary mechanisms allow deeper networks while preserving discriminative node representations: **residual (skip) connections** and the **Jumping Knowledge** (JK) architecture.

### 7.4.1 Residual Connections

A **residual connection** (or **skip connection**) adds the input representation of a layer directly to its output:

\[
\mathbf{h}_v^{(k)} = \sigma\!\left(\text{GNN}^{(k)}\!\left(\mathbf{h}_v^{(k-1)},\, \{\mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v)\}\right)\right) + \mathbf{h}_v^{(k-1)}
\]

The additive identity path means that even a very deep GNN can always fall back to its earlier representations — gradient flow is improved (no vanishing gradients), and the network can learn to use deeper aggregation only when it is beneficial. The requirement that input and output have the same dimension (for the addition to be defined) can be relaxed by using a learned linear projection: \( W_{\text{skip}}\, \mathbf{h}_v^{(k-1)} \) instead of \( \mathbf{h}_v^{(k-1)} \).

The You et al. (2020) design space study found that skip connections are nearly universally beneficial across GNN architectures, tasks, and datasets — one of the few unambiguous findings in a space full of dataset-specific best practices.

### 7.4.2 Jumping Knowledge Networks

The **Jumping Knowledge Network** (JK-Net, Xu et al., 2018) takes a different approach: rather than adding intermediate representations to the final output, it concatenates representations from **all** layers:

\[
\mathbf{h}_v^{\text{final}} = f\!\left(\mathbf{h}_v^{(1)} \;\big\|\; \mathbf{h}_v^{(2)} \;\big\|\; \cdots \;\big\|\; \mathbf{h}_v^{(K)}\right)
\]

where \( f \) can be a concatenation (forming a \( Kd \)-dimensional vector), a max operation (taking the element-wise maximum across layers), or a learned LSTM operating over the layer sequence. The JK-Net insight is that different nodes may benefit from different effective depths: nodes near labeled boundary regions may need only 1-hop context, while nodes deep inside a homogeneous cluster may need 3-hop context to access label information at all. By preserving all layer representations, JK-Net lets a downstream aggregation function adapt the effective depth **per node** rather than applying a uniform depth to all nodes.

The trade-off relative to residual connections is parameter cost: JK-Net with concatenation expands the final representation dimension by a factor of \( K \), while residual connections add no extra parameters.

---

## 7.5 Graph-Level Readout and Pooling

For **graph-level tasks** (graph classification and regression), a graph-level representation must be computed by aggregating the final node embeddings \( \{\mathbf{h}_v^{(K)} : v \in V\}  \) into a single fixed-size vector \( \mathbf{h}_G \). Two categories of readout exist: flat global pooling and hierarchical pooling.

### 7.5.1 Flat Global Pooling

The simplest graph-level readout operations apply a permutation-invariant function directly to the set of all node embeddings:

**Global mean pooling:**

\[
\mathbf{h}_G = \frac{1}{|V|} \sum_{v \in V} \mathbf{h}_v^{(K)}
\]

**Global sum pooling:**

\[
\mathbf{h}_G = \sum_{v \in V} \mathbf{h}_v^{(K)}
\]

**Global max pooling:** element-wise maximum across all node embeddings.

Mean pooling is scale-invariant (graph size does not affect magnitude) and robust to outlier nodes, but it destroys information about graph size and the total quantity of a feature across the graph. Sum pooling preserves size sensitivity and is more expressive (two graphs with different node counts cannot produce the same sum if their node embeddings are consistent), but gradients can become large for big graphs. In practice, both variants are used, and they are often combined: concatenating mean and sum pooling provides both size-invariant and size-sensitive signals.

### 7.5.2 Hierarchical Pooling

Flat pooling discards all intermediate structural information — it treats the graph as a bag of node embeddings without regard for which nodes are connected. For graphs where hierarchical community structure carries predictive signal (e.g., functional groups in molecules, subregions in brain connectivity graphs), **hierarchical pooling** methods iteratively cluster nodes and coarsen the graph.

#### DiffPool

**DiffPool** (Ying et al., 2018) learns a soft cluster assignment matrix \( S^{(l)} \in \mathbb{R}^{n_l \times n_{l+1}} \) at each pooling level using a dedicated GNN:

\[
S^{(l)} = \text{softmax}\!\left(\text{GNN}_{\text{pool}}^{(l)}\!\left(A^{(l)},\, X^{(l)}\right)\right)
\]

where \( n_l \) is the number of nodes at level \( l \) and \( n_{l+1} < n_l \) is the target cluster count. A second GNN simultaneously computes node embeddings at level \( l \):

\[
Z^{(l)} = \text{GNN}_{\text{embed}}^{(l)}\!\left(A^{(l)},\, X^{(l)}\right)
\]

The coarsened graph at the next level is obtained by aggregating node features and adjacency according to the soft assignment:

\[
X^{(l+1)} = S^{(l)T} Z^{(l)}, \qquad A^{(l+1)} = S^{(l)T} A^{(l)} S^{(l)}
\]

The assignment \( S^{(l)T} Z^{(l)} \) computes a weighted average of node embeddings within each cluster, while \( S^{(l)T} A^{(l)} S^{(l)} \) computes a dense cluster-level adjacency. DiffPool is trained end-to-end with two auxiliary losses: an entropy loss encouraging discrete cluster assignments (rather than soft uniform distributions) and a link prediction loss encouraging cluster-level adjacency to reflect the original adjacency.

The major limitation of DiffPool is computational: the cluster-level adjacency \( A^{(l+1)} \) is generally dense, losing the sparsity that makes GNN message passing efficient. For graphs with more than a few hundred nodes, DiffPool becomes memory-intensive.

#### MinCutPool

**MinCutPool** (Bianchi et al., 2020) provides a spectral relaxation interpretation of the pooling problem. Rather than the entropy regularization of DiffPool, MinCutPool minimizes a normalized cut objective over the cluster assignments directly in the loss function:

\[
\mathcal{L}_{\text{MinCut}} = -\frac{\text{tr}\!\left(S^T A S\right)}{\text{tr}\!\left(S^T D S\right)} + \left\| \frac{S^T S}{\| S^T S \|_F} - \frac{I_{n'}}{\sqrt{n'}} \right\|_F
\]

The first term encourages clusters with dense internal connections (small normalized cut), while the second term (an orthogonality regularizer) penalizes the cluster assignments from collapsing to a trivial solution where all nodes are assigned to the same cluster. MinCutPool has a tighter connection to spectral clustering than DiffPool and often achieves better cluster quality on structured graphs.

!!! mascot-encourage "Hierarchical Pooling Is Worth the Complexity"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage encouraging">
    DiffPool and MinCutPool introduce substantial complexity — two GNNs running simultaneously, auxiliary losses, and dense intermediate adjacency matrices. If that feels overwhelming, hold onto the core idea: they are learning to do hierarchically what label propagation does level by level. At each pooling step, nodes that are tightly connected and have similar features are merged into a super-node. The resulting coarser graph preserves the global structure while discarding local detail. For molecular graphs, this naturally discovers functional groups; for social graphs, it discovers communities. The complexity is the price of doing graph classification properly when structure at multiple scales matters.

---

## 7.6 Graph Augmentation: Virtual Nodes and Edges

Two simple but highly effective graph preprocessing strategies improve GNN performance on tasks that involve long-range dependencies.

**Virtual node augmentation** adds a single new node \( v_G \) to the graph and connects it to every other node with bidirectional edges. During message passing, the virtual node aggregates information from the entire graph (via its direct connections to all nodes) and distributes global context back to every node at each layer. This effectively gives every node in the graph a 2-hop path to every other node — radically shrinking the effective graph diameter while adding only \( 2|V| \) edges. The virtual node is initialized with a learned feature vector (or zero vector) and is discarded at readout, where it is either excluded or used as the graph-level representation directly. Hamilton et al. (2017) showed that virtual nodes consistently improve performance on molecular property prediction benchmarks.

**Virtual edge augmentation** adds edges between pairs of nodes that are currently disconnected but share structural features — for example, adding edges between all pairs of nodes within 2 hops (completing the 2-hop neighborhood into a dense subgraph) or between nodes with high Personalized PageRank scores. This technique is particularly useful in chemistry, where reactive sites in a molecule may be geometrically proximal but graph-distance far due to the rigid ring-based topology of many drug candidates. The **SEAL** framework (Zhang and Chen, 2018) for link prediction systematically adds edges based on local subgraph topology.

The following table summarizes the two augmentation strategies and their primary use cases, which we have just described.

| Augmentation | What it adds | Primary benefit | Overhead |
|---|---|---|---|
| Virtual node | One super-node connected to all \( V \) nodes | Global context at each layer; shorter effective diameter | \( O(|V|) \) edges added |
| Virtual edge | Edges between structurally proximate node pairs | Long-range interactions without extra GNN layers | \( O(|V|^2) \) at worst; typically \( O(|V| \cdot k) \) for \( k \)-NN |

---

## 7.7 The GNN Design Space: A Systematic View

The You et al. (2020) paper "Design Space of Graph Neural Networks" ran a large-scale empirical study evaluating GNN designs across four axes — message passing, aggregation, skip connections, and training regularization — on 12 datasets spanning node classification, link prediction, and graph classification. Their key findings are worth internalizing before choosing a GNN architecture.

Before reading the summary table below, note the definitions: **intra-layer design** covers how a single GNN layer aggregates and transforms; **inter-layer design** covers how layers are connected (skip connections, JK-Net); **learning configuration** covers batch size, learning rate schedule, dropout placement.

| Design axis | Key finding | Practical implication |
|---|---|---|
| Aggregation function | Sum aggregation outperforms mean on most graph classification tasks; mean is more robust for node classification | Match aggregation to task: sum for graph-level, mean for node-level |
| Skip connections | Residual connections improve accuracy on 10 of 12 datasets | Default to residual connections for any GNN with depth ≥ 2 |
| Batch normalization | Post-aggregation batch norm is beneficial; pre-aggregation hurts | Apply batch norm after the update function, not inside aggregation |
| Layer depth | 2–4 layers optimal across all tasks; deeper is rarely better | Start with 2 layers, ablate to 4; never start with 8+ |
| Pre-processing MLP | Applying a 2-layer MLP to input features before the first GNN layer consistently helps | Always pre-process node features before GNN layers |

!!! mascot-tip "How to Pick Your Architecture"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage giving a tip">
    Here is the decision flowchart for practitioners. (1) Is inductiveness required? If you need to embed new nodes at inference time without retraining, use GraphSAGE or GAT; rule out GCN. (2) Do neighbor weights matter semantically? If adjacent nodes have very different relevance levels (heterogeneous neighborhoods), use GAT; otherwise, start with GCN or GraphSAGE-mean. (3) Is your graph large (>100K nodes)? GraphSAGE's neighborhood sampling is essential; full-graph GCN/GAT will OOM. (4) Is the task graph-level? Add global mean + sum pooling; try a virtual node; run a DiffPool ablation only if your graph has clear hierarchical structure and you have the compute budget. (5) When in doubt: 2-layer GCN with residual connections, batch normalization, and pre-processing MLP is a competitive baseline on almost every benchmark.

---

## 7.8 PyG and DeepSNAP: Frameworks for Multi-Task GNNs

Chapter 6 introduced PyTorch Geometric (PyG) as the primary GNN framework. For this chapter's broader set of prediction tasks — particularly heterogeneous graphs and multi-task pipelines — **DeepSNAP** extends PyG with a higher-level graph abstraction.

**DeepSNAP** (Hu et al., 2021) wraps PyG's `Data` and `HeteroData` objects in a `Graph` class that explicitly tracks multiple node types, edge types, label locations (node/edge/graph), and task configurations. Its primary design goal is enabling heterogeneous multi-task training without manually managing multiple tensor indices. Key features:

- **Typed node and edge features:** `graph.node_feature`, `graph.edge_feature`, and `graph.graph_feature` are keyed dictionaries for heterogeneous graphs.
- **Task-aware splitting:** `graph.split()` correctly partitions node, edge, or graph labels into train/val/test while respecting graph structure (e.g., not including test edges in the training message-passing graph).
- **Negative sampling:** built-in support for link prediction negative sampling with configurable strategies (random, structure-aware).

For the homogeneous, single-task scenarios covered in most of this textbook, raw PyG is sufficient. DeepSNAP becomes valuable in Chapters 12–17 (knowledge graphs, heterogeneous graphs, recommender systems) where multi-relational and multi-task training are the norm.

---

## 7.9 Code: GraphSAGE and GAT on Cora with PyG

The following implementation trains both GraphSAGE and GAT on the Cora citation network for node classification. The key parameters are: `SAGEConv` uses `aggr='mean'` by default (can be changed to `'max'` or `'lstm'`); `GATConv`'s `heads` parameter controls multi-head attention width, with `concat=True` for all but the final layer.

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv, GATConv

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# ── GraphSAGE ────────────────────────────────────────────────────────────────
class GraphSAGE(torch.nn.Module):
    """
    Two-layer GraphSAGE with mean aggregation.
    SAGEConv(in, out, aggr='mean') computes:
        h_N(v) = mean({h_u : u in N(v)})
        h_v    = W · [h_v || h_N(v)]   (concat then linear)
    """
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hid_ch, aggr='mean')
        self.conv2 = SAGEConv(hid_ch, out_ch, aggr='mean')

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, ei)
        return F.log_softmax(x, dim=1)

# ── GAT ──────────────────────────────────────────────────────────────────────
class GAT(torch.nn.Module):
    """
    Two-layer GAT.
    Layer 1: 8 attention heads, hidden_dim=8 per head → 64-dim output (concatenated).
    Layer 2: 1 head (averaging), out_dim = num_classes.
    GATConv(in, out_per_head, heads=K, concat=True) computes:
        alpha_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
        h_i'     = sigma(sum_j alpha_ij * W * h_j)   (one head)
    Concatenation across K heads → K * out_per_head final dim.
    """
    def __init__(self, in_ch, hid_ch, out_ch, heads=8):
        super().__init__()
        # Layer 1: heads=8, each head output dim = hid_ch, concat → 8*hid_ch
        self.conv1 = GATConv(in_ch, hid_ch, heads=heads, dropout=0.6, concat=True)
        # Layer 2: 1 head → out_ch (no concat needed)
        self.conv2 = GATConv(hid_ch * heads, out_ch, heads=1, dropout=0.6, concat=False)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, ei))        # ELU instead of ReLU (from GAT paper)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, ei)
        return F.log_softmax(x, dim=1)

# ── Training loop ─────────────────────────────────────────────────────────────
def run_model(model, lr=0.005, wd=5e-4, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = out.argmax(dim=1)
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        best_val = max(best_val, val_acc)

    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    return best_val, test_acc

sage = GraphSAGE(dataset.num_node_features, 64, dataset.num_classes)
gat  = GAT(dataset.num_node_features, 8, dataset.num_classes, heads=8)

sage_val, sage_test = run_model(sage)
gat_val,  gat_test  = run_model(gat, lr=0.005)

print(f"GraphSAGE  | Val: {sage_val:.3f} | Test: {sage_test:.3f}")
print(f"GAT        | Val: {gat_val:.3f}  | Test: {gat_test:.3f}")
```

#### Diagram: GNN Design Space Interactive Comparison


<iframe src="../../sims/ch07-gnn-design-comparison/main.html" width="100%" height="522px" scrolling="no"></iframe>
[Run GNN Design Space Interactive Comparison Fullscreen](../../sims/ch07-gnn-design-comparison/main.html)

---

## 7.10 Benchmark Comparison

The following table reports test accuracy on three node classification benchmarks and graph classification accuracy on the TUDataset MUTAG molecular graph benchmark.

| Method | Cora | CiteSeer | ogbn-arxiv | MUTAG (graph cls.) |
|---|---|---|---|---|
| GCN (Kipf & Welling 2017) | 81.5% | 70.3% | 71.7% | 85.7% |
| GraphSAGE-mean (Hamilton et al. 2017) | 82.0% | 71.3% | 71.5% | 87.6% |
| GAT (Veličković et al. 2018) | 83.0% | 72.5% | 73.9% | 89.4% |
| GATv2 (Brody et al. 2022) | 83.7% | 73.1% | 74.1% | — |
| GCN + Residual + BN | 82.8% | 71.9% | 72.6% | 87.2% |
| GraphSAGE + Virtual Node | 83.1% | 72.0% | 73.2% | 89.0% |

Several patterns are worth noting. GAT consistently outperforms both GCN and GraphSAGE on homophilic citation networks, reflecting the value of content-dependent neighbor weighting when features are informative. The residual + batch norm augmentation (fifth row) substantially closes the gap between GCN and GAT on Cora, suggesting that much of GAT's advantage comes from implicit regularization rather than the attention mechanism itself. Virtual node augmentation benefits GraphSAGE on graph classification (MUTAG) significantly, as molecular graphs often contain reactive groups far apart in graph distance.

---

## 7.11 Common Pitfalls

**1. Using GAT without the self-loop in the softmax denominator.** The attention normalization sums over \( \mathcal{N}(i) \cup \{i\} \) — including the self-loop — not just \( \mathcal{N}(i) \). Omitting the self-loop inflates attention weights on neighbors (they sum to 1 instead of sharing probability mass with the self-connection), which tends to push attention-weighted aggregation toward pure neighborhood averaging and loses the self-representation benefit of the self-loop in GCN. PyG's `GATConv` adds self-loops by default; if building GAT manually, do not forget to include \( (i, i) \) in the attention denominator.

**2. Forgetting to divide hidden dimension by the number of heads.** With 8 attention heads and hidden dimension 64, each head should operate on dimension 8, not 64 — otherwise the first layer outputs \( 8 \times 64 = 512 \) dimensions, not 64. A common mistake is setting `GATConv(in, 64, heads=8)`, which gives a 512-dimensional intermediate representation and a parameter count 8× larger than intended. The correct call is `GATConv(in, 8, heads=8)` for a 64-dimensional output.

**3. Applying DiffPool to sparse graphs.** DiffPool's coarsened adjacency \( S^T A S \) is dense by construction: even if the original graph is sparse, any two clusters share some weighted connection after pooling. For graphs with more than 1,000 nodes, this dense intermediate adjacency grows quadratically in memory and makes DiffPool impractical without truncation. Use MinCutPool with sparse approximations, or flat global pooling with a virtual node, for large graphs.

**4. Using graph-level mean pooling without also trying sum pooling.** Mean pooling is scale-invariant — a graph with 10 nodes and a graph with 100 nodes will produce the same mean representation if their per-node embeddings are identical. For tasks where graph size is predictive (e.g., predicting molecular weight, which scales with atom count), mean pooling discards critical information. Always try concatenating mean and sum pooling before settling on one.

**5. Treating link prediction as a node-level task.** A common implementation mistake is using node classification infrastructure (softmax over classes at each node) to perform link prediction. Link prediction requires forming an edge-level score from the representations of both endpoint nodes — typically via dot product or bilinear scoring — followed by binary cross-entropy with negative samples. The negative sampling strategy (random vs. hard negatives) significantly affects both training dynamics and evaluation metrics; choose carefully based on the application domain.

---

!!! mascot-celebration "The Design Space Is Now Yours"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating">
    Twenty-two concepts in one chapter — that is a lot of aggregating. You now have a complete picture of the modern GNN design space: from the inductive sampling of GraphSAGE and the learned attention of GAT, through skip connections and jumping knowledge that tame over-smoothing, to the full taxonomy of node-, edge-, and graph-level prediction tasks and the hierarchical pooling strategies that serve them. Most importantly, you have a decision framework rather than a list of facts. The next two chapters (GNN Training and GNN Theory) examine the mechanics and limits of everything you just learned. Your neighbors have insights too — and now you know exactly which ones to listen to.

---

## 7.12 Exercises

### Remember

1. State the GraphSAGE layer update equation, identifying the three components: the neighborhood aggregator \( \text{AGGREGATE}_k \), the concatenation, and the linear transformation with nonlinearity. How does this update differ structurally from the GCN update?

2. Define the three levels of GNN prediction tasks (node, edge, graph). Give one real-world application example for each level from a different domain (social networks, chemistry, and biology).

### Understand

3. Explain why multi-head attention in GAT improves stability over single-head attention. What is the information-theoretic argument for maintaining multiple independent attention patterns simultaneously, and how does the averaging operation in the final layer relate to ensemble learning?

4. A residual connection adds the input representation to the output of a GNN layer: \( \mathbf{h}_v^{(k)} = \text{GNN}^{(k)}(\mathbf{h}_v^{(k-1)}) + \mathbf{h}_v^{(k-1)} \). Explain how this addition helps with gradient flow in deep networks and why it mitigates (but does not eliminate) the over-smoothing problem.

### Apply

5. You are building a link prediction system for a protein-protein interaction (PPI) network. The network has 20,000 proteins (nodes) with 128-dimensional feature vectors and 150,000 known interactions (edges). Propose a complete GNN pipeline: specify the architecture (GCN, GraphSAGE, or GAT), number of layers, hidden dimension, pooling strategy if any, edge scoring function, and loss function. Justify each choice.

6. Implement a two-layer GAT with \( K = 4 \) attention heads on the Karate Club graph using PyG's `GATConv`. For each node, print the attention weight vector over its neighborhood after training for 100 epochs on the faction labels (2 classes). Report the node (besides the two seeds) that receives the most concentrated attention (highest max \( \alpha_{ij} \) over its neighbors).

### Analyze

7. GraphSAGE's LSTM aggregator applies an LSTM to the sequence of neighbor embeddings in a **random permutation**. Analyze why this is a design flaw from a theoretical standpoint: what property does a valid aggregation function for sets of neighbors need to satisfy, and how does the LSTM aggregator violate it? What are the practical consequences on training stability?

8. DiffPool learns a soft assignment \( S^{(l)} \) using a GNN. At convergence, rows of \( S^{(l)} \) ideally approach one-hot vectors (hard cluster assignments). Analyze the trade-off: if the entropy regularization is too strong, what happens to gradient flow during training? If it is too weak, what happens to the quality of cluster assignments? How does MinCutPool's orthogonality regularizer avoid this particular tension?

### Evaluate

9. Compare the computational complexity of GAT and GCN, both in terms of space (memory per forward pass) and time (FLOPs per layer). Assume a graph with \( |V| \) nodes, \( |E| \) edges, input dimension \( d \), hidden dimension \( d' \), and \( K \) attention heads. Under what graph regime (dense vs. sparse, large vs. small) does the attention overhead of GAT become prohibitive relative to GCN?

10. Evaluate the claim: "Virtual node augmentation is strictly better than adding a 4th GNN layer because it achieves global information exchange without over-smoothing." Identify any conditions under which this claim is false, and describe a graph structure where adding the virtual node would actually hurt performance rather than help it.

### Create

11. Design a **hierarchical attention pooling** architecture that combines the ideas of DiffPool (learned cluster assignments) and GAT (content-dependent attention). Specifically, define a cluster assignment mechanism that uses attention scores rather than a separate GNN for assignment, write the cluster-level adjacency and feature update equations, and describe the auxiliary losses needed to train the system.

12. Propose a **task-adaptive readout** layer that dynamically selects between global mean pooling, global sum pooling, and virtual-node-based readout based on the graph's structural properties (e.g., size, density, diameter). Formally define the selection mechanism (what input does it use, what is the output distribution over readout strategies), and describe how you would train this mechanism end-to-end with the rest of the GNN.

---

## 7.13 Further Reading

1. **Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs.** *Advances in Neural Information Processing Systems (NeurIPS 30).* The GraphSAGE paper. Pay particular attention to the theoretical analysis (Section 4) showing that GraphSAGE embeddings satisfy a VC-dimension bound that enables generalization to unseen nodes — the key result that makes inductiveness more than a convenience feature.

2. **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks.** *International Conference on Learning Representations (ICLR 2018).* The GAT paper. Section 2.2's analysis of attention as a generalization of GCN's fixed structural weights is essential reading. Compare the computational complexity table (Table 1) with GCN and SplineCNN.

3. **Brody, S., Alon, U., & Yahav, E. (2022). How attentive are graph attention networks?** *International Conference on Learning Representations (ICLR 2022).* Demonstrates that GAT's attention mechanism is, counterintuitively, static — it cannot distinguish different graphs with the same local structure. GATv2 fixes this with a modified attention that is strictly more expressive. Essential reading before using GAT in production.

4. **Xu, K., Li, C., Tian, Y., Sonobe, T., Kawarabayashi, K., & Jegelka, S. (2018). Representation learning on graphs with jumping knowledge networks.** *Proceedings of ICML 2018.* Introduces JK-Net with a clean analysis of why different nodes benefit from different effective depths. The empirical study across 4 aggregation strategies (max, concatenation, LSTM, mean) is directly applicable when choosing a JK-Net variant.

5. **You, J., Ying, R., & Leskovec, J. (2020). Design space for graph neural networks.** *Advances in Neural Information Processing Systems (NeurIPS 33).* The systematic empirical study covering 315,000 GNN configurations across 12 tasks. Table 3's ranking of design choices by importance should be committed to memory by anyone building GNNs in practice.

6. **Ying, R., You, J., Morris, C., Ren, X., Hamilton, W. L., & Leskovec, J. (2018). Hierarchical graph representation learning with differentiable pooling.** *Advances in Neural Information Processing Systems (NeurIPS 31).* The DiffPool paper. The visualization of learned cluster assignments on the ENZYMES dataset (Figure 3) provides strong intuition for what hierarchical pooling is actually doing.

7. **Bianchi, F. M., Grattarola, D., & Alippi, C. (2020). Spectral clustering with graph neural networks for graph pooling.** *Proceedings of ICML 2020.* The MinCutPool paper. Section 3's derivation of the MinCut loss from the normalized cut objective makes the connection to spectral clustering precise and readable.

8. **Zhang, M., & Chen, Y. (2018). Link prediction based on graph neural networks.** *Advances in Neural Information Processing Systems (NeurIPS 31).* Introduces SEAL — an enclosing-subgraph-based link prediction framework that systematically uses virtual edges (local subgraph extraction) to provide structural context for link prediction. Benchmark results show consistent improvements over pure embedding-based methods on citation and biological networks.

[See Annotated References](./references.md)
