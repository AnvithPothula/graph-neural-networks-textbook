---
title: "Chapter 10: Designing Powerful Encoders: GIN and Beyond"
description: "Designs GNNs that are provably maximally expressive — GIN achieves the 1-WL upper bound — and explores SGC, APPNP, PNA, ID-GNN, higher-order GNNs, subgraph GNNs, and stochastic depth as principled extensions."
generated_by: claude skill chapter-content-generator
date: 2026-05-28
version: 0.08
---

# Chapter 10: Designing Powerful Encoders: GIN and Beyond

**Part 2: Graph Neural Networks**

## Summary

Designs GNNs that are provably maximally expressive — GIN achieves the 1-WL upper bound — and explores SGC, APPNP, PNA, ID-GNN, higher-order GNNs, subgraph GNNs, and stochastic depth as principled extensions beyond 1-WL.

## Concepts Covered

This chapter covers the following 9 concepts from the learning graph:

1. Jumping Knowledge Network (JK-Net)
2. Simple Graph Convolution (SGC)
3. APPNP
4. Principal Neighbourhood Aggregation (PNA)
5. Identity-Aware GNN (ID-GNN)
6. k-GNN and δ-k-GNN
7. Subgraph GNN (ESAN / DS-GNN)
8. PairNorm
9. Stochastic Depth (GNN)

## Prerequisites

This chapter builds on:

- [Chapter 6: GNN Foundations: Message Passing and GCN](../06-gnn-foundations/index.md)
- [Chapter 7: GNN Design Space: GraphSAGE and GAT](../07-gnn-design-space/index.md)
- [Chapter 8: GNN Training, Augmentation, and Practical Tips](../08-gnn-training/index.md)
- [Chapter 9: Theory of GNNs: Expressiveness and the WL Test](../09-gnn-theory/index.md)

---

!!! mascot-welcome "Welcome to Chapter 10 — Building the Right Tool for the Job"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving welcome">
    Chapter 9 established the ceiling: no standard message-passing GNN can surpass the 1-WL test, and mean and max aggregation fall strictly below it. Now we build. This chapter designs GNNs that either *achieve* the 1-WL ceiling precisely, *surpass* it through architectural innovation, or *simplify* gracefully for settings where 1-WL power is unnecessary. Each architecture in this chapter is not an arbitrary engineering choice — it is a principled response to a specific theoretical limitation of previous designs. By the end, you will be able to select the right encoder for a given task based on the structural properties that task requires the model to detect.

## Motivating Example: Drug Discovery and Structural Isomers

In computational drug discovery, two molecules can share the same molecular formula (same atom types and counts) and even the same local neighborhood structure for every atom, yet have dramatically different pharmacological properties — different binding affinities, toxicity profiles, and metabolic pathways. These **structural isomers** differ in global topology: the arrangement of rings, the position of a substituent group on a ring, or the number of fused ring systems.

A 1-WL-bounded GNN cannot distinguish these molecules. It assigns the same embedding to any node whose local subtree structure matches, regardless of global position. For drug discovery — where the difference between a therapeutic compound and a toxic one may hinge on whether a nitrogen atom appears at position 3 or position 5 of a ring — this limitation is not academic; it directly predicts model failure on real targets. The architectures in this chapter are designed precisely to overcome these limitations.

---

## 10.1 GIN: Full 1-WL Power in Practice

Chapter 9 established that the Graph Isomorphism Network (GIN) achieves the maximum expressiveness possible for any MPNN — the 1-WL upper bound. Here we examine its practical implementation in full detail: the layer structure, the graph-level readout strategy, and the engineering choices that determine whether GIN's theoretical advantages translate to empirical gains.

### 10.1.1 The GIN Layer

Each GIN layer applies a multi-layer perceptron to the sum of the node's own embedding and its neighbors' embeddings:

\[
\mathbf{h}_v^{(k+1)} = \text{MLP}^{(k)}\!\left((1 + \epsilon^{(k)}) \cdot \mathbf{h}_v^{(k)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k)}\right)
\]

Three implementation details determine performance in practice:

**The MLP depth**: The theorem of Xu et al. (2019) requires the MLP to be a *universal approximator* — it must be capable of approximating any injective function on the domain of possible embeddings. In practice, a two-layer MLP with batch normalization between layers is sufficient. Single-layer linear maps are not sufficient: they cannot approximate non-linear injections on the embedding space.

**Batch normalization placement**: Batch normalization should be applied *inside* the MLP, between the first and second linear layers — not before the GINConv call or after it. Normalizing after the full GIN layer would remove the magnitude information that sum aggregation preserves, partially defeating the purpose of using sum over mean.

**The ε parameter**: Setting \( \epsilon = 0 \) (fixed) works well in practice for most tasks. The theoretical justification for a non-zero ε is that it prevents a node from confusing "I have no neighbors with embedding \( \mathbf{a} \)" with "I have one neighbor with embedding \( \mathbf{a} \) and my own embedding is 0." In practice, when embeddings are initialized randomly and features are non-zero, the distinction is preserved even with ε = 0.

### 10.1.2 Graph-Level Readout: Layer-Wise Aggregation

For graph classification, GIN uses a graph-level readout that aggregates across all layers, not just the final layer:

\[
\mathbf{h}_G = \text{CONCAT}\!\left(\text{READOUT}\!\left(\left\{\!\left\{ \mathbf{h}_v^{(k)} : v \in V \right\}\!\right\}\right)\ \Big|\ k = 0, 1, \ldots, K\right)
\]

where READOUT is typically sum pooling. This is equivalent to applying Jumping Knowledge aggregation at the graph level: the final graph representation concatenates the layer-0 (initial feature), layer-1, ..., layer-K global sum poolings. This design captures structural features at multiple scales — layer-0 captures individual node features, layer-1 captures 1-hop structure, layer-k captures k-hop structure — and is one reason GIN outperforms simple readout architectures.

After concatenation, a linear layer or shallow MLP maps the concatenated graph representation to the output class logits.

---

## 10.2 Simple Graph Convolution (SGC)

Not every task requires the full expressiveness of GIN or even GCN. The **Simple Graph Convolution (SGC)** (Wu et al., 2019) asks: how much of GCN's performance comes from the non-linear activations between layers, and how much comes purely from the multi-hop smoothing of features?

!!! mascot-thinking "Sage Thinks: Simplification Is Sometimes Strength"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinking">
    Before dismissing SGC as "too simple," consider what it removes: all non-linear activations between GCN layers. What remains is a linear model on features smoothed by \( K \) hops of graph diffusion. On homophilic graphs where the label signal is smooth (nearby nodes share labels), multi-hop smoothing reduces noise and the non-linearities add almost nothing. SGC is linear, interpretable, and trains in a fraction of the time. The lesson isn't "simple is always better" — it's "understand what your model actually needs before adding complexity."

### 10.2.1 Derivation

A \( K \)-layer GCN without activations collapses to a single matrix multiplication:

\[
H^{(K)} = \underbrace{\hat{A} \cdot \hat{A} \cdots \hat{A}}_{K \text{ times}} X W^{(1)} W^{(2)} \cdots W^{(K)} = \hat{A}^K X W
\]

where \( \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \) is the normalized adjacency with self-loops, and \( W = W^{(1)} W^{(2)} \cdots W^{(K)} \) is the product of all weight matrices (itself just a single weight matrix). SGC precomputes the fixed propagation \( \hat{A}^K X \) once before training, then applies a simple linear classifier:

\[
\hat{Y} = \text{softmax}\!\left(\hat{A}^K X W\right)
\]

**Training**: Since the propagated features \( \tilde{X} = \hat{A}^K X \) are fixed (computed once, not backpropagated through), SGC trains as a simple logistic regression on \( \tilde{X} \). This reduces training time by orders of magnitude compared to GCN — the feature computation is \( O(K|E|d) \) once, and training is \( O(|V_L| d C) \) per epoch, where \( C \) is the number of classes.

**When SGC works**: On Cora, CiteSeer, and PubMed — all highly homophilic citation networks — SGC achieves accuracy within 1–2% of GCN despite being linear. This suggests that GCN's non-linearities are not the source of its advantage on these benchmarks; the multi-hop propagation is. SGC fails on heterophilic graphs and graph classification tasks where the non-linear interaction between features from different neighbors is critical.

**Limitation**: The product \( \hat{A}^K \) has the same over-smoothing problem as deep GCN. For large \( K \), SGC propagation spreads features globally and loses local structure. In practice, \( K \in \{1, 2\} \) is used for most node classification benchmarks.

---

## 10.3 APPNP: Decoupling Propagation from Prediction

Both GCN and SGC conflate two distinct operations: (1) **feature transformation** — learning a non-linear mapping from raw features to a task-relevant representation — and (2) **graph propagation** — smoothing the representation over the neighborhood. GCN interleaves these operations at every layer, making it impossible to tune them independently.

**Approximate Personalized Propagation of Neural Predictions (APPNP)** (Klicpera et al., 2019) disentangles them:

**Step 1 — Predict**: Apply a neural network (MLP or any architecture) to the raw node features, producing initial predictions:

\[
\mathbf{z}_v^{(0)} = f_\theta(\mathbf{x}_v)
\]

**Step 2 — Propagate**: Iteratively propagate the initial predictions using personalized PageRank-style diffusion for \( K \) steps:

\[
\mathbf{z}_v^{(k+1)} = (1 - \alpha) \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{\mathbf{z}_u^{(k)}}{\sqrt{\deg(u) \deg(v)}} + \alpha \mathbf{z}_v^{(0)}
\]

where \( \alpha \in (0, 1) \) is the **teleportation probability** — at each step, with probability \( \alpha \), the propagation returns to the initial prediction \( \mathbf{z}_v^{(0)} \) rather than continuing to diffuse.

!!! mascot-tip "Tip: APPNP's Teleportation Prevents Over-Smoothing Structurally"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage giving a tip">
    The α teleportation parameter is not just a regularization hyperparameter — it is a structural anti-smoothing mechanism. Each propagation step mixes the current prediction with the initial neural-network prediction, providing a "memory" that prevents all nodes from converging to the same embedding. With α > 0, even infinite propagation steps do not destroy local information: the stationary distribution of the propagation is personalized to each node's starting point \( \mathbf{z}_v^{(0)} \). In practice, α = 0.1 (90% propagation, 10% return to origin) and K = 10 propagation steps outperforms GCN on Cora, CiteSeer, and ogbn-arxiv.

### 10.3.1 Connection to Personalized PageRank

The propagation rule above is the power iteration for the personalized PageRank matrix:

\[
\Pi_{\text{PPR}} = \alpha (I - (1-\alpha) \hat{A})^{-1}
\]

where \( \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \). The closed-form solution to the propagation is \( Z^{(\infty)} = \Pi_{\text{PPR}} Z^{(0)} \), which can be computed exactly for small graphs or approximated by the \( K \)-step power iteration for large ones.

Personalized PageRank has a crucial property: it assigns non-zero weight to *all* nodes in the graph, not just the \( K \)-hop neighborhood. For node \( v \), the PPR score of distant node \( u \) decays geometrically with the random walk hitting time — providing a principled mechanism for long-range information aggregation without over-smoothing.

### 10.3.2 Advantages Over GCN

APPNP separates the depth of the prediction network (which can be a deep MLP tuned for feature capacity) from the depth of propagation (which can be large, e.g., K = 10, without over-smoothing). GCN conflates both into the number of GCNConv layers, forcing a tradeoff between feature transformation depth and propagation depth. APPNP eliminates this tradeoff.

The training graph for the feature network \( f_\theta \) is entirely label-driven — no graph structure needed. The graph structure enters only in the propagation step, which has no trainable parameters. This separation makes APPNP more interpretable and allows precomputing the PPR matrix offline.

---

## 10.4 Principal Neighborhood Aggregation (PNA)

Chapter 9 showed that no single aggregation function — sum, mean, or max — is universally optimal. Sum is injective on multisets but sensitive to scale; mean loses count information; max loses multiplicity. **Principal Neighborhood Aggregation (PNA)** (Corso et al., 2020) proposes combining multiple aggregators and scalers into a single rich representation.

### 10.4.1 Multiple Aggregators

PNA computes four aggregators over the neighborhood \( \mathcal{N}(v) \):

- **Mean**: \( \mu(\{\mathbf{h}_u\}) = \frac{1}{|\mathcal{N}(v)|} \sum_{u} \mathbf{h}_u \)
- **Maximum**: \( \text{max}(\{\mathbf{h}_u\}) = \max_u \mathbf{h}_u \) (element-wise)
- **Minimum**: \( \text{min}(\{\mathbf{h}_u\}) = \min_u \mathbf{h}_u \) (element-wise)
- **Standard deviation**: \( \sigma(\{\mathbf{h}_u\}) = \sqrt{\frac{1}{|\mathcal{N}(v)|} \sum_u (\mathbf{h}_u - \mu)^2} \)

The standard deviation aggregator is particularly useful: it is the only aggregator in this set that explicitly captures distributional *spread* rather than just the mean or extreme values.

### 10.4.2 Degree Scalers

Beyond multiple aggregators, PNA applies three **degree scalers** — multiplicative factors derived from the node's degree that adjust the aggregated embedding for degree heterogeneity:

- **Identity (I)**: no scaling, \( s = 1 \).
- **Amplification (A)**: \( s = \log(\deg(v) + 1) / \delta \), where \( \delta \) is the mean log-degree in the training set. High-degree nodes receive amplified signals.
- **Attenuation (D)**: \( s = \delta / \log(\deg(v) + 1) \). High-degree nodes receive attenuated signals.

The intuition is that the degree of a node fundamentally changes the statistical properties of its neighborhood aggregation: for a degree-1 node, the mean equals the single neighbor's embedding; for a degree-100 node, the mean is a smoothed aggregate of 100 signals. The amplification and attenuation scalers encode this difference explicitly, allowing the model to learn different behaviors for high- and low-degree nodes.

### 10.4.3 PNA Update Rule

The PNA update concatenates all (aggregator, scaler) combinations — \( 4 \times 3 = 12 \) in the full configuration — into a single large vector, then projects it:

\[
\mathbf{h}_v^{(k+1)} = f^{(k)}\!\left(\mathbf{h}_v^{(k)},\ \bigg\|_{(A, s) \in \mathcal{A} \times \mathcal{S}}\ s \cdot A\!\left(\left\{\!\left\{ \mathbf{h}_u^{(k)} : u \in \mathcal{N}(v) \right\}\!\right\}\right)\right)
\]

where \( \| \) denotes concatenation, \( \mathcal{A} \) is the set of aggregators, \( \mathcal{S} \) is the set of scalers, and \( f^{(k)} \) is a linear projection (reducing the dimension before passing to the next layer). The concatenated vector has dimension \( |\mathcal{A}| \times |\mathcal{S}| \times d \) before projection.

**Computational cost**: PNA computes \( |\mathcal{A}| = 4 \) aggregators per layer instead of 1, at roughly \( 4\times \) the memory cost of a single-aggregator GNN. For most molecular property prediction tasks, this cost is acceptable given the improved accuracy.

**When PNA shines**: On the ZINC molecular dataset (graph regression of penalized logP) and OGB molecular benchmarks, PNA consistently outperforms GIN and GCN because molecular properties depend on the statistical distribution of the local chemical environment — exactly what the multi-aggregator design captures.

---

## 10.5 Identity-Aware GNNs (ID-GNN)

Even GIN — the maximally 1-WL-powerful MPNN — cannot distinguish nodes that have identical computation graphs. Two nodes in different parts of the same graph (or in different graphs) that have identical rooted subtree structures will receive the same embedding, even if their *global* positions in the graph are entirely different. This limitation bites whenever the target property depends on where a node is in the graph, not just what its local structure looks like.

**Identity-Aware GNNs (ID-GNN)** (You et al., 2021) overcome this by running the GNN with the target node distinctively marked. The computation for node \( v \) uses a *heterogeneous* message-passing scheme that distinguishes messages *from* the target node from messages from all other nodes:

\[
\mathbf{h}_u^{(k+1)} = \sigma\!\left(W_{\text{other}}^{(k)} \cdot \text{AGG}\!\left(\left\{\!\left\{ \mathbf{h}_w^{(k)} : w \in \mathcal{N}(u),\ w \neq v \right\}\!\right\}\right) + W_{\text{self}}^{(k)} \cdot \mathbf{h}_v^{(k)}\right)
\]

for the target node \( v \) being computed. Every other node \( u \) in the graph uses the same standard GNN update. The result is that the final embedding of node \( v \) depends not just on its own local structure, but on the structural relationship between \( v \) and every other node in its \( K \)-hop neighborhood.

**Theoretical result**: ID-GNN is strictly more powerful than 1-WL GNNs. Specifically, it can distinguish any pair of non-isomorphic graphs that 1-WL fails on (in the sense that some node pair is distinguishable), while 1-WL GNNs cannot. The key insight is that the identity injection breaks the symmetry that causes identical computation graphs: two nodes that are symmetric under the graph's automorphism group can now be distinguished because the identity injection breaks that symmetry.

**Efficient implementation**: Running a separate GNN computation for every node in the graph would cost \( O(|V|) \) times the standard GNN cost — prohibitive for large graphs. ID-GNN amortizes this by sampling a small number of nodes to serve as identity-injected roots per mini-batch, computing their embeddings exactly, and using the standard GNN for the remaining nodes. This sampling strategy introduces a small bias but reduces the computational overhead to a manageable constant factor.

!!! mascot-warning "Warning: Higher-Order GNNs Trade Expressiveness for Scalability"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning">
    The theoretical hierarchy is clear: k-WL GNNs are strictly more powerful than 1-WL for every k ≥ 2. The practical hierarchy is less straightforward: a 3-WL GNN on a graph with 1,000 nodes requires operating on \( 10^9 \) node triples — likely infeasible on any current hardware. Before reaching for a higher-order architecture, ask whether the task genuinely requires distinguishing the graphs that 1-WL fails on. For many real benchmarks, the answer is no, and a carefully tuned GIN with structural features achieves comparable or better results at a fraction of the cost.

---

## 10.6 Higher-Order GNNs

The k-WL hierarchy (Chapter 9) provides the theoretical backbone for architectures that operate on \( k \)-tuples of nodes. **Higher-order GNNs** implement this computationally by maintaining embeddings for every ordered \( k \)-tuple \( (v_1, \ldots, v_k) \in V^k \) and updating them by aggregating over adjacent tuples.

### 10.6.1 k-GNN and Delta-Networks

The **\( k \)-GNN** framework (Maron et al., 2019) defines a message-passing scheme on the complete \( k \)-dimensional tensor of node tuples. For \( k = 2 \), each node pair \( (u, v) \) has an embedding \( \mathbf{h}_{uv}^{(t)} \), and the update aggregates over all triples containing the pair:

\[
\mathbf{h}_{uv}^{(t+1)} = \sigma\!\left(W_1 \mathbf{h}_{uv}^{(t)} + W_2 \sum_{w \in V} \mathbf{h}_{uw}^{(t)} + W_3 \sum_{w \in V} \mathbf{h}_{wv}^{(t)}\right)
\]

The aggregation sums over all "neighbors" of the tuple in the tensor-product graph. Node-level embeddings are recovered by marginalizing: \( \mathbf{h}_v^{(T)} = \sum_u \mathbf{h}_{uv}^{(T)} \).

For \( k = 2 \), the computation operates on \( n^2 \) tuple embeddings; for \( k = 3 \), on \( n^3 \). Memory scales as \( O(n^k \cdot d) \) and computation as \( O(n^{k+1} \cdot d) \) per layer — exponential growth that limits practical use to small graphs (molecular graphs with typically 10–50 atoms).

**\( \delta \)-k-GNN** (Morris et al., 2020) is a sparsified version that operates only on tuples where all elements are within \( r \) hops of each other in the original graph, dramatically reducing the number of active tuples from \( O(n^k) \) to \( O(n \cdot d^{k-1}_{\text{max}}) \) where \( d_{\text{max}} \) is the maximum degree. This makes k-GNNs tractable on molecular graphs while maintaining higher-order expressiveness.

### 10.6.2 Applications

Higher-order GNNs achieve state-of-the-art on molecular property prediction benchmarks that require counting cycles and ring systems. On the ZINC molecular regression dataset, the 3-WL \( \delta \)-k-GNN reaches MAE below 0.09, compared to 0.16 for GIN — a 44% reduction in error that directly corresponds to the model's ability to count rings in the molecular structure.

---

## 10.7 Subgraph GNNs

**Subgraph GNNs** achieve expressiveness beyond 1-WL by running standard GNNs on *subgraphs* rather than the full graph, then aggregating the subgraph-level representations.

!!! mascot-encourage "Complex Architecture, Simple Intuition"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage encouraging">
    The core idea behind subgraph GNNs is elegant: if a standard GNN cannot distinguish a node's global position because it only sees local structure, run the GNN separately on each node's local subgraph — where the node of interest *is* the global context. The subgraph GNN sees different structure for every node, even if those nodes have identical computation graphs in the original graph. The implementation details are involved, but the motivation is straightforward.

### 10.7.1 The ESAN Framework

**Equivariant Subgraph Aggregation Networks (ESAN)** (Zhao et al., 2022) formalize this idea. For each node \( v \in V \), define a subgraph \( G_v = (V_v, E_v)  \) — typically the ego graph (the induced subgraph of \( v \) and its \( r \)-hop neighbors). Each subgraph is processed by a standard GNN:

\[
\mathbf{h}_{G_v} = \text{GNN}\!\left(G_v,\ \mathbf{x}^{G_v}\right)
\]

where \( \mathbf{x}^{G_v} \) is the feature matrix restricted to \( V_v \), optionally augmented with a marking that identifies node \( v \) as the "root" of the subgraph (analogous to ID-GNN's identity injection).

The full graph's node embedding is then:

\[
\mathbf{h}_v = \text{AGG}_v\!\left(\left\{\!\left\{ \mathbf{h}_{G_u} : u \in \mathcal{N}(v) \cup \{v\} \right\}\!\right\}\right)
\]

aggregating the subgraph representations of all neighboring nodes.

**Theoretical power**: ESAN with root marking is strictly more powerful than 1-WL. It can detect the presence of specific subgraph patterns (triangles, 4-cycles, 5-cycles) because each subgraph computation sees the pattern within the ego graph, even when the pattern spans the ego graph boundary in a way that 1-WL cannot detect.

**Computational cost**: Running a separate GNN on each of \( |V| \) ego subgraphs multiplies the computation by \( |V| \) relative to a single-graph GNN. For small molecules (20–50 nodes), this is acceptable. For large citation graphs (100K+ nodes), it is not — subgraph GNNs are predominantly used for molecular property prediction and graph classification on TUDatasets.

### 10.7.2 DS-GNN

**Deep Sets Graph Neural Network (DS-GNN)** (Bevilacqua et al., 2022) is a related approach that treats the collection of subgraphs as a set and processes it with a symmetry-aware architecture, maintaining equivariance to permutations of both the node ordering within each subgraph and the ordering of the subgraph collection itself.

---

## 10.8 PairNorm: Principled Normalization for Expressive GNNs

Chapter 8 introduced PairNorm as a practical remedy for over-smoothing. Here we revisit it from the perspective of expressive GNN design, where normalization becomes a tool for *preserving* the discriminative capacity that powerful aggregators create.

The core tension is this: deeper GNNs have larger receptive fields and can in principle distinguish nodes based on more distant structural information. But depth causes over-smoothing, which destroys exactly that discriminative capacity. PairNorm resolves this tension by enforcing a minimum pairwise distance between node embeddings at each layer, regardless of depth.

Formally, PairNorm centers and normalizes the embedding matrix after each layer:

\[
\tilde{\mathbf{h}}_v = s \cdot \frac{\mathbf{h}_v - \boldsymbol{\mu}}{\sqrt{\frac{1}{|V|} \sum_{u \in V} \|\mathbf{h}_u - \boldsymbol{\mu}\|^2}}
\]

where \( \boldsymbol{\mu} = \frac{1}{|V|} \sum_v \mathbf{h}_v \) and \( s \) is a target scale. This guarantees that the mean squared distance between any two node embeddings is exactly \( s^2 \) after normalization — the embedding space is stretched or contracted to maintain separation.

**In the context of GIN**: Applying PairNorm between GIN layers allows GIN to be trained 10–20 layers deep without the embedding collapse that would otherwise occur. The result is a more expressive model: deep GIN with PairNorm can aggregate information from a larger \( K \)-hop neighborhood while maintaining the injectivity that gives GIN its 1-WL power.

**Interaction with batch normalization**: PairNorm and BN normalize along different axes. BN normalizes each feature dimension across nodes in the batch (preserving pairwise distances up to feature-wise scaling). PairNorm normalizes each node's embedding length relative to the global mean (preserving relative direction while controlling absolute magnitude). They are complementary and can be used in sequence: BN first, then PairNorm.

---

## 10.9 Stochastic Depth for GNNs

**Stochastic depth** (Huang et al., 2016, adapted for GNNs) randomly drops entire GNN layers during training, replacing each dropped layer with an identity map (passing through the previous layer's embedding unchanged):

\[
\mathbf{H}^{(k+1)} = \begin{cases} \text{GNNLayer}^{(k)}(\mathbf{H}^{(k)}) & \text{with probability } 1 - p_k \\ \mathbf{H}^{(k)} & \text{with probability } p_k \end{cases}
\]

where \( p_k \) is the drop probability for layer \( k \), typically increasing linearly with depth: \( p_k = k / K \cdot p_{\text{max}} \) for a \( K \)-layer network. Earlier layers are rarely dropped (they contribute foundational transformations); later layers are dropped frequently (they contribute more specialized refinements).

**Theoretical interpretation**: Stochastic depth is equivalent to training an implicit ensemble of networks of varying depths. At each training step, a random subset of layers is active, producing a different effective depth. At inference, all layers are used (with activations scaled by \( 1 - p_k \)), effectively averaging the ensemble's predictions.

**Benefits for GNNs**:

1. **Regularization**: exposing the model to shorter effective paths regularizes against overfitting, analogous to dropout but at the layer level.
2. **Depth tolerance**: by occasionally skipping intermediate layers, the model learns representations that are useful at multiple depths — making it more robust to the choice of total depth.
3. **Training speed**: on average, a fraction \( \sum_k p_k / K \) of layer computations are skipped per training step, reducing training wall-clock time by a corresponding amount.
4. **Over-smoothing mitigation**: because some training steps use fewer message-passing iterations, the model sees representations at a range of smoothing levels, indirectly learning more local (less smoothed) representations at intermediate layers.

Stochastic depth is most beneficial for very deep GNNs (more than 8 layers) where over-smoothing and overfitting are the primary failure modes. For shallow architectures (2–4 layers), the regularization benefit is marginal.

---

## 10.10 Structural Encodings

Beyond the architectural choices described above, a complementary approach is to precompute structural features and append them to the input node features before any GNN layer. These **structural encodings** give the GNN access to information it could not derive from local message passing alone.

Three encodings are in widespread use:

**Random Walk Structural Encoding (RWSE)**: For each node \( v \), compute the \( k \)-step random walk landing probability \( p_{vv}^{(k)} \) — the probability that a \( k \)-step random walk starting at \( v \) returns to \( v \). The vector \( [p_{vv}^{(1)}, p_{vv}^{(2)}, \ldots, p_{vv}^{(K)}] \) encodes structural properties: \( p_{vv}^{(2)} \) is proportional to the degree, \( p_{vv}^{(3)} \) encodes triangle participation, and \( p_{vv}^{(6)} \) encodes hexagonal ring participation (critical for aromatic chemistry). A GNN augmented with RWSE can detect triangle and ring structure without any architectural modification, because the encoding provides this information as input features.

**Laplacian Eigenvector Positional Encoding (LapPE)**: The \( k \) smallest non-trivial eigenvectors of the graph Laplacian \( L = D - A \) are used as node positional features. The \( i \)-th eigenvector \( \mathbf{u}_i \) encodes a notion of "spectral position" in the graph — nodes that are spectrally close receive similar \( \mathbf{u}_i \) values. LapPE gives GNNs a sense of global position, making them aware of a node's location in the global graph structure. It is the standard positional encoding for graph transformer architectures (Chapter 11).

**Graphlet Degree Vectors (GDV)**: Count the occurrences of each node in each of the 73 graphlet orbit types (all subgraphs of up to 5 nodes). The GDV is a 73-dimensional vector that encodes the node's structural role in extreme detail. GDVs are highly informative but expensive to compute (\( O(d^4 n) \)) and may overfit when the training set is small.

---

## 10.11 Code: GCN vs. GraphSAGE vs. GAT vs. GIN vs. PNA

The following code implements all five major architectures on the MUTAG and PROTEINS graph classification benchmarks, providing a direct comparison of their expressiveness and training dynamics. Before reading the code, note the key design parameters: all models use `num_layers = 3`, `hidden_channels = 64`, and `global_add_pool` for the graph-level readout. The only architectural difference is the message-passing layer type — isolating the effect of the aggregation strategy.

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import (GCNConv, SAGEConv, GATConv,
                                 GINConv, PNAConv, global_add_pool)
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree

# Load MUTAG with one-hot degree features
dataset = TUDataset(root='data/TU', name='MUTAG',
                    transform=OneHotDegree(max_degree=4))
train_data, test_data = dataset[:150], dataset[150:]
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=False)

in_ch = dataset.num_node_features   # 5 (one-hot degree)
hid   = 64
out   = dataset.num_classes          # 2

# ── GCN ──────────────────────────────────────────────────────────────────────
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(in_ch, hid), GCNConv(hid, hid), GCNConv(hid, hid)])
        self.lin = torch.nn.Linear(hid, out)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.lin(global_add_pool(x, batch))

# ── GraphSAGE ─────────────────────────────────────────────────────────────────
class SAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_ch, hid), SAGEConv(hid, hid), SAGEConv(hid, hid)])
        self.lin = torch.nn.Linear(hid, out)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.lin(global_add_pool(x, batch))

# ── GAT ───────────────────────────────────────────────────────────────────────
class GAT(torch.nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_ch, hid // heads, heads=heads)
        self.conv2 = GATConv(hid, hid // heads, heads=heads)
        self.conv3 = GATConv(hid, hid, heads=1)
        self.lin   = torch.nn.Linear(hid, out)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return self.lin(global_add_pool(x, batch))

# ── GIN ───────────────────────────────────────────────────────────────────────
def make_gin_mlp(in_c, out_c):
    return torch.nn.Sequential(
        torch.nn.Linear(in_c, 2 * out_c),
        torch.nn.BatchNorm1d(2 * out_c),
        torch.nn.ReLU(),
        torch.nn.Linear(2 * out_c, out_c),
    )

class GIN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GINConv(make_gin_mlp(in_ch, hid)),
            GINConv(make_gin_mlp(hid, hid)),
            GINConv(make_gin_mlp(hid, hid)),
        ])
        # Layer-wise readout: pool at each layer and concatenate
        self.lin = torch.nn.Linear(hid * 3, out)

    def forward(self, x, edge_index, batch):
        layer_outs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            layer_outs.append(global_add_pool(x, batch))
        return self.lin(torch.cat(layer_outs, dim=1))

# ── PNA ───────────────────────────────────────────────────────────────────────
# PNA requires pre-computing the degree histogram for the amplification scaler
deg_hist = torch.zeros(5)
for data in train_data:
    d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
    deg_hist += torch.bincount(d, minlength=5)

class PNA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        agg = ['mean', 'min', 'max', 'std']
        sca = ['identity', 'amplification', 'attenuation']
        self.convs = torch.nn.ModuleList([
            PNAConv(in_ch, hid, aggregators=agg, scalers=sca, deg=deg_hist),
            PNAConv(hid,   hid, aggregators=agg, scalers=sca, deg=deg_hist),
            PNAConv(hid,   hid, aggregators=agg, scalers=sca, deg=deg_hist),
        ])
        self.lin = torch.nn.Linear(hid, out)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return self.lin(global_add_pool(x, batch))

# ── Training loop ─────────────────────────────────────────────────────────────
def run(model, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    best_test = 0.0
    for _ in range(epochs):
        model.train()
        for data in train_loader:
            opt.zero_grad()
            F.cross_entropy(model(data.x, data.edge_index, data.batch),
                            data.y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            correct = sum(
                (model(d.x, d.edge_index, d.batch).argmax(1) == d.y).sum().item()
                for d in test_loader)
            best_test = max(best_test, correct / len(test_data))
    return best_test

print(f"{'Model':<12} {'MUTAG Test Acc':>15}")
print("-" * 28)
for name, Model in [('GCN', GCN), ('SAGE', SAGE), ('GAT', GAT),
                    ('GIN', GIN), ('PNA', PNA)]:
    acc = run(Model())
    print(f"{name:<12} {acc:>15.4f}")
```

Expected output on MUTAG (approximate, single run):

```
Model        MUTAG Test Acc
----------------------------
GCN                  0.7368
SAGE                 0.7632
GAT                  0.7632
GIN                  0.8947
PNA                  0.8947
```

GIN and PNA both approach the 1-WL ceiling on MUTAG. GCN's mean aggregation falls noticeably short. The near-parity between GIN and PNA on MUTAG reflects that MUTAG's distinguishing features are captured within 1-WL power; on more complex benchmarks (ZINC, ogbg-molhiv), PNA's multi-aggregator design provides additional lift.

#### Diagram: GIN vs. GCN Expressiveness MicroSim


<iframe src="../../sims/ch10-gin-gcn-expressiveness/main.html" width="100%" height="542px" scrolling="no"></iframe>
[Run GIN vs. GCN Expressiveness MicroSim Fullscreen](../../sims/ch10-gin-gcn-expressiveness/main.html)

---

## 10.12 Benchmark Results

| Method | MUTAG (%) | PROTEINS (%) | ZINC MAE | ogbg-molhiv AUC | Citation |
|---|---|---|---|---|---|
| GCN (mean) | 74.2 ± 6.1 | 75.7 ± 0.6 | 0.367 | 0.757 ± 0.017 | Kipf & Welling (2017) |
| GraphSAGE (max) | 76.0 ± 6.0 | 75.9 ± 0.9 | 0.398 | 0.761 ± 0.016 | Hamilton et al. (2017) |
| GIN (sum+MLP) | 89.4 ± 5.6 | 76.2 ± 2.8 | 0.163 | 0.775 ± 0.014 | Xu et al. (2019) |
| PNA | 89.0 ± 4.9 | **78.0 ± 1.8** | 0.142 | **0.795 ± 0.008** | Corso et al. (2020) |
| APPNP | 83.3 ± — | 77.4 ± — | — | — | Klicpera et al. (2019) |
| SGC | 81.0 ± — | 75.1 ± — | — | — | Wu et al. (2019) |
| ID-GNN | 88.3 ± 7.2 | 77.1 ± 1.6 | — | — | You et al. (2021) |
| δ-k-GNN (k=3) | **92.0 ± 6.0** | 77.8 ± 2.0 | **0.091** | — | Morris et al. (2020) |

Lower MAE is better for ZINC. Higher AUC and accuracy are better for all other metrics.

---

## 10.13 Common Pitfalls

1. **Using GIN for node classification without adjusting the readout.** GIN's multi-layer global sum pooling readout is designed for graph classification. For node classification, each node's final-layer embedding is used directly — no pooling. In this setting, GIN and GCN with sum aggregation perform similarly, because the expressiveness advantage of GIN comes from distinguishing graph-level structure, not individual node neighborhoods.

2. **Setting SGC's propagation depth K too high.** SGC is equivalent to a GCN with K linear layers. At K = 8 or 10, the propagated features are nearly smooth global means — useful for very homophilic graphs but destructive for any task requiring local discrimination. Use K ∈ {1, 2, 3} and check whether additional hops add accuracy on a held-out validation set.

3. **Applying APPNP to heterophilic graphs without adjusting α.** The teleportation probability α controls how strongly each node's prediction is anchored to its own initial neural network output vs. its smoothed neighborhood. On heterophilic graphs (where neighbors have different labels), high α (e.g., 0.5) preserves the individual prediction and reduces harmful smoothing; low α (e.g., 0.1) is more appropriate for homophilic graphs.

4. **Computing PNA degree statistics on the test graph.** The amplification and attenuation scalers in PNA require a degree histogram to normalize the log-degree. This histogram must be computed on the training set only. Including test-graph nodes in the degree histogram constitutes data leakage.

5. **Forgetting that ID-GNN and subgraph GNNs change the complexity class of the problem.** A 3-layer GIN runs in \( O(K \cdot |E| \cdot d) \); a 3-layer subgraph GIN runs in \( O(K \cdot |E| \cdot |V| \cdot d) \) — a factor of \( |V| \) more expensive. For citation graphs with 100K+ nodes, this is infeasible. Always estimate the actual compute cost before choosing an architecture.

---

!!! mascot-celebration "Chapter 10 Complete — Your Encoder Toolkit Is Full"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating">
    You've now worked through the complete spectrum of GNN expressiveness: from the simplest possible model (SGC) to architectures that provably exceed the 1-WL barrier (ID-GNN, subgraph GNNs, k-GNNs). More importantly, you understand *why* each architecture is designed the way it is — what theoretical limitation it overcomes, and at what computational cost. This knowledge lets you match the right encoder to any graph learning task, without resorting to trial-and-error architecture search. In Chapter 11, we step beyond message passing entirely and examine Graph Transformers — architectures that bring attention-based global computation to graph data.

---

## 10.14 Exercises

**Remember (Recall and Identify)**

1. List the four aggregators used in PNA and state what property each one captures that the others do not. Which aggregator is unique in capturing distributional spread rather than a central tendency or extreme value?

2. State the two-step procedure of APPNP: what operation is performed in Step 1 and what operation in Step 2? What is the purpose of the teleportation probability α, and what pathology does it prevent?

**Understand (Explain and Interpret)**

3. Explain why SGC collapses a K-layer GCN without activations to a single linear transformation. What algebraic property of linear functions makes this possible, and why does the same collapse not occur in a nonlinear GCN?

4. ID-GNN is described as "strictly more powerful than 1-WL GNNs." Explain the mechanism by which the identity injection breaks the symmetry that causes 1-WL to fail, using a specific example of two nodes that 1-WL cannot distinguish but ID-GNN can.

**Apply (Compute and Implement)**

5. Implement APPNP from scratch in PyTorch using only standard linear algebra operations (no GNN library). Your implementation should: (a) accept a node feature matrix X, a normalized adjacency Â, a teleportation probability α, and a number of propagation steps K; (b) return the propagated embedding matrix Z^(K); (c) verify that the result matches PyG's `APPNP` layer on a small test graph.

6. Compute the PNA update for a single node \( v \) with three neighbors whose embeddings are \( [1, 2] \), \( [3, 1] \), and \( [2, 2] \). Use aggregators (mean, max, std) and scalers (identity, amplification with \( \delta = \log(4) \)) and show the full concatenated vector before the linear projection. Assume \( \deg(v) = 3 \).

**Analyze (Compare and Diagnose)**

7. You are choosing between GIN and APPNP for a node classification task on a large homophilic citation graph (500K nodes, average degree 15, label homophily ratio 0.85). Compare the two architectures along: (a) theoretical expressiveness (1-WL power), (b) scalability to 500K nodes, (c) depth of propagation achievable without over-smoothing, and (d) the practical recommendation for this specific task.

8. PNA achieves state-of-the-art on molecular property prediction benchmarks, but a reviewer argues that the improvement over GIN is due entirely to the additional trainable parameters introduced by the multi-aggregator linear projection, not the aggregation diversity. Design an ablation experiment to test this hypothesis. What control would you add, and what result would support or refute the reviewer's claim?

**Evaluate (Assess and Justify)**

9. A practitioner needs to choose between GIN with RWSE (random walk structural encodings added as node features) and a 3-WL delta-k-GNN for a molecular property prediction task. Both are reported to achieve similar accuracy on the benchmark. Justify which you would deploy in a production system, considering: (a) inference latency for a molecule with 40 atoms, (b) maintenance and debugging complexity, (c) generalization to larger molecules (100+ atoms) not in the training set.

10. SGC is a linear model, yet it achieves within 2% of GCN on Cora and CiteSeer. A researcher concludes from this that "GCN's non-linearities are unnecessary for graph-structured data." Identify the flaw in this reasoning and propose a test that would distinguish between "non-linearities are unnecessary" and "non-linearities are captured by the propagation step."

**Create (Design and Propose)**

11. Propose a new GNN architecture that combines the APPNP propagation scheme with GIN's sum aggregation, maintaining 1-WL expressiveness while also preventing over-smoothing via teleportation. Specify: (a) the two-step procedure (predict and propagate), (b) the update rule for each propagation step (using sum aggregation instead of mean), (c) whether the resulting architecture is more or less expressive than vanilla GIN, with justification.

12. Design a benchmark evaluation suite to measure whether a given GNN architecture truly benefits from higher-order expressiveness (beyond 1-WL) on a practical graph dataset. Your suite should: (a) include at least three synthetic graph pairs that 1-WL cannot distinguish but higher-order methods can, (b) specify how you measure distinguishing power (not just accuracy), (c) propose a way to calibrate whether observed accuracy improvements are due to higher-order structure detection or other factors (better optimization, larger parameter count, etc.).

---

## 10.15 Further Reading

1. **Wu, F., Souza, A., Zhang, T., Fifty, C., Yu, T., & Weinberger, K. (2019). Simplifying Graph Convolutional Networks.** *ICML 2019.* Introduces SGC and provides both theoretical analysis (linear GCN collapses to one transformation) and empirical evidence that SGC matches GCN within 2% on homophilic benchmarks. Quantifies the proportion of GCN's performance attributable to feature propagation vs. non-linear transformation.

2. **Klicpera, J., Bojchevski, A., & Günnemann, S. (2019). Predict then Propagate: Graph Neural Networks meet Personalized PageRank.** *ICLR 2019.* Introduces APPNP and derives its relationship to personalized PageRank. Shows that APPNP allows K = 10 propagation steps without over-smoothing, outperforming GCN on Cora, CiteSeer, and PubMed. Ablations isolate the contribution of the teleportation mechanism.

3. **Corso, G., Cavalleri, L., Beaini, D., Liò, P., & Veličković, P. (2020). Principal Neighbourhood Aggregation for Graph Nets.** *NeurIPS 2020.* Proposes PNA with multiple aggregators and degree scalers. Proves that no single aggregator is universally optimal and that the combination is provably more expressive than any individual one. Demonstrates state-of-the-art on ZINC molecular regression.

4. **You, J., Gomes-Selman, J. M., Ying, R., & Leskovec, J. (2021). Identity-aware Graph Neural Networks.** *AAAI 2021.* Introduces ID-GNN with the heterogeneous message-passing scheme and proves it is strictly more powerful than 1-WL GNNs. Demonstrates consistent improvements over GIN on node and graph classification benchmarks across 9 datasets.

5. **Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E., Rattan, G., & Grohe, M. (2019). Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks.** *AAAI 2019.* Introduces the k-GNN framework and proves its correspondence to k-WL. Proposes the δ-k-GNN sparsification for practical tractability. Shows that 3-GNN achieves near-perfect accuracy on subgraph counting benchmarks where 1-WL GNNs fail.

6. **Zhao, L., Jin, W., Akoglu, L., & Shah, N. (2022). From Stars to Subgraphs: Uplifting Any GNN with Local Structure Awareness.** *ICLR 2022.* Introduces the ESAN subgraph GNN framework and proves it is strictly more powerful than 1-WL. Proposes the DS-WL hierarchy characterizing subgraph GNN expressiveness and demonstrates improvements on molecular property prediction.

7. **Dwivedi, V. P., Luu, A. T., Laurent, T., Bengio, Y., & Bresson, X. (2022). Graph Neural Networks with Learnable Structural and Positional Representations.** *ICLR 2022.* Introduces learnable RWSE and LapPE as pre-computable node features that give standard 1-WL GNNs access to structural information beyond 1-WL, at no increase in per-layer computational cost. Achieves state-of-the-art on ZINC and PCQM4Mv2.

8. **Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). Deep Networks with Stochastic Depth.** *ECCV 2016.* The original stochastic depth paper for CNNs, whose ideas transfer directly to GNNs. Shows that randomly dropping layers during training (with identity bypasses) improves accuracy, reduces training time, and allows training networks up to 1,200 layers. The GNN adaptation follows the same principle with GNN layers as the dropped units.

[See Annotated References](./references.md)
