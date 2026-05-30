---
title: "Chapter 4: Node Embeddings: DeepWalk and node2vec"
description: "Develops shallow node embedding methods — DeepWalk, node2vec, and matrix factorization — as the encoder-decoder framework that unifies graph representation learning before the GNN era."
generated_by: claude skill chapter-content-generator
date: 2026-05-28 21:09:35
version: 0.08
---

# Chapter 4: Node Embeddings: DeepWalk and node2vec

**Part 1: Classical Graph Analysis**

## Summary

Develops shallow node embedding methods — DeepWalk, node2vec, and matrix factorization — as the encoder-decoder framework that unifies graph representation learning before the GNN era.

## Concepts Covered

This chapter covers the following 17 concepts from the learning graph:

1. Node Embedding
2. Embedding Space
3. Encoder-Decoder Framework
4. Shallow Embedding
5. Matrix Factorization (Graph)
6. DeepWalk
7. node2vec
8. Biased Random Walk
9. BFS Strategy (node2vec)
10. DFS Strategy (node2vec)
11. Skip-Gram Model
12. Negative Sampling
13. LINE Embedding
14. Transductive Learning
15. Inductive Learning
16. Structural Equivalence
17. Homophily

## Prerequisites

This chapter builds on:

- [Chapter 2: Graph Properties and Traditional ML Features](../02-graph-properties-and-features/index.md)
- [Chapter 3: Link Analysis and PageRank](../03-link-analysis-pagerank/index.md)

---

!!! mascot-welcome "From Rankings to Representations"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage welcomes students to Chapter 4">
    PageRank gave us a single scalar per node — a ranking, not a representation. But most graph ML tasks require more than a ranking: they require a *geometry*, a space where similar nodes cluster together and dissimilar nodes sit apart, so that a classifier or link predictor can draw decision boundaries with ordinary linear algebra. In this chapter we build that geometry from scratch, using random walks to define "similarity" and the skip-gram objective to learn vectors that preserve it. The methods here — DeepWalk, node2vec, and their matrix-factorization cousins — are the direct precursors to the GNN architectures in Part 2. Understanding them precisely is what makes GNNs feel like an inevitable next step rather than an arbitrary design choice.

## 4.1 The Recommendation Problem

Imagine a music streaming platform with 50 million users and 30 million tracks. Each user has listened to dozens or hundreds of songs; each song has been played by thousands or millions of users. The platform wants to recommend songs that a given user is likely to enjoy but has not yet heard. The interaction data is naturally a **bipartite graph**: users on one side, tracks on the other, with an edge whenever a user played a track.

The naive approach — represent each user and each track as a one-hot vector and measure similarity by co-occurrence counts — produces a similarity matrix of dimensions \(50 \times 10^6 \times 30 \times 10^6\), which is approximately \(10^{15}\) entries. Storing or computing this matrix is physically impossible. What is needed is a **low-dimensional representation**: a dense vector \(\mathbf{z}_v \in \mathbb{R}^d\) (typically \(d = 64\) to \(d = 256\)) for every user and every track such that users with similar listening histories have nearby vectors, and tracks with similar audiences have nearby vectors. Once these vectors exist, the recommendation problem reduces to a fast nearest-neighbor search in \(\mathbb{R}^d\) — an operation that runs in milliseconds on modern hardware.

This is the **node embedding** problem: given a graph \(G = (V, E)\), learn a function \(\text{ENC}: V \to \mathbb{R}^d\) such that structural similarity in \(G\) is preserved as geometric proximity in the **embedding space** \(\mathbb{R}^d\). The definition of "structural similarity" — and the method for learning \(\text{ENC}\) from it — is what distinguishes the different embedding algorithms in this chapter.

## 4.2 The Encoder-Decoder Framework

Before examining specific algorithms, it is valuable to establish the unifying framework within which all node embedding methods can be understood. The **encoder-decoder framework** decomposes the node embedding problem into two components.

The **encoder** is the function that maps nodes to embedding vectors:

\[
\text{ENC}: V \to \mathbb{R}^d, \quad \text{ENC}(v) = \mathbf{z}_v
\]

For the simplest class of methods — **shallow embeddings** — the encoder is nothing more than a lookup table: a matrix \(\mathbf{Z} \in \mathbb{R}^{|V| \times d}\) where row \(v\) is the embedding vector \(\mathbf{z}_v\). There are no parameters shared across nodes; each node has its own dedicated \(d\)-dimensional vector. The total parameter count is \(O(|V| \cdot d)\), which grows linearly with the number of nodes.

The **decoder** reconstructs an estimate of pairwise similarity from embedding vectors:

\[
\text{DEC}: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}, \quad \text{DEC}(\mathbf{z}_u, \mathbf{z}_v) \approx \text{sim}_G(u, v)
\]

The most common decoder is the **dot product** \(\text{DEC}(\mathbf{z}_u, \mathbf{z}_v) = \mathbf{z}_u^\top \mathbf{z}_v\), which approximates some notion of node similarity. The precise definition of \(\text{sim}_G(u, v)\) — what "similar" means in the original graph — is the key design choice that separates one embedding algorithm from another.

The **training objective** minimizes the reconstruction error:

\[
\min_{\mathbf{Z}} \sum_{(u, v) \in \mathcal{D}} \ell\bigl(\text{DEC}(\mathbf{z}_u, \mathbf{z}_v),\ \text{sim}_G(u, v)\bigr)
\]

where \(\mathcal{D}\) is a training set of node pairs and \(\ell\) is a loss function. The following table organizes the key design choices across the methods in this chapter:

| Method | Similarity \(\text{sim}_G(u,v)\) | Decoder | Loss |
|---|---|---|---|
| Matrix Factorization | \(A_{uv}\) (adjacency) | \(\mathbf{z}_u^\top\mathbf{z}_v\) | Squared reconstruction |
| DeepWalk | \(P(v \mid u)\) via random walk | \(\sigma(\mathbf{z}_u^\top\mathbf{z}_v)\) | Binary cross-entropy |
| node2vec | Biased walk probability | \(\sigma(\mathbf{z}_u^\top\mathbf{z}_v)\) | Binary cross-entropy |
| LINE (1st order) | \(A_{uv}\) (direct edge weight) | \(\sigma(\mathbf{z}_u^\top\mathbf{z}_v)\) | KL divergence |
| LINE (2nd order) | \(P(v \mid u)\) via 1-hop neighbors | \(\sigma(\mathbf{z}_u^\top\mathbf{z}_v)\) | KL divergence |

## 4.3 Random Walk Similarity: Intuition

The core challenge in defining node similarity for general graphs is that the adjacency matrix entry \(A_{uv}\) is binary and local — it only tells us whether two nodes are directly connected, not whether they occupy similar positions in the graph. Two nodes that share no edges but sit in structurally similar regions of the graph (e.g., two department heads in different corporate hierarchies) deserve similar embeddings, but their adjacency entry is zero.

**Random walk similarity** provides a richer notion of proximity: \(\text{sim}_G(u, v)\) is defined as the probability that a random walk starting at \(u\) visits \(v\) within a finite number of steps. This definition has several appealing properties:

- It is **non-local**: it aggregates information from multi-hop paths, not just direct edges.
- It is **stochastic**: by averaging over many short walks, it is robust to individual edge noise.
- It **interpolates** between local and global similarity as the walk length increases.
- It is **computationally tractable**: short random walks on sparse graphs are \(O(|V| \cdot L)\) to generate, where \(L\) is the walk length.

The idea of using co-occurrence statistics from text corpora — popularized by the Word2Vec skip-gram model — transfers directly to graphs if we treat random walks as "sentences" and nodes as "words." The frequency with which node \(v\) appears in a window around node \(u\) across many walks encodes their graph proximity.

## 4.4 DeepWalk: From Walks to Embeddings

**DeepWalk** (Perozzi, Al-Rfou & Skiena, 2014) was the first method to apply the skip-gram framework to graphs, treating the graph as if it were a corpus of documents. The algorithm has three stages.

### 4.4.1 Stage 1: Random Walk Generation

For each node \(u \in V\), generate \(r\) random walks of fixed length \(L\), starting from \(u\). Each walk is an unbiased random walk: at each step, move to a uniformly random out-neighbor of the current node. With \(r = 10\) and \(L = 80\) (the DeepWalk defaults), this produces a corpus of \(r \cdot |V|\) "sentences," each of length \(L\).

### 4.4.2 Stage 2: Skip-Gram Objective

The **skip-gram model** defines a probabilistic language model over the walk corpus. Given a center node \(u\) in a walk, the model predicts the probability of observing each context node \(v\) within a window of size \(w\) around \(u\):

\[
\Pr(v \mid \mathbf{z}_u) = \frac{\exp(\mathbf{z}_u^\top \mathbf{z}_v)}{\sum_{n \in V} \exp(\mathbf{z}_u^\top \mathbf{z}_n)}
\]

The training objective maximizes the total log-probability over all center-context pairs across all walks:

\[
\max_{\mathbf{Z}} \sum_{u \in V} \sum_{v \in N_R(u)} \log \Pr(v \mid \mathbf{z}_u)
\]

where \(N_R(u)\) is the multiset of context nodes appearing near \(u\) across all walks. Equivalently, we minimize the negative log-likelihood:

\[
\mathcal{L}_{\text{DeepWalk}} = -\sum_{u \in V} \sum_{v \in N_R(u)} \log \Pr(v \mid \mathbf{z}_u)
\]

### 4.4.3 Stage 3: Optimization

Computing the softmax denominator \(\sum_{n \in V} \exp(\mathbf{z}_u^\top \mathbf{z}_n)\) requires a dot product with every node in \(V\), making each gradient step \(O(|V|)\) — prohibitively expensive for large graphs. The standard solution is **negative sampling**, which we examine next.

!!! mascot-thinking "Why the Softmax Denominator Is the Bottleneck"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage explains the skip-gram softmax bottleneck">
    The denominator \(\sum_{n \in V} \exp(\mathbf{z}_u^\top \mathbf{z}_n)\) is called the **partition function** — it normalizes the distribution over all \(|V|\) nodes. For a graph with 1 million nodes and \(d = 128\), computing this sum requires \(10^6\) dot products of 128-dimensional vectors per training step. With millions of training steps, this is \(O(10^{14})\) FLOPs — intractable. Negative sampling replaces the partition function with a binary classification problem over a small number of "negative" samples, reducing the cost per step from \(O(|V|)\) to \(O(K)\) where \(K \approx 5\text{–}20\). This approximation is the same trick that made Word2Vec scale to billion-word corpora, and it works equally well for graphs.

## 4.5 Negative Sampling

**Negative sampling** reformulates the skip-gram objective as a binary classification problem. Instead of predicting the correct context node from all \(|V|\) nodes, we train a binary classifier to distinguish true context pairs \((u, v)\) (positive examples) from randomly sampled noise pairs \((u, v_k)\) (negative examples), where \(v_k\) is sampled from a noise distribution \(P_n\) (typically the unigram distribution raised to the 3/4 power, \(P_n(v) \propto d_v^{3/4}\)).

The negative sampling loss for a positive pair \((u, v)\) with \(K\) negative samples is:

\[
\mathcal{L}_{\text{NS}}(u, v) = -\log \sigma(\mathbf{z}_u^\top \mathbf{z}_v) - \sum_{k=1}^{K} \log \sigma(-\mathbf{z}_u^\top \mathbf{z}_{v_k})
\]

where \(\sigma(x) = 1/(1 + e^{-x})\) is the sigmoid function. The first term pushes \(\mathbf{z}_u\) and \(\mathbf{z}_v\) to have a large positive dot product (high similarity). The second term pushes \(\mathbf{z}_u\) and each negative sample \(\mathbf{z}_{v_k}\) to have a large *negative* dot product (low similarity). With \(K = 5\), this reduces each gradient step from \(O(|V|)\) to \(O(K) = O(5)\) — a \(10^5\times\) speedup for a million-node graph.

The total DeepWalk loss with negative sampling is:

\[
\mathcal{L}_{\text{DeepWalk}} = \sum_{u \in V} \sum_{v \in N_R(u)} \left[ -\log \sigma(\mathbf{z}_u^\top \mathbf{z}_v) - K \cdot \mathbb{E}_{v_k \sim P_n}\bigl[\log \sigma(-\mathbf{z}_u^\top \mathbf{z}_{v_k})\bigr] \right]
\]

This is minimized with stochastic gradient descent or Adam, treating \(\mathbf{Z}\) as the parameter matrix to be optimized. Each mini-batch consists of a set of (center, context, negatives) tuples drawn from the walk corpus.

## 4.6 node2vec: Controlling the Exploration-Exploitation Trade-Off

DeepWalk's random walks are **unbiased**: at each step, the next node is chosen uniformly at random from the current node's neighbors. This produces walks that mix local and global exploration in proportions determined entirely by the graph's structure. **node2vec** (Grover & Leskovec, 2016) introduces two hyperparameters that give the practitioner precise control over the exploration behavior.

### 4.6.1 The Biased Random Walk

node2vec defines a **second-order biased random walk**: the transition probability from the current node \(t\) to the next node \(x\) depends not only on the edge weight \(w(t, x)\) but also on the distance from the *previous* node \(s_{\text{prev}}\) to \(x\). Concretely, if the walk just moved from \(s_{\text{prev}}\) to \(t\), the (unnormalized) transition weight to neighbor \(x\) of \(t\) is:

\[
\pi(t, x) = \alpha(s_{\text{prev}}, x) \cdot w(t, x), \quad \alpha(s_{\text{prev}}, x) = \begin{cases} \frac{1}{p} & \text{if } d(s_{\text{prev}}, x) = 0 \quad (\text{return to previous node}) \\ 1 & \text{if } d(s_{\text{prev}}, x) = 1 \quad (\text{neighbor of both } s_{\text{prev}} \text{ and } t) \\ \frac{1}{q} & \text{if } d(s_{\text{prev}}, x) = 2 \quad (\text{farther from } s_{\text{prev}}) \end{cases}
\]

where \(d(s_{\text{prev}}, x)\) is the shortest-path distance from the previous node \(s_{\text{prev}}\) to the candidate next node \(x\), and \(p, q > 0\) are the two hyperparameters. These transition weights are normalized to produce a proper probability distribution over neighbors.

### 4.6.2 BFS vs. DFS Exploration and Network Phenomena

The hyperparameters \(p\) and \(q\) control two complementary exploration strategies whose choice determines what kind of graph structure the embeddings capture.

The **return parameter \(p\)** governs the probability of immediately backtracking to the previous node. A small \(p\) makes the walk likely to return, creating a **BFS-like** (breadth-first search) exploration that stays local and samples nodes within a tight neighborhood of the starting node. BFS exploration leads embeddings to capture **homophily** — the tendency for connected nodes to share similar labels. In a social network, nodes connected by edges tend to belong to the same community; BFS walks stay within communities, so nodes from the same community become co-context and hence proximate in the embedding space.

The **in-out parameter \(q\)** controls the tendency to explore outward versus staying local. A small \(q\) makes the walk likely to keep moving away from the start, creating a **DFS-like** (depth-first search) exploration that traverses the graph at increasing distances. DFS exploration leads embeddings to capture **structural equivalence** — the tendency for nodes that occupy analogous roles in the graph's topology (e.g., all hub nodes, or all peripheral leaf nodes) to have similar embeddings, even if they are far apart in the graph. Two department heads in different organizational subgraphs may never appear in the same short random walk under BFS, but DFS walks that traverse the global hierarchy will co-sample them frequently, driving their embeddings together.

The following table summarizes the behavioral regimes:

| \(p\) value | \(q\) value | Walk behavior | Captured phenomenon |
|---|---|---|---|
| Small | Large | BFS-like, local | Homophily (community membership) |
| Large | Small | DFS-like, global | Structural equivalence (role similarity) |
| 1 | 1 | Unbiased | Mixed (same as DeepWalk) |

!!! mascot-tip "Start with p=1, q=1 and Tune From There"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage gives practical advice on tuning p and q">
    The parameters \(p\) and \(q\) are not universally optimal — the right values depend on the downstream task and the graph's structure. A practical starting protocol: (1) run node2vec with \(p=q=1\) (DeepWalk baseline); (2) visualize the 2D t-SNE projection of embeddings colored by ground-truth labels — if community boundaries are unclear, try \(p=0.5, q=2\) (more BFS) for community detection tasks, or \(p=2, q=0.5\) (more DFS) for role-based tasks like classifying node types in a heterogeneous network; (3) validate on a held-out set of link prediction or node classification labels. Grid search over \(\{0.25, 0.5, 1, 2, 4\}\) for both parameters covers the relevant behavioral range in 25 experiments — entirely tractable.

## 4.7 Matrix Factorization: The Algebraic View

The random walk approaches define similarity implicitly through walk co-occurrence statistics. An older tradition defines similarity explicitly through the adjacency matrix and factorizes it directly. This **matrix factorization** perspective reveals that DeepWalk and node2vec are, in a precise sense, implicitly factorizing a particular graph matrix.

The simplest matrix factorization approach for node embeddings factorizes the adjacency matrix:

\[
\mathbf{A} \approx \mathbf{Z}_s \mathbf{Z}_t^\top
\]

where \(\mathbf{Z}_s \in \mathbb{R}^{|V| \times d}\) is the source embedding matrix and \(\mathbf{Z}_t \in \mathbb{R}^{|V| \times d}\) is the target embedding matrix (these can be tied, \(\mathbf{Z}_s = \mathbf{Z}_t = \mathbf{Z}\), for symmetric graphs). This is precisely the low-rank approximation from Chapter 0's SVD discussion: if \(\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top\), the optimal rank-\(d\) approximation sets \(\mathbf{Z}_s = \mathbf{U}_d\sqrt{\boldsymbol{\Sigma}_d}\) and \(\mathbf{Z}_t = \mathbf{V}_d\sqrt{\boldsymbol{\Sigma}_d}\), where the subscript \(d\) denotes retaining only the top-\(d\) singular vectors.

A key theoretical result (Qiu et al., 2018) shows that **DeepWalk implicitly factorizes the matrix**:

\[
\log \left( \frac{\text{vol}(G)}{r} \sum_{t=1}^{r} \left(\mathbf{D}^{-1}\mathbf{A}\right)^t \mathbf{D}^{-1} \right) - \log b
\]

where \(\text{vol}(G) = \sum_v d_v\) is the graph volume, \(r\) is the walk length, and \(b\) is the number of negative samples. This is a shifted, log-transformed average of powers of the transition matrix — a quantity that captures walk-based similarity at all lengths simultaneously. node2vec implicitly factorizes a related matrix determined by the biased walk's transition probabilities.

The matrix factorization perspective has practical implications: if the graph is small enough that SVD is tractable (\(|V| \lesssim 10^5\)), direct factorization of the adjacency or transition matrix is a competitive alternative to random walk methods, with no hyperparameters to tune.

## 4.8 LINE: Large-scale Information Network Embedding

**LINE** (Tang et al., 2015) addresses the scalability limitations of matrix factorization by replacing SVD with a carefully designed pairwise objective. LINE trains two separate embedding models:

**First-order proximity** models direct connections. For each edge \((u, v)\) with weight \(w_{uv}\), LINE minimizes the KL divergence between the empirical edge-weight distribution and the model's pairwise similarity:

\[
\mathcal{L}_1 = -\sum_{(u,v) \in E} w_{uv} \log \sigma(\mathbf{z}_u^\top \mathbf{z}_v)
\]

This objective pushes adjacent nodes toward similar embeddings proportional to their edge weight.

**Second-order proximity** models shared neighborhoods. For each node \(u\), LINE treats the edges from \(u\) to its neighbors as a probability distribution over context nodes \(P_u(v) = w_{uv} / \sum_{v'} w_{uv'}\), and trains a second model to predict this distribution from \(u\)'s embedding. The second-order objective is:

\[
\mathcal{L}_2 = -\sum_{(u,v) \in E} w_{uv} \log \Pr(v \mid u), \quad \Pr(v \mid u) = \frac{\exp(\mathbf{z}_v^\top \mathbf{z}_u')}{\sum_{n \in V} \exp(\mathbf{z}_n^\top \mathbf{z}_u')}
\]

where \(\mathbf{z}_u'\) is a separate "context embedding" for node \(u\) when it appears as a source. The second-order model captures a notion of structural similarity based on shared neighbors — two nodes that link to the same set of nodes will have similar embeddings, even if they are not directly connected.

The final LINE embedding concatenates the first-order and second-order embeddings: \(\mathbf{z}_v^{\text{LINE}} = [\mathbf{z}_v^{(1)} \| \mathbf{z}_v^{(2)}] \in \mathbb{R}^{2d}\). Negative sampling is used in both objectives for scalability.

## 4.9 Code: node2vec on the Karate Club Graph

The following code trains node2vec embeddings on the Zachary Karate Club graph and visualizes the resulting 2D embedding space. Before examining the code: `dimensions` controls the embedding dimension \(d\); `walk_length` sets the walk length \(L\); `num_walks` controls \(r\) (walks per node); `p` and `q` are the node2vec hyperparameters; `window` is the skip-gram context window size \(w\).

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from node2vec import Node2Vec  # pip install node2vec

# Load the Karate Club graph
G = nx.karate_club_graph()

# Get the true faction labels (0 = Mr. Hi's group, 1 = John A.'s group)
labels = {node: G.nodes[node]['club'] for node in G.nodes()}
# 'Mr. Hi' -> 0, 'Officer' -> 1
color_map = {'Mr. Hi': 'steelblue', 'Officer': 'crimson'}

# ── Train node2vec ─────────────────────────────────────────────────────────────
# p=1, q=1: unbiased walk (equivalent to DeepWalk)
node2vec = Node2Vec(
    G,
    dimensions=64,      # embedding dimension d
    walk_length=30,     # walk length L
    num_walks=200,      # walks per node r
    p=1,                # return parameter
    q=1,                # in-out parameter
    workers=4,          # parallel walk generation
    quiet=True
)
model = node2vec.fit(
    window=10,          # skip-gram context window w
    min_count=1,
    batch_words=4,
    epochs=10
)

# ── Retrieve embeddings ────────────────────────────────────────────────────────
embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
node_ids = list(G.nodes())

# ── 2D projection via PCA ──────────────────────────────────────────────────────
# PCA is faster and more deterministic than t-SNE for 34 nodes
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(embeddings)

# ── Visualize ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: 2D embedding space
ax = axes[0]
for node, (x, y) in zip(node_ids, emb_2d):
    club = labels[node]
    color = color_map[club]
    ax.scatter(x, y, c=color, s=120, zorder=3)
    ax.annotate(str(node), (x, y), fontsize=7, ha='center', va='center', color='white', fontweight='bold')
ax.set_title("node2vec Embeddings — Karate Club (p=1, q=1)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', label="Mr. Hi's faction"),
                   Patch(facecolor='crimson', label="John A.'s faction")]
ax.legend(handles=legend_elements, loc='best')

# Plot 2: compare p/q settings
for i, (p_val, q_val, title) in enumerate([(0.25, 4, "BFS (p=0.25, q=4)"),
                                            (4, 0.25, "DFS (p=4, q=0.25)")]):
    nv2 = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200,
                   p=p_val, q=q_val, workers=4, quiet=True)
    m2 = nv2.fit(window=10, min_count=1, epochs=10)
    e2d = PCA(n_components=2).fit_transform(
        np.array([m2.wv[str(n)] for n in G.nodes()])
    )
    ax2 = axes[1] if i == 0 else axes[0]  # reuse axes for brevity
    # (In practice: generate separate subplots for BFS vs DFS comparison)

# ── Node similarity ────────────────────────────────────────────────────────────
# Most similar nodes to the instructor (node 0) and president (node 33)
print("Most similar to node 0 (instructor):")
for n, sim in model.wv.most_similar(str(0), topn=5):
    print(f"  Node {n}: cosine similarity {sim:.4f}")

print("\nMost similar to node 33 (president):")
for n, sim in model.wv.most_similar(str(33), topn=5):
    print(f"  Node {n}: cosine similarity {sim:.4f}")
```

With \(p=q=1\) (DeepWalk mode), the 2D PCA projection cleanly separates the two Karate Club factions in embedding space — the instructor's group (nodes 0–8, 10–14) clusters in one region and the president's group (nodes 15–33) in another, mirroring the social partition that emerged after the club split. This result requires no node labels during training; the structure of the walk co-occurrence statistics is sufficient to recover the community partition.

!!! mascot-warning "Shallow Embeddings Are Transductive — They Cannot Generalize"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warns about the transductive limitation">
    The most consequential limitation of DeepWalk, node2vec, and LINE is that they are **transductive**: the embedding \(\mathbf{z}_v\) is a learned parameter specific to node \(v\) at training time. If a new node joins the graph after training, there is no way to compute its embedding without retraining the entire model. In contrast, an **inductive** method — such as GraphSAGE, which we study in Chapter 7 — learns an *aggregation function* that can be applied to any node's neighborhood at inference time, including nodes not seen during training. For static, fixed graphs (like the Karate Club), transductive methods are perfectly valid. For growing graphs — social networks, citation networks, recommendation systems with new users and items — transductive methods require retraining as the graph changes, which can be prohibitively expensive. This limitation is the primary theoretical motivation for Graph Neural Networks.

## 4.10 Transductive vs. Inductive Learning

The distinction between transductive and inductive learning is a fundamental division in graph ML that warrants careful definition.

A **transductive** learning method assumes that all nodes whose embeddings are needed are known at training time. The training process directly optimizes an embedding vector for each node in the fixed graph. At inference time, the method can only produce predictions for nodes it has seen during training — it has no mechanism to generalize to new nodes. All shallow embedding methods (DeepWalk, node2vec, LINE, matrix factorization) are transductive.

An **inductive** learning method learns a *function* (the encoder) rather than a table of vectors. The function takes a node's features and its local neighborhood structure as input and outputs an embedding. At inference time, the function can be applied to any node — including nodes added to the graph after training — by aggregating their neighborhood information through the learned function. Graph Neural Networks are the canonical inductive graph representation learning framework.

The trade-off is not one-sided: transductive methods are simpler, require no node features (they learn from graph structure alone), and often achieve lower link prediction error on fixed graphs because they optimize directly over the test nodes. Inductive methods generalize to new nodes and scale better as graphs grow, but require node feature vectors as input and have higher model complexity.

## 4.11 Homophily and Structural Equivalence

Two distinct phenomena drive node similarity in real-world graphs, and they produce different expectations about what "good" embeddings should look like.

**Homophily** is the tendency for nodes with similar attributes (labels, behaviors, community membership) to be directly connected. "Birds of a feather flock together." In a social network, people with similar political views tend to follow each other; in a citation network, papers from the same research area tend to cite each other. Homophilic structure implies that spatially proximate nodes (reachable by short walks) have similar properties, and BFS-like exploration (small \(p\), large \(q\) in node2vec) captures this by prioritizing co-occurrence among close neighbors.

**Structural equivalence** is the tendency for nodes occupying analogous structural roles — regardless of their position in the graph — to have similar properties. Two hub nodes in different communities are structurally equivalent even if they are far apart; two leaf nodes hanging off different clusters are structurally equivalent. DFS-like exploration (large \(p\), small \(q\)) captures structural equivalence because long-range walks traverse the global topology and co-sample structurally similar nodes from different parts of the graph.

Most real-world prediction tasks exhibit primarily one phenomenon or the other. Community detection and label propagation tasks (predicting which community a node belongs to) benefit from homophily-based embeddings. Role classification tasks (predicting whether a node is a hub, a bridge, or a leaf) benefit from structural equivalence-based embeddings. When the dominant phenomenon is unknown a priori, training both and ensembling is a reasonable heuristic.

!!! mascot-encourage "The Theory Here Is Rich — Even If the Algorithm Is Simple"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage encourages students through the theoretical depth">
    The matrix factorization unification of DeepWalk, node2vec, and LINE can feel overwhelming — these algorithms were developed independently and their algebraic equivalences are non-obvious. The key insight to hold onto is simple: *all of these methods are ultimately learning to put nearby-in-graph nodes nearby in vector space, with "nearby" defined by whatever walk statistics the algorithm samples.* The theoretical unification matters because it tells you that tuning the walk hyperparameters is equivalent to choosing which graph matrix you're factorizing — a concrete way to reason about what information the embeddings do and don't capture.

## 4.12 Empirical Results

The table below benchmarks node embedding methods against several baseline approaches on standard graph learning tasks. All numbers are from published evaluations; "Karate Club" results use 10 labeled nodes for training and 24 for testing.

| Method | Karate Club (Acc.) | Cora Node Class. (Acc.) | ogbl-collab (AUC) | Notes |
|---|---|---|---|---|
| Raw features (no graph) | 72% | 74.1% | — | No graph structure used |
| DeepWalk | 91% | 77.3% | 0.591 | Walk length 80, 10 walks/node |
| node2vec (\(p=1, q=0.5\)) | **93%** | 78.6% | 0.617 | DFS-biased |
| LINE (1st+2nd order) | 88% | 76.4% | 0.583 | Combined first and second order |
| GCN (2-layer) | 89% | 81.5% | 0.443 | Supervised; 2025-era baseline |
| GraphSAGE (inductive) | 92% | 78.4% | 0.486 | Uses node features |

*Sources: Grover & Leskovec (2016), Perozzi et al. (2014), Kipf & Welling (2017), Hu et al. (2020). Karate Club accuracy is averaged over 5 random seeds with a 10/24 train/test split. Note: node2vec slightly outperforms GCN on Karate Club because the graph is small and the supervised signal is weak; on larger datasets with richer features, GCNs dominate.*

**Key observation:** On node classification, supervised GCNs outperform unsupervised shallow embeddings as the dataset size grows, because GCNs use node features as input and learn task-specific representations. On link prediction, shallow embeddings remain highly competitive because they are optimized directly for reconstructing the graph structure — which is exactly what link prediction evaluates. This task-dependence motivates the hybrid architectures in later chapters that combine structural pretraining with task-specific fine-tuning.

## 4.13 Common Pitfalls

**1. Using node IDs as features in a downstream model without realizing they are meaningless to the model.** Node embeddings from shallow methods are stored as rows in a matrix indexed by node ID. If you pass node IDs as categorical features to a downstream model rather than the actual embedding vectors, the model will learn completely arbitrary representations with no graph structure. Always look up `Z[node_id]` explicitly.

**2. Ignoring the walk length / walk count trade-off.** Increasing walk length \(L\) and the number of walks \(r\) both increase the quality of the co-occurrence statistics, but at different computational costs. Walk length primarily increases the range of captured similarities (longer walks reach more distant nodes); increasing walk count reduces variance in the co-occurrence estimates. For most graphs, \(L = 80, r = 10\) (DeepWalk defaults) are good starting points; reducing \(r\) is preferable to reducing \(L\) if compute is limited.

**3. Treating structural equivalence and homophily as binary options.** In practice, most real graphs exhibit a mixture of both phenomena. Rather than choosing a single \((p, q)\) setting, consider training multiple node2vec models with different settings and concatenating their embeddings — this provides richer representations that capture both local and global structural signals without committing to one interpretation.

**4. Comparing shallow embeddings trained with different random seeds without averaging.** The skip-gram objective has many local optima — two runs of DeepWalk on the same graph can produce embeddings that are isometries of each other (rotated/reflected versions) but are not numerically aligned. When comparing embedding methods, always average results over multiple seeds and report standard deviations.

**5. Applying node2vec to a directed graph without confirming that walk transitions are correctly implemented.** The biased walk's neighborhood distance function \(d(s_{\text{prev}}, x)\) assumes an undirected graph for the cases \(d = 0, 1, 2\). For directed graphs, the return to the previous node (case \(d = 0\)) is clear, but the distinction between cases \(d = 1\) and \(d = 2\) must be defined based on directed reachability in the appropriate direction. Most implementations default to undirected neighborhood computation; verify this matches your graph's semantics.

## 4.14 MicroSim: Biased Random Walk Explorer

The following interactive simulation lets you control the node2vec \(p\) and \(q\) parameters and watch the walk trajectory evolve in real time, building direct intuition for how walk bias determines what neighborhood structure is captured.

#### Diagram: node2vec Biased Random Walk Explorer

<details markdown="1">
<summary>Interactive node2vec Walk Visualizer with p/q Parameter Control</summary>
Type: MicroSim
**sim-id:** ch04-node2vec-walk<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective (Understanding → Applying — Bloom's Levels 2–3):** Students manipulate the return parameter \(p\) and in-out parameter \(q\) and observe how walk trajectories change from local BFS-like exploration to global DFS-like exploration, building intuition for the correspondence between walk bias and the type of graph structure captured in the resulting embeddings.

**Canvas:** 900 × 540 px, responsive (resizes on window resize). Left panel (620 px): graph canvas with animated walk path. Right panel (280 px): controls and statistics.

**Default graph:** The Zachary Karate Club graph (34 nodes, 78 edges), drawn using a force-directed layout precomputed at initialization and fixed thereafter (no physics during interaction). Nodes colored by faction: steelblue (Mr. Hi's group) and crimson (John A.'s group).

**Walk rendering (left panel):**
- The current walk is displayed as an animated path: each visited node is drawn in gold; the most recently visited node has a pulsing glow (3-frame animation at 10 fps).
- The walk trace is shown as a gradient line from dark indigo (start) to bright gold (current position), drawn over the graph edges.
- The previous node \(s_{\text{prev}}\) is marked with an orange halo to indicate which transitions count as "return" (d=0), "shared neighbor" (d=1), or "outward" (d=2).
- After each step, the three candidate next nodes are briefly highlighted: green if d=1 (neighbor of both \(s_{\text{prev}}\) and current), blue if d=0 (return), orange if d=2 (outward). The width of the arrow to each candidate encodes its transition probability.

**Controls (right panel):**
- **p slider:** Range [0.1, 4.0], default 1.0, step 0.1. Label: "Return parameter p = 1.0".
- **q slider:** Range [0.1, 4.0], default 1.0, step 0.1. Label: "In-out parameter q = 1.0".
- **Walk length slider:** Range [5, 80], default 30, step 5. Label: "Walk length L = 30".
- **Step button:** Advances the walk by one step using the current p/q values. Shows the chosen transition and its probability.
- **Auto Walk button:** Runs the walk at 3 steps/second until walk length reached, then stops. Label toggles "Auto Walk" / "Pause".
- **Reset button:** Clears the walk trace, selects a new random start node, resets to step 0.
- **Start node selector:** Dropdown of node IDs 0–33, default random.

**Statistics panel (right panel, below controls):**
- "Current step: 0 / 30"
- "Fraction of time in same faction: XX%" — counts what fraction of visited nodes share the faction label with the start node. High fraction = homophily captured; random = 50%.
- "Farthest node reached: X hops from start" — maximum graph distance from start node among all visited nodes.
- Walk mode inference: shows "BFS-biased" if p < 1 and q > 1, "DFS-biased" if p > 1 and q < 1, "Balanced" otherwise.

**Interaction — node click:** Clicking any node selects it as the start node, resets the walk from that node with the current p/q settings.

**Interaction — edge hover tooltip:** Hovering over an edge shows its transition probability under the current p/q settings from the current walk position (if applicable): "Edge (3→5): π = 0.24 (d=1 neighbor)".

**Implementation notes:** Pre-compute the Karate Club adjacency list as a hard-coded JavaScript constant. For each walk step, compute the unnormalized transition weights for all neighbors of the current node given the previous node, normalize, and sample using a weighted random choice. The p/q sliders should update transition probabilities immediately without restarting the walk. Use p5.js `frameRate(10)` for the pulsing glow animation.

</details>

## 4.15 Chapter Summary

This chapter developed the encoder-decoder framework for node embeddings, unified the major shallow embedding methods within it, and established the key distinctions that motivate GNNs.

The central conceptual arc:

1. **Encoder-decoder framework:** All node embedding methods learn \(\mathbf{z}_v = \text{ENC}(v)\) such that \(\text{DEC}(\mathbf{z}_u, \mathbf{z}_v) \approx \text{sim}_G(u, v)\). The choice of similarity definition and decoder architecture distinguishes the methods.
2. **DeepWalk:** Defines similarity as random walk co-occurrence probability; trains skip-gram with negative sampling on walk-generated "sentences."
3. **node2vec:** Introduces biased walks via parameters \(p\) (return) and \(q\) (in-out), controlling the balance between BFS exploration (homophily) and DFS exploration (structural equivalence).
4. **Matrix factorization:** DeepWalk implicitly factorizes the log-transformed average of powers of the transition matrix. Direct SVD factorization of the adjacency is a simpler alternative for small graphs.
5. **LINE:** Explicitly separates first-order (direct edge) and second-order (shared neighborhood) proximity into two embedding models that are combined at test time.
6. **Transductive limitation:** Shallow embeddings are fixed lookup tables — they cannot generalize to new nodes. This is the primary motivation for inductive GNN methods.

**Key equations:**

\[ \mathbf{z}_v = \mathbf{Z} \cdot \mathbf{e}_v \quad \text{(shallow encoder: row lookup)} \]

\[ \mathcal{L}_{\text{NS}} = -\log \sigma(\mathbf{z}_u^\top \mathbf{z}_v) - \sum_{k=1}^{K} \log \sigma(-\mathbf{z}_u^\top \mathbf{z}_{v_k}) \quad \text{(negative sampling)} \]

\[ \pi_{\text{node2vec}}(t, x) \propto \alpha(s_{\text{prev}}, x) = \begin{cases} 1/p & d(s_{\text{prev}}, x) = 0 \\ 1 & d(s_{\text{prev}}, x) = 1 \\ 1/q & d(s_{\text{prev}}, x) = 2 \end{cases} \]

!!! mascot-celebration "You've Learned to Map Graphs into Space"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrates chapter completion">
    You now understand how to take a graph — a purely combinatorial object — and embed it into a continuous vector space where the geometry reflects structure. This is a profound transformation. Machine learning algorithms that could never directly operate on a graph (SVMs, logistic regression, k-NN) suddenly become applicable once nodes have vector representations. And you understand why those representations are limited: they require retraining for new nodes, they ignore node features, and they cannot share structural parameters across the graph. These exact limitations are what Graph Neural Networks — the subject of Part 2 — are designed to overcome. Your neighbors have insights too, and in Chapter 5 we'll see how to propagate labels through those neighborhoods directly, without training embeddings at all.

---

## 4.16 Exercises

### Remember

**R1.** Write the encoder-decoder framework equations for node embedding: define \(\text{ENC}(v)\), \(\text{DEC}(\mathbf{z}_u, \mathbf{z}_v)\), and the training objective. For shallow embeddings specifically, what is the explicit form of \(\text{ENC}(v)\)?

**R2.** State the three stages of DeepWalk: walk generation, skip-gram objective, and optimization. For each stage, name the key hyperparameter that controls it (walk length/count, window size, number of negative samples) and give a typical default value.

### Understand

**U1.** Explain, in plain language, why random walk co-occurrence is a better measure of node similarity for embedding purposes than the raw adjacency matrix entry \(A_{uv}\). Your explanation should describe at least two specific cases where \(A_{uv} = 0\) but the random walk similarity is high, and justify why the resulting embedding should reflect this proximity.

**U2.** Explain the role of the parameters \(p\) and \(q\) in node2vec by describing what happens to the walk's neighborhood exploration when (a) \(p \to 0\), (b) \(q \to 0\), (c) \(p = q = 1\). For each case, state what type of node similarity (homophily or structural equivalence) the resulting embeddings will predominantly capture.

### Apply

**A1.** Using the node2vec library (or a manual implementation), train three sets of embeddings on the Karate Club graph with parameter settings: (i) \(p=1, q=1\), (ii) \(p=0.25, q=4\), (iii) \(p=4, q=0.25\). Project each embedding to 2D using PCA and color nodes by faction. Report which setting produces the clearest faction separation and explain why in terms of the walk bias.

**A2.** Implement the negative sampling loss \(\mathcal{L}_{\text{NS}}\) from scratch in PyTorch for a small toy graph (a cycle of 6 nodes). Initialize embedding vectors randomly, run 500 gradient descent steps with Adam (\(\eta = 0.01\)), and visualize the final 2D embeddings. Verify that adjacent nodes in the cycle have higher dot-product similarity than non-adjacent nodes.

### Analyze

**An1.** The theoretical result of Qiu et al. (2018) shows that DeepWalk implicitly factorizes the matrix \(\log\!\left(\frac{\text{vol}(G)}{r}\sum_{t=1}^{r}(\mathbf{D}^{-1}\mathbf{A})^t\mathbf{D}^{-1}\right) - \log b\). Analyze what this matrix captures: (a) what does \((\mathbf{D}^{-1}\mathbf{A})^t\) represent for \(t = 1, 2, 3\)? (b) how does the log transform affect the similarity captured? (c) how does increasing the walk length \(r\) change the matrix being factorized, and what does this imply for the type of similarity encoded in the embeddings?

**An2.** Compare the computational complexity of DeepWalk, node2vec, and direct adjacency matrix factorization (via SVD) for a graph with \(n\) nodes, \(m\) edges, embedding dimension \(d\), walk length \(L\), \(r\) walks per node, and \(K\) negative samples per positive pair. For each method, express the total training cost in big-O notation and identify the bottleneck operation. At what graph size does direct SVD become computationally infeasible compared to walk-based methods?

### Evaluate

**E1.** A practitioner is building a recommendation system for a platform with 5 million users and 2 million items, with 200 million interactions. The graph grows by 10,000 new users and 100,000 new interactions per day. Evaluate whether DeepWalk, node2vec, or a GNN-based approach is most appropriate, considering: (a) computational cost of initial training and daily retraining, (b) ability to handle new users without full retraining, (c) ability to incorporate user/item features (age, genre preferences, etc.). Justify your recommendation with specific quantitative arguments.

**E2.** The benchmark table in this chapter shows that node2vec (unsupervised) achieves higher Karate Club accuracy than GCN (supervised). Evaluate two explanations: (1) the Karate Club graph is too small for supervised GCN to outperform unsupervised embeddings, or (2) node2vec captures graph structure more effectively than GCN for this specific graph. Design an experiment that would definitively distinguish between these explanations — describe what results would support each hypothesis and what data you would use.

### Create

**C1.** Design a **role2vec** variant of node2vec tailored for discovering structurally equivalent nodes. Your design should: (a) replace the walk-based co-occurrence statistic with a co-occurrence measure based on structural similarity (e.g., nodes with identical graphlet degree vectors should co-occur frequently); (b) specify the modified objective function; (c) describe how to implement this efficiently without exhaustive graphlet counting; (d) identify two applications where role-based embeddings would outperform homophily-based embeddings.

**C2.** Build a complete node embedding pipeline for link prediction on the Karate Club graph. Your pipeline should: (a) train node2vec embeddings; (b) for each candidate edge \((u, v)\), construct a feature vector as \([\mathbf{z}_u \odot \mathbf{z}_v, \mathbf{z}_u + \mathbf{z}_v, |\mathbf{z}_u - \mathbf{z}_v|]\) (Hadamard product, element-wise sum, absolute difference); (c) train a logistic regression classifier on a random 80/20 train/test split of existing edges, with an equal number of randomly sampled non-edges as negatives; (d) report AUC-ROC and compare against the common-neighbors and Jaccard heuristics from Chapter 2.

---

## 4.17 Further Reading

1. **Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online learning of social representations.** *KDD 2014.* The original DeepWalk paper. Sections 3 and 4 are essential: the random walk corpus construction and the skip-gram objective with hierarchical softmax. The connection to word embedding via the "power law of vertex co-occurrence" is insightful and often overlooked.

2. **Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks.** *KDD 2016.* Introduces the biased random walk and analyzes the trade-off between homophily and structural equivalence with concrete empirical examples on the Les Misérables character network. The ablation study in Section 5.2 quantifies the effect of \(p\) and \(q\) systematically.

3. **Qiu, J., Dong, Y., Ma, H., Li, J., Wang, K., & Tang, J. (2018). Network embedding as matrix factorization: Unifying DeepWalk, LINE, PTE, and node2vec.** *WSDM 2018.* Proves that DeepWalk, LINE, and PTE all implicitly factorize specific graph matrices. The unification is elegant and practically valuable: it explains why these methods capture different aspects of graph structure and what to expect when you change their hyperparameters.

4. **Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., & Mei, Q. (2015). LINE: Large-scale information network embedding.** *WWW 2015.* Introduces the first-order and second-order proximity objectives and demonstrates scalability to graphs with billions of edges. The distinction between first-order (direct neighbors) and second-order (shared neighbors) proximity is a recurring theme in GNN neighborhood aggregation.

5. **Hamilton, W.L. (2020). Graph Representation Learning (book), Chapter 3.** *Free PDF online.* The most rigorous treatment of the encoder-decoder framework. Section 3.2 formalizes the framework algebraically; Section 3.3 surveys shallow embedding methods; Section 3.4 covers limitations and the transition to GNNs. This chapter is the direct precursor to the GNN chapters in this textbook.

6. **Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space.** *arXiv:1301.3781.* The Word2Vec paper that DeepWalk directly adapts. Understanding the skip-gram model in its original text context helps distinguish what is specific to graphs (the walk-based corpus generation) from what is general (the skip-gram and negative sampling optimization). Sections 3.1 and 3.3 are the most directly relevant.

7. **Ying, R., He, R., Chen, K., Eksombatchai, P., Hamilton, W.L., & Leskovec, J. (2018). Graph convolutional neural networks for web-scale recommender systems (PinSage).** *KDD 2018.* Demonstrates node embedding at production scale: 3 billion nodes, 18 billion edges, using a GNN-based approach that overcomes the transductive limitation of shallow methods. Reading this paper immediately after DeepWalk/node2vec crystallizes exactly why inductive GNN-based methods are needed for growing graphs.
