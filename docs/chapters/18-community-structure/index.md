---
title: "Chapter 18: Community Structure in Networks"
description: "Modularity, Louvain, BigCLAM, and graph-based fraud detection — the algorithmic toolkit for discovering hidden structure in real-world networks."
generated_by: claude skill chapter-content-generator
date: 2026-05-29 23:15:00
version: 0.08
---

# Chapter 18: Community Structure in Networks

**Part 4: Graphs in the Wild**

## Summary

Studies the structural organization of real networks through community detection algorithms — Louvain, spectral clustering, BigCLAM — and their applications to social and fraud networks.

## Concepts Covered

This chapter covers the following 8 concepts from the learning graph:

1. Community Detection
2. Modularity (Network)
3. Louvain Algorithm
4. Girvan-Newman Algorithm
5. Overlapping Community
6. BigCLAM Model
7. Social Network Analysis
8. Fraud Detection (Graph)

## Prerequisites

This chapter builds on:

- [Chapter 1: Introduction to Graphs and Networks](../01-intro-to-graphs/index.md)
- [Chapter 2: Graph Properties and Traditional ML Features](../02-graph-properties-and-features/index.md)
- [Chapter 3: Link Analysis and PageRank](../03-link-analysis-pagerank/index.md)

---

!!! mascot-welcome "Welcome to Chapter 18"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waves hello at the chapter opening">
    Networks rarely evolve as featureless clouds of randomly wired nodes. Real systems — social platforms, biological interaction networks, citation graphs, financial transaction networks — tend to organize into **communities**: dense clusters of internally connected nodes that are sparsely linked to the rest of the graph. This chapter develops the algorithmic machinery for finding those clusters and explains why the problem is far subtler than it first appears. By the end, you will have moved from intuition to a precise quality function, from a principled-but-slow hierarchical algorithm to a fast greedy alternative that scales to millions of nodes, and from the assumption of disjoint partitions to models where every node can participate in multiple overlapping communities simultaneously.

## 18.1 The Hidden Architecture of Real Networks

In 1977, anthropologist Wayne Zachary published a study of a university karate club that had recently split into two factions following a dispute between its instructor and its administrator. The network Zachary collected — 34 members, 78 edges encoding observed social interactions — has since become one of the most studied graphs in network science, not because karate is particularly important but because its ground-truth community structure is unambiguously known: the club fractured along almost exactly the lines predicted by its internal density patterns. Every major community detection algorithm is benchmarked on it to this day.

The Karate Club graph is a pedagogical convenience, but the phenomenon it illustrates is universal. Protein-protein interaction networks cluster into functional modules whose members participate in the same biochemical pathway. Web pages cluster into topical communities where internal hyperlinks outnumber external ones by orders of magnitude. Academic citation networks cluster into research communities whose members preferentially cite each other. Financial transaction networks cluster into trading groups — and some of those groups are fraud rings.

The central hypothesis of community detection is that the edge distribution in a real network is not uniform at random; it is systematically biased toward within-group connections. Detecting communities means recovering that bias from the observed graph structure alone, without access to node labels or domain annotations. This is an unsupervised structural inference problem, and as we will see, it is considerably more difficult than it appears.

## 18.2 Social Network Analysis and the Sociology of Communities

Community detection is not a recent invention of machine learning. Its intellectual roots lie in sociometry — the quantitative study of social relationships — which dates to the 1930s work of Jacob Moreno, who developed the sociogram as a visualization of interpersonal ties. By the 1970s, sociologists were already asking whether a network's observable friendship structure reveals latent social groups, and the computational tools they developed for this purpose — hierarchical clustering, blockmodeling, and spectral methods — remain foundational.

The sociological intuition is captured in Mark Granovetter's seminal 1973 observation that strong ties (frequent, intimate contacts) tend to cluster locally, while weak ties (infrequent, casual contacts) serve as bridges between otherwise disconnected communities. This **strength of weak ties** hypothesis explains why communities have dense internal edges (strong ties accumulate locally) and sparse cross-community edges (bridges are structurally fragile). It also explains why removing high-betweenness edges — the bridges — is a principled way to divide a network into communities, which is precisely what the Girvan-Newman algorithm does.

Modern social network analysis (SNA) adds a richer vocabulary to this picture. A **clique** is a maximal complete subgraph — every member is directly connected to every other. A **k-core** is the maximal subgraph in which every node has at least k internal neighbors, a useful relaxation of the clique concept that handles the sparsity of real networks. A **structural hole** is a position in the network that bridges two communities, occupied by a node with high betweenness centrality. Communities themselves are sometimes defined modularly — as groups whose internal edge density substantially exceeds what a random graph with the same degree sequence would produce — and this modularity-based definition gives rise to the most widely used community quality metric.

## 18.3 Defining Community: What Are We Looking For?

Before we can detect communities algorithmically, we must commit to a precise definition of what a community is. This is harder than it sounds: the term is used in at least four distinct senses in the literature, and different definitions lead to different algorithms and different results on the same graph.

The four major definitional paradigms are as follows. First, a **partition-based** community is a disjoint subset of nodes such that internal edge density is high relative to external edge density — the intuition that nodes in a community know each other better than they know outsiders. Second, a **clique-based** definition requires near-completeness: every node connects to some minimum fraction of its community's other nodes. Third, a **spectral** definition treats communities as the leading eigenvectors of the graph Laplacian or related matrices, corresponding to low-frequency structural modes. Fourth, a **generative** definition specifies a probabilistic model for how edges arise given community memberships, then recovers communities by fitting the model to the observed graph.

Each paradigm has operational implications:

- Partition-based definitions admit efficient greedy algorithms (Louvain) but assume disjoint communities.
- Clique-based definitions find dense substructures but scale poorly.
- Spectral definitions have clean theoretical guarantees but require solving eigenproblems.
- Generative definitions accommodate overlapping memberships and model uncertainty, but optimization is non-convex.

Throughout this chapter we will work primarily with the partition-based and generative paradigms, because they produce the algorithms most widely deployed in practice.

!!! mascot-thinking "What Makes a Good Community?"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage looking thoughtful">
    Here's a subtlety that catches students off guard: you can always find communities in a graph if you're willing to call any subset a community. The hard part is finding communities that are **meaningfully denser internally than a random baseline would predict**. Modularity operationalizes this by comparing your partition's edge density to the expected density under a random null model. Without the null model, you have no way to distinguish a real community from a cluster that happens to be dense purely by chance.

## 18.4 Modularity: A Quality Function for Community Partitions

The most influential community quality measure in network science is **modularity**, introduced by Mark Newman and Michelle Girvan in 2004. Modularity \( Q \) quantifies how much better a given partition of a graph's nodes is, in terms of internal edge density, compared to what we would expect if edges were placed randomly while preserving each node's degree.

To derive modularity, we need a null model — a random graph that has the same degree sequence as our real graph but otherwise connects edges uniformly at random. For a network with \( m \) edges, the expected number of edges between nodes \( i \) and \( j \) in this null model (the **configuration model**) is:

\[
\mathbb{E}[A_{ij}] = \frac{k_i k_j}{2m}
\]

where \( k_i \) and \( k_j \) are the degrees of nodes \( i \) and \( j \) respectively. The configuration model preserves degree heterogeneity — high-degree nodes are expected to connect to many others — so if a partition puts two high-degree nodes in the same community, it deserves credit for that edge only if the edge exists more than the null model would predict.

Modularity for a partition \( \mathcal{C} = \{C_1, C_2, \ldots, C_k\} \) is:

\[
Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
\]

Here \( A_{ij} \) is the adjacency matrix (1 if edge exists, 0 otherwise), \( c_i \) is the community label of node \( i \), and \( \delta(c_i, c_j) = 1 \) if nodes \( i \) and \( j \) belong to the same community and 0 otherwise. The sum accumulates the excess edge density — actual edges minus expected edges — over all pairs of same-community nodes, normalized by \( 2m \).

Modularity ranges from approximately \( -0.5 \) (worst possible structure, anti-communities) to \( 1 \) (perfect community structure, never achieved in practice). Values above \( 0.3 \) are conventionally interpreted as evidence of meaningful community structure. The Karate Club graph, with its two ground-truth factions, achieves a modularity of approximately \( 0.37 \) under the correct partition.

The following table summarizes what different modularity ranges imply about network structure:

| Q Range | Interpretation |
|---|---|
| Q < 0 | Anti-community structure (edges avoid same-community pairs) |
| 0 ≤ Q < 0.3 | Weak or no meaningful community structure |
| 0.3 ≤ Q < 0.7 | Moderate to strong community structure |
| Q ≥ 0.7 | Very strong community structure (rare in real networks) |

Modularity maximization is NP-hard in general — there are exponentially many partitions to evaluate — so practical algorithms use greedy or approximation strategies to find high-modularity partitions without exhaustive search.

## 18.5 Divisive Hierarchical Clustering: The Girvan-Newman Algorithm

The Girvan-Newman algorithm, introduced in 2002 and predating the formalization of modularity, offers a conceptually elegant approach to community detection grounded in the sociology of bridges: communities are separated by edges that carry high traffic between them, so removing the most trafficked bridges exposes the community structure underneath.

The key concept underlying Girvan-Newman is **edge betweenness centrality**: for an edge \( (u, v) \), its betweenness is the number of shortest paths between all pairs of nodes in the graph that pass through that edge. Formally:

\[
B(u, v) = \sum_{s \neq t} \frac{\sigma(s, t \mid u, v)}{\sigma(s, t)}
\]

where \( \sigma(s, t) \) is the total number of shortest paths from \( s \) to \( t \), and \( \sigma(s, t \mid u, v) \) is the number of those paths that traverse edge \( (u, v) \). Bridge edges — those connecting two communities — lie on nearly every shortest path that crosses community boundaries, so they accumulate enormous betweenness scores.

The Girvan-Newman procedure is iterative and divisive:

1. Compute edge betweenness for every edge in the current graph.
2. Remove the edge with the highest betweenness.
3. Recompute edge betweenness for all remaining edges (the removal changes shortest paths).
4. Repeat until the graph decomposes into isolated nodes.

Each removal potentially splits a connected component into two, producing a hierarchical **dendrogram** — a tree of successive splits. To select the final partition, one typically cuts the dendrogram at the level that maximizes modularity.

Applied to the Karate Club graph, Girvan-Newman correctly identifies the two ground-truth factions: the edge with the highest betweenness is the one connecting the instructor's cluster to the administrator's cluster, and its removal produces the historically observed split.

The critical limitation of Girvan-Newman is computational: computing exact edge betweenness requires \( O(mn) \) time using Brandes' algorithm, and this computation must be repeated after every edge removal. For a graph with \( m \) edges, the full algorithm runs in \( O(m^2 n) \) time — feasible for networks with a few thousand nodes, but completely impractical for graphs with millions. Modern social and biological networks require a faster approach.

#### Diagram: Girvan-Newman Step-by-Step on the Karate Club Graph

<details markdown="1">
<summary>Girvan-Newman algorithm animated on the Karate Club graph</summary>
Type: MicroSim
**sim-id:** ch18-girvan-newman
**Library:** p5.js
**Status:** Specified

**Learning objective:** Understand (Bloom's Level 2) — the student observes how edge betweenness centrality identifies bridge edges and how their successive removal exposes community structure.

**Canvas:** 760 × 480 px, responsive to window resize.

**Layout:**
- Left panel (500 px wide): Karate Club graph with 34 nodes, force-directed layout. Nodes colored by current component membership (start all the same color, split into distinct colors as components separate). Edges colored by betweenness value: gradient from light gray (low betweenness) to vivid red (highest betweenness).
- Right panel (260 px): current iteration number, current modularity Q, list of last 5 removed edges.

**Controls (below the canvas):**
- "Step" button: remove one edge (highest betweenness), recompute betweenness, rerender.
- "Auto Play" button: continuously step every 800ms until graph is fully disconnected.
- "Reset" button: restore original graph.

**Interaction:**
- Hovering any edge shows a tooltip with its betweenness value and rank.
- Hovering any node shows its degree and community ID.

**Behavior:**
- On each step, highlight the to-be-removed edge in thick orange before deleting it (200ms pause).
- After removal, recompute betweenness and update edge colors immediately.
- Track and display modularity Q after each removal (using the formula shown in §18.4).
- When Q is maximized, display a gold banner: "Maximum modularity reached: Q = {value}".

**Visual style:** White background, indigo node fill, node labels showing node ID. Font: sans-serif, 12px.

**Data:** Hardcode the adjacency list of the Karate Club graph (34 nodes, 78 edges).
</details>

## 18.6 The Louvain Algorithm: Greedy Modularity Optimization at Scale

The Louvain algorithm, introduced by Blondel et al. (2008), is by far the most widely deployed community detection algorithm in practice. It directly maximizes modularity using a greedy two-phase strategy that scales to networks with hundreds of millions of nodes. Understanding why Louvain is both efficient and effective requires understanding the key insight that makes it possible: the modularity gain from moving a single node between communities can be computed in \( O(k_i) \) time, where \( k_i \) is the node's degree.

To derive this incremental update, suppose we want to move node \( i \) from its current community \( C_{\text{old}} \) to a neighboring community \( C_{\text{new}} \). The change in modularity \( \Delta Q \) is:

\[
\Delta Q = \left[ \frac{\Sigma_{\text{in}} + 2k_{i,\text{in}}}{2m} - \left(\frac{\Sigma_{\text{tot}} + k_i}{2m}\right)^2 \right] - \left[ \frac{\Sigma_{\text{in}}}{2m} - \left(\frac{\Sigma_{\text{tot}}}{2m}\right)^2 - \left(\frac{k_i}{2m}\right)^2 \right]
\]

where \( \Sigma_{\text{in}} \) is the sum of weights of edges internal to \( C_{\text{new}} \), \( k_{i,\text{in}} \) is the sum of weights of edges connecting \( i \) to nodes in \( C_{\text{new}} \), \( \Sigma_{\text{tot}} \) is the sum of degrees of all nodes in \( C_{\text{new}} \), and \( m \) is the total edge weight in the graph. This formula can be evaluated with only local information — the degrees and community assignments of node \( i \)'s immediate neighbors.

The Louvain algorithm alternates between two phases, repeating the two-phase cycle until no further modularity improvement is possible:

**Phase 1 — Local Optimization.** Initially assign every node to its own singleton community. For each node \( i \) in random order, compute \( \Delta Q \) for moving \( i \) into each of its neighbors' communities. Move \( i \) to the community that yields the largest positive \( \Delta Q \); if no move improves modularity, leave \( i \) in its current community. Repeat this pass over all nodes until no node changes community.

**Phase 2 — Graph Compression.** Construct a new "super-graph" where each discovered community becomes a single super-node. The weight of an edge between two super-nodes equals the total weight of all edges between their constituent nodes. Self-loops on each super-node encode the total internal edge weight of the original community.

After compression, the algorithm returns to Phase 1 on the super-graph. Each iteration of this two-phase cycle is called a **pass**. The algorithm terminates when a complete pass produces no improvement, typically in \( O(\log n) \) passes.

The time complexity of Louvain is nearly linear: \( O(m) \) per pass, with \( O(\log n) \) passes giving total complexity \( O(m \log n) \). This is fast enough to process graphs with hundreds of millions of edges in minutes on a single machine.

The following Python code demonstrates Louvain community detection on the Karate Club graph using the `python-louvain` package:

```python
import networkx as nx
import community as community_louvain  # pip install python-louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the Karate Club graph
G = nx.karate_club_graph()

# Run Louvain — returns dict {node_id: community_id}
# resolution parameter controls community granularity (default 1.0)
partition = community_louvain.best_partition(G, resolution=1.0)

# Compute modularity of the discovered partition
modularity = community_louvain.modularity(partition, G)
print(f"Number of communities: {len(set(partition.values()))}")
print(f"Modularity Q: {modularity:.4f}")

# Visualize: color nodes by community
pos = nx.spring_layout(G, seed=42)
cmap = cm.get_cmap('tab10', max(partition.values()) + 1)
node_colors = [cmap(partition[node]) for node in G.nodes()]

plt.figure(figsize=(10, 8))
nx.draw_networkx(G, pos, node_color=node_colors,
                 node_size=300, with_labels=True,
                 font_size=9, edge_color='gray', alpha=0.9)
plt.title(f"Louvain Communities (Q = {modularity:.3f})")
plt.axis('off')
plt.tight_layout()
plt.show()
```

The `resolution` parameter \( \gamma \) generalizes the null model: modularity becomes \( Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \gamma \frac{k_i k_j}{2m} \right] \delta(c_i, c_j) \). Increasing \( \gamma \) above 1.0 favors smaller, more refined communities; decreasing it below 1.0 encourages larger, coarser ones. This parameter is critical for applications because "the right scale" for communities is domain-dependent.

!!! mascot-tip "The Louvain Resolution Limit"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage offering a helpful tip">
    Louvain has a known **resolution limit**: it cannot reliably detect communities smaller than approximately \( \sqrt{m} \) edges. In a network with 10 million edges, communities with fewer than ~3,162 edges may be merged with neighbors even when they have perfect internal structure. This is not a bug in the implementation — it is a fundamental property of the modularity function itself, proven by Fortunato and Barthélemy (2007). If you need to detect small communities in a large network, use the `resolution` parameter to zoom in, or switch to algorithms that optimize local criteria such as conductance instead.

#### Diagram: Louvain Two-Phase Iteration Explorer

<details markdown="1">
<summary>Interactive visualization of Louvain's two-phase modularity optimization</summary>
Type: MicroSim
**sim-id:** ch18-louvain-explorer
**Library:** p5.js
**Status:** Specified

**Learning objective:** Analyzing (Bloom's Level 4) — the student traces how local node moves aggregate into global community structure across multiple passes, and observes the compression operation that enables scaling.

**Canvas:** 900 × 500 px, responsive to window resize.

**Three-panel layout:**
- Left panel (300 px): Original Karate Club graph. Nodes colored by current community assignment. Highlighted node (if any) shown in gold.
- Center panel (300 px): Current super-graph after the last Phase 2 compression. Nodes are community super-nodes sized proportionally to the number of original nodes they represent. Edge thickness proportional to cross-community edge weight.
- Right panel (300 px): Line chart tracking modularity Q across passes (x-axis: pass number, y-axis: Q value, range [0, 1]). A horizontal dashed line marks "Q = 0.3" as the community detection threshold.

**Controls:**
- "Run Phase 1" button: animates node reassignment — for each node (in order), shows the ΔQ arrow pointing to its new community (300ms per node), then snaps to final state.
- "Run Phase 2" button: animates the compression — communities collapse into super-nodes with a brief animation, then the super-graph appears in center panel.
- "Next Pass" button: triggers both Phase 1 and Phase 2 sequentially.
- Slider: "Resolution γ" from 0.5 to 2.0, step 0.1. Resets and reruns when changed.
- "Reset" button: restore initial state.

**Interaction:**
- Click any node in the left panel to see its ΔQ values for each neighboring community (shown as a tooltip bar chart).
- Click any super-node in the center panel to expand it and see its constituent original nodes highlighted in the left panel.

**Data:** Hardcode the Karate Club graph (34 nodes, 78 edges). Ground-truth faction labels available for comparison.
</details>

## 18.7 Overlapping Communities: Beyond Disjoint Partitions

Both Girvan-Newman and Louvain treat communities as a **partition** of the node set: every node belongs to exactly one community, and communities are mutually exclusive. This is a strong assumption that is violated in virtually every real social network. A person may simultaneously be a member of a family, a professional organization, a neighborhood association, and a recreational club — and their friendships reflect all four memberships simultaneously.

The **overlapping community** model relaxes the partition assumption by allowing each node to participate in multiple communities with varying degrees of membership. This richer representation requires a different mathematical formulation. Instead of a discrete community label \( c_i \in \{1, 2, \ldots, K\} \) for each node, the overlapping model assigns each node \( u \) a **membership strength vector** \( \mathbf{F}_u \in \mathbb{R}_{\geq 0}^K \), where \( F_{u,c} \) quantifies how strongly node \( u \) participates in community \( c \). A node with \( \mathbf{F}_u = [0.8, 0.0, 0.2] \) primarily belongs to community 1 and weakly to community 3.

The challenge for overlapping community detection is that we must simultaneously determine the number of communities \( K \), the membership vectors \( \{\mathbf{F}_u\}_{u \in V} \), and some way to relate membership vectors to the observed edges. The BigCLAM model provides a principled generative answer to this challenge.

!!! mascot-encourage "Overlapping Communities Are Harder — Here's Why That Matters"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage looking encouraging">
    If the overlapping community model seems more complex than the partition model — it is. The search space is continuous and high-dimensional, the optimization is non-convex, and selecting the number of communities \( K \) requires model selection. But the added complexity is worth confronting, because partition-based methods systematically misclassify nodes that sit at the intersection of multiple communities, labeling them as "noise" or misattributing them to a single group. In fraud detection, financial regulation, and epidemiology, those boundary nodes are often the most important ones to identify correctly.

## 18.8 BigCLAM: Community Affiliation Graph Model

BigCLAM (Yang and Leskovec, 2013) — short for **Cluster Affiliation Model for Big Networks** — formalizes overlapping communities through a generative probabilistic model. The model specifies how edges arise given community memberships, then recovers memberships by fitting the model to the observed graph via maximum likelihood estimation.

**The generative model.** Given \( K \) communities and membership strengths \( \mathbf{F}_u \in \mathbb{R}_{\geq 0}^K \) for each node \( u \), the probability that an edge exists between nodes \( u \) and \( v \) is:

\[
P(A_{uv} = 1 \mid \mathbf{F}) = 1 - \exp\!\left(-\mathbf{F}_u \cdot \mathbf{F}_v\right)
\]

where \( \mathbf{F}_u \cdot \mathbf{F}_v = \sum_{c=1}^K F_{u,c} \cdot F_{v,c} \) is the dot product of their membership vectors. The intuition is elegant: two nodes are likely to be connected if they share strong memberships in the same communities. If their dot product is large, the exponent \( \exp(-\mathbf{F}_u \cdot \mathbf{F}_v) \) is small, so \( P(A_{uv} = 1) \) approaches 1. If their dot product is zero (they share no community), the edge probability approaches \( 1 - \exp(0) = 0 \).

The model is non-negative: all entries of \( \mathbf{F}_u \) are constrained to \( F_{u,c} \geq 0 \). This makes BigCLAM a form of **non-negative matrix factorization** applied to graph adjacency matrices, with the edge-probability function providing the link between the latent factor matrix and the observed binary adjacency.

**Fitting the model.** Given the observed adjacency \( A \), we seek the membership matrix \( F \in \mathbb{R}_{\geq 0}^{n \times K} \) that maximizes the log-likelihood:

\[
\ell(F) = \sum_{(u,v) \in E} \log\!\left(1 - e^{-\mathbf{F}_u \cdot \mathbf{F}_v}\right) - \sum_{(u,v) \notin E} \mathbf{F}_u \cdot \mathbf{F}_v
\]

The first sum rewards high membership overlap for observed edges; the second penalizes overlap between non-adjacent node pairs. This log-likelihood is maximized using coordinate ascent: iterating over each node \( u \) and optimizing \( \mathbf{F}_u \) while holding all other membership vectors fixed. The gradient with respect to \( F_{u,c} \) is:

\[
\frac{\partial \ell}{\partial F_{u,c}} = \sum_{v \in \mathcal{N}(u)} \frac{F_{v,c} e^{-\mathbf{F}_u \cdot \mathbf{F}_v}}{1 - e^{-\mathbf{F}_u \cdot \mathbf{F}_v}} - \sum_{v \notin \mathcal{N}(u)} F_{v,c}
\]

The second term — a sum over all non-neighbors — is expensive to compute exactly for large graphs, but BigCLAM approximates it efficiently using degree-based summaries.

**Selecting K.** The number of communities \( K \) is a hyperparameter. In practice, one typically runs BigCLAM for a range of \( K \) values and selects the value that maximizes a held-out link prediction score or the model's likelihood adjusted for complexity (e.g., BIC or MDL criteria).

The following code demonstrates BigCLAM-style NMF community detection using the SNAP Python library on the Karate Club graph:

```python
import networkx as nx
import numpy as np
from sklearn.decomposition import NMF

# Load graph and construct adjacency matrix
G = nx.karate_club_graph()
n = G.number_of_nodes()
A = nx.to_numpy_array(G)  # shape: (n, n)

# Number of communities to find
K = 3

# NMF approximates A ≈ W @ H (both non-negative)
# W[u, c] ≈ F_u,c (membership strength of node u in community c)
# We set W == H (symmetric: nodes are both "source" and "target")
nmf = NMF(n_components=K, init='random', random_state=42,
          max_iter=500, solver='mu',  # multiplicative updates
          beta_loss='frobenius')

# F: membership matrix of shape (n, K)
# Note: true BigCLAM uses the specific log-likelihood above;
# this NMF approximates it for pedagogical clarity.
F = nmf.fit_transform(A)  # shape: (n, K)

# Assign each node to its dominant community
community_labels = np.argmax(F, axis=1)

# Identify overlapping nodes (those with significant membership in >1 community)
threshold = 0.3 * F.max(axis=1, keepdims=True)  # 30% of max membership
overlapping = [u for u in range(n) if np.sum(F[u] > threshold[u]) > 1]
print(f"Overlapping nodes (members of ≥2 communities): {overlapping}")

# Display membership matrix
import pandas as pd
membership_df = pd.DataFrame(F, columns=[f"C{c+1}" for c in range(K)])
membership_df.index.name = "Node"
print(membership_df.round(3).head(10))
```

The key parameters are: `n_components` (= K, the number of communities), `init` (initialization strategy — `'random'` is standard; `'nndsvd'` is more deterministic), `max_iter` (maximum coordinate-descent iterations), and `beta_loss` (`'frobenius'` minimizes squared reconstruction error; `'kullback-leibler'` is appropriate for count data).

The following table compares disjoint (Louvain) and overlapping (BigCLAM) community detection on common benchmark tasks:

| Property | Louvain (Disjoint) | BigCLAM (Overlapping) |
|---|---|---|
| Membership model | Hard partition | Soft membership vector |
| Optimization | Greedy ΔQ | Coordinate ascent on log-likelihood |
| Scalability | Excellent (near-linear) | Good (linear in edges) |
| Handles bridge nodes | No — assigns to one community | Yes — continuous membership |
| Requires specifying K | No | Yes |
| Ground-truth comparison | NMI, ARI | F1 over community sets |
| Typical use case | Social clustering, graph partitioning | Author-topic networks, protein functions |

## 18.9 Community Structure in Fraud Detection

The same structural properties that make communities detectable — high internal edge density, sparse cross-community connections, coordinated behavioral patterns — also make communities a powerful signal for identifying fraud. Financial fraud, social media manipulation, insurance fraud, and identity theft all tend to produce characteristic community signatures that differ from legitimate behavior.

**Fraud rings in financial networks.** Consider a transaction network where nodes are bank accounts and directed edges encode money transfers. A legitimate customer's transaction graph is typically sparse: a few edges to employers (payroll), utility companies (bills), and retailers (purchases). A money laundering ring, by contrast, generates a dense clique-like structure — the ring repeatedly cycles funds through its members to obscure the money's origin. This cyclic, high-density internal structure makes the fraud ring highly detectable as a community with an anomalously high modularity contribution.

**Account compromise clusters.** In social networks, compromised accounts that are used for coordinated spam campaigns share behavioral patterns: they post at similar times, interact with the same target accounts, and follow each other for mutual amplification. This coordination manifests as a tightly connected community in the follower graph or the interaction graph. Community detection identifies these clusters even when individual accounts appear legitimate in isolation.

**The adversarial challenge.** Sophisticated fraud actors are aware that community detection is a common tool, and they deliberately introduce noise to evade it. A fraud ring might add a small number of edges to legitimate-looking nodes to reduce its apparent internal density, or distribute its members across multiple detected communities. This creates an arms race between community detection algorithms and adversarial graph manipulation. Robust fraud detection therefore combines community structure with node-level features, temporal patterns, and anomaly scores.

The following pipeline demonstrates graph-based fraud detection using community structure as a feature:

```python
import networkx as nx
import community as community_louvain
import numpy as np
import pandas as pd

def build_transaction_graph(transactions_df):
    """
    Build a directed weighted graph from transaction records.
    Each row of transactions_df: {sender, receiver, amount, timestamp}
    Edge weight = total transaction volume between account pairs.
    """
    G = nx.DiGraph()
    for _, row in transactions_df.iterrows():
        u, v = row['sender'], row['receiver']
        w = row['amount']
        if G.has_edge(u, v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G

def extract_community_features(G):
    """
    Extract per-node community-based features for fraud classification.
    These features capture structural anomalies at the community level.
    """
    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(G_undirected)

    # Group nodes by community
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    features = []
    for node in G.nodes():
        comm_id = partition[node]
        comm_members = communities[comm_id]
        comm_size = len(comm_members)

        # Internal degree: edges to same community
        internal_degree = sum(1 for nbr in G_undirected.neighbors(node)
                              if partition.get(nbr) == comm_id)
        total_degree = G_undirected.degree(node)

        # Internal edge fraction (embeddedness)
        embeddedness = internal_degree / max(total_degree, 1)

        # Community internal density
        subgraph = G_undirected.subgraph(comm_members)
        n_internal_edges = subgraph.number_of_edges()
        max_edges = comm_size * (comm_size - 1) / 2
        internal_density = n_internal_edges / max(max_edges, 1)

        features.append({
            'node': node,
            'community_id': comm_id,
            'community_size': comm_size,
            'embeddedness': embeddedness,       # high → likely belongs to ring
            'internal_density': internal_density,  # high → tight-knit cluster
            'degree': total_degree,
        })

    return pd.DataFrame(features)
```

The `embeddedness` feature — the fraction of a node's edges that stay within its community — is a particularly strong fraud signal. Legitimate accounts have moderate embeddedness, reflecting a mix of local and external connections. Fraud ring members have embeddedness approaching 1.0, because nearly all their edges go to ring confederates.

!!! mascot-warning "Community Labels Are Not Ground Truth"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning about a common mistake">
    A common mistake in applied fraud detection is treating detected community labels as reliable ground truth for flagging accounts as fraudulent. They are not. Community detection is unsupervised — it finds the densest internal structure, regardless of whether that structure corresponds to fraud. Legitimate communities — trading desks, family networks, corporate suppliers — can look structurally identical to fraud rings. Always combine community features with domain-specific signals and validate with labeled data before using community membership as an input to any enforcement action.

## 18.10 Common Pitfalls in Community Detection

**The resolution limit.** As noted in §18.6, modularity optimization suffers from a fundamental resolution limit: communities smaller than \( \sqrt{m} \) edges may be merged into larger groups even when perfectly well-defined. This is not a numerical artifact — it is an inherent property of the global modularity function, which must balance local density improvements against the global null model. Multi-resolution methods (varying \( \gamma \) in the generalized modularity) or locally-defined criteria (conductance, internal edge density) can partially circumvent this limitation.

**Degeneracy of the modularity landscape.** Real networks typically have not one but an exponentially large number of partitions with modularity values within a small \( \epsilon \) of the maximum. This means that Louvain (and other modularity-maximizing methods) is not converging to a unique answer — it is sampling from a degenerate landscape. Different random initializations, different orderings of the greedy node passes, or different tie-breaking rules can produce qualitatively different partitions with nearly identical Q values. If you run Louvain five times on the same graph with different seeds and get five very different community assignments, the graph's community structure may be genuinely ambiguous, not a bug.

**Choosing K in BigCLAM.** There is no universally agreed-upon criterion for selecting the number of communities \( K \) in overlapping models. BIC, MDL, cross-validated link prediction, and domain knowledge all give different answers. Treat K as a hyperparameter to sweep over, and report sensitivity to K in any published analysis.

**Comparing community assignments.** When evaluating a detected partition against ground truth, the choice of metric matters enormously. Normalized Mutual Information (NMI) rewards overlap in cluster assignments but is sensitive to the number of communities. Adjusted Rand Index (ARI) accounts for chance agreement but assumes disjoint partitions. F1 over community sets is appropriate for overlapping communities but requires defining what counts as a "match." These metrics are not interchangeable, and papers that cherry-pick metrics to show favorable results are common.

## 18.11 Key Takeaways

Community detection transforms the question "which nodes are connected?" into the deeper question "what groups do those connections reveal?" We covered eight interconnected concepts in this chapter:

- **Social network analysis** provides the sociological vocabulary — strong and weak ties, structural holes, centrality — that motivates the computational problem.
- **Community detection** is the task of finding groups of nodes with dense internal and sparse external connections, operationalized through a chosen quality criterion.
- **Modularity** quantifies community quality by comparing observed internal edge density to the configuration model null; values above 0.3 signal meaningful structure.
- **Girvan-Newman** is a principled divisive algorithm based on iterative removal of high-betweenness bridge edges, producing a dendrogram whose optimal cut maximizes Q; it is exact but scales as \( O(m^2 n) \).
- **Louvain** greedily maximizes modularity using local \( \Delta Q \) updates and graph compression, scaling nearly linearly to massive networks; its resolution limit prevents detection of very small communities.
- **Overlapping communities** abandon the partition assumption, allowing each node to belong to multiple communities with continuous membership strengths.
- **BigCLAM** operationalizes overlapping communities through a generative model — \( P(\text{edge}) = 1 - \exp(-\mathbf{F}_u \cdot \mathbf{F}_v) \) — optimized via coordinate ascent on the log-likelihood.
- **Fraud detection** exploits the fact that fraud rings produce anomalously dense, high-embeddedness communities; community features (embeddedness, internal density, community size) are powerful inputs to fraud classifiers.

## 18.12 Further Reading

The literature on community detection is vast. The following five works provide a solid foundation spanning theory, algorithms, and applications.

**Foundational algorithms:**

- Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). **Fast unfolding of communities in large networks.** *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008. [arXiv:0803.0476](https://arxiv.org/abs/0803.0476). The original Louvain paper — concise, well-written, and directly implementable.

- Yang, J., & Leskovec, J. (2013). **Overlapping community detection at scale: A non-negative matrix factorization approach (BigCLAM).** *WSDM '13*. [ACM DL](https://dl.acm.org/doi/10.1145/2433396.2433471). The BigCLAM paper with the full derivation of the coordinate-ascent update and experiments on large-scale social networks.

**Surveys:**

- Fortunato, S. (2010). **Community detection in graphs.** *Physics Reports*, 486(3–5), 75–174. [doi:10.1016/j.physrep.2009.11.002](https://doi.org/10.1016/j.physrep.2009.11.002). Comprehensive 100-page survey covering spectral methods, modularity, NMF, and planted partition models; the standard reference.

**Theoretical limits:**

- Fortunato, S., & Barthélemy, M. (2007). **Resolution limit in community detection.** *PNAS*, 104(1), 36–41. Proves that modularity maximization cannot detect communities smaller than \( \sqrt{m} \) edges, regardless of how dense those communities are.

**Applications:**

- Akoglu, L., Tong, H., & Koutra, D. (2015). **Graph-based anomaly detection and description: A survey.** *Data Mining and Knowledge Discovery*, 29(3), 626–688. Covers community-structure-based fraud detection, anomaly scoring, and adversarial graph manipulation in depth.

## 18.13 Exercises

The following twelve exercises span all six Bloom's taxonomy levels.

**Remember (Level 1)**

1. State the modularity formula \( Q \) and identify the role of each term. What does the configuration model null \( \frac{k_i k_j}{2m} \) represent, and why is it subtracted from \( A_{ij} \)?

2. List the two phases of the Louvain algorithm in order. What does each phase accomplish, and what stops the algorithm from running indefinitely?

**Understand (Level 2)**

3. Explain why Girvan-Newman removes edges with high betweenness centrality rather than edges with low weight. What property of bridge edges makes betweenness centrality a reliable criterion?

4. In BigCLAM, why is the edge probability modeled as \( P(\text{edge}) = 1 - \exp(-\mathbf{F}_u \cdot \mathbf{F}_v) \) rather than simply \( \mathbf{F}_u \cdot \mathbf{F}_v \)? What mathematical property does the exponential form guarantee that the linear form does not?

**Apply (Level 3)**

5. Run Louvain with three different random seeds on the Karate Club graph. Report the modularity Q and the number of communities for each run. Are the partitions identical? Discuss what disagreements among runs imply about the network's community structure.

6. Implement the \( \Delta Q \) formula for moving a single node from one community to another (§18.6). Verify your implementation by checking that moving node 0 in the Karate Club graph to its correct ground-truth community increases Q compared to a random singleton initialization.

**Analyze (Level 4)**

7. Apply Louvain to the Karate Club graph with resolution parameters \( \gamma \in \{0.5, 1.0, 2.0\} \). Plot the number of communities found versus \( \gamma \). At which value of \( \gamma \) does the algorithm best recover the two ground-truth factions? Explain the direction of the effect.

8. Compute the embeddedness of every node in the Karate Club graph under the two-community ground-truth partition. Which five nodes have the lowest embeddedness (i.e., the highest fraction of cross-community edges)? What structural role do these nodes likely play?

**Evaluate (Level 5)**

9. Girvan-Newman and Louvain both attempt to find high-modularity partitions, but via very different mechanisms. For a social network with 10,000 nodes and 50,000 edges, which algorithm would you use in a production fraud detection pipeline, and why? Quantify the computational cost of each approach for this scale.

10. A colleague argues that a detected community with internal density 0.95 and modularity contribution 0.08 is strong evidence of a fraud ring. Identify at least three reasons why this conclusion might be premature, and describe what additional evidence you would collect before flagging the community.

**Create (Level 6)**

11. Design a community detection evaluation pipeline for a network where ground-truth community labels are partially available for 20% of nodes. Specify: (a) how you would use the labeled subset to calibrate algorithm hyperparameters, (b) which metrics you would compute on the labeled and unlabeled subsets separately, and (c) how you would handle the case where detected communities span both labeled and unlabeled nodes.

12. Propose a modification to BigCLAM that incorporates node features \( \mathbf{x}_u \in \mathbb{R}^d \) into the membership vector computation. Specifically, define a parameterized function \( \mathbf{F}_u = f_\theta(\mathbf{x}_u) \) and write out the modified edge probability and log-likelihood. Discuss: does this modification turn BigCLAM into a supervised model, an unsupervised model, or something in between?

!!! mascot-celebration "Chapter 18 Complete!"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating the end of the chapter">
    You have now traversed the full arc of community detection — from the sociological intuition of strong and weak ties, through the modularity quality function and the algorithms that optimize it, to overlapping memberships and adversarial applications in fraud detection. The Karate Club graph's two factions, which Zachary observed by hand in 1977, turn out to be discoverable by every algorithm in this chapter: Girvan-Newman finds the bridge edge that separated them, Louvain assigns each node to its correct faction, and BigCLAM places the boundary-spanning members in both communities simultaneously. Real networks are rarely this clean — but understanding why they sometimes are, and why they often are not, is the foundation for everything that comes next.
