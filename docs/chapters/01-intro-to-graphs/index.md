---
title: "Chapter 1: Introduction to Graphs and Networks"
description: "Introduces the language of graphs: nodes, edges, adjacency matrices, paths, components, and real-world network properties including power-law degree distributions and small-world structure."
generated_by: claude skill chapter-content-generator
date: 2026-05-28 16:55:11
version: 0.08
---

# Chapter 1: Introduction to Graphs and Networks

**Part 0: Prerequisites**

## Summary

Introduces the language of graphs: nodes, edges, adjacency matrices, paths, components, and real-world network properties including power-law degree distributions and small-world structure.

## Concepts Covered

This chapter covers the following 33 concepts from the learning graph:

1. Graph (Undirected)
2. Graph (Directed)
3. Node (Vertex)
4. Edge (Link)
5. Adjacency Matrix
6. Node Degree
7. In-Degree
8. Out-Degree
9. Degree Distribution
10. Power-Law Degree Distribution
11. Graph Path
12. Shortest Path
13. Graph Diameter
14. Connected Component
15. Strongly Connected Component
16. Weakly Connected Component
17. Bipartite Graph
18. Heterogeneous Graph
19. Multigraph
20. Weighted Graph
21. Attribute Graph
22. Subgraph
23. Ego Network
24. Clique
25. Cycle
26. Tree (Graph Theory)
27. Spanning Tree
28. Planar Graph
29. Graph Isomorphism
30. Graph Homomorphism
31. Breadth-First Search
32. Depth-First Search
33. NetworkX

## Prerequisites

This chapter builds on:

- [Chapter 0: Math and Programming Prerequisites](../00-math-prerequisites/index.md)

---

## Motivating Example: Three Problems, One Language

Consider three problems from entirely different fields, each seemingly requiring its own toolbox.

**Drug discovery.** A pharmaceutical company wants to predict whether a new molecule will inhibit a protein linked to Alzheimer's disease. The molecule is a collection of atoms held together by chemical bonds. Predicting its properties requires reasoning about *how atoms are arranged and connected*, not just what types of atoms are present.

**Social recommendation.** A streaming service wants to recommend music to a new user. The service knows which songs existing users have listened to and which users tend to share similar tastes. Predicting what the new user will enjoy requires reasoning about *the network of user-song interactions*, not just the new user's stated preferences.

**Fraud detection.** A bank wants to flag suspicious transactions in real time. Fraudsters rarely act alone: compromised accounts transfer money through a chain of intermediate accounts before cashing out. Detecting this pattern requires reasoning about *the flow structure of the transaction network*, not just each transaction in isolation.

These three problems — in chemistry, recommendation, and finance — share a hidden common structure. Each can be represented as a **graph**: a collection of entities connected by relationships. Atoms and bonds; users and songs; accounts and transactions. When we represent these systems as graphs, we gain access to a unified set of analytical and machine learning tools that would be impossible or impractical to apply to raw tabular or pixel data.

This chapter is your introduction to that language.

!!! mascot-welcome "Hi — I'm Sage. Let me introduce myself."
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving hello and introducing themselves">
    Welcome to **Graph Neural Networks**! I'm **Sage** — a glowing graph node who learns by aggregating insights from my neighborhood, exactly like the GNNs you're about to build. My name is a deliberate nod to GraphSAGE, one of the architectures you'll implement from scratch in a few chapters. I'll appear throughout this textbook, but I only show up when I have a specific job to do. I have exactly **six jobs**, and you'll learn to recognize me by which one I'm doing:

    1. **Welcome you** at the opening of every chapter — that's what I'm doing right now.
    2. **Highlight key concepts** when an idea is the kind that reshapes how you read everything that follows — I show up with my thinking face.
    3. **Give you practical tips** — the moves a working graph ML practitioner knows but that rarely appear in papers or slides.
    4. **Warn you** about the common mistakes that trip up even experienced researchers.
    5. **Encourage you** when the math or code gets genuinely difficult and you need to know it gets easier.
    6. **Celebrate with you** at the end of each chapter, when you have earned it.

    That's it. If I'm not doing one of those six things, I'm not in the chapter. Let's aggregate some knowledge!

---

## 1.1 The Mathematical Language of Graphs

### 1.1.1 Defining a Graph

A **graph** is a pair $G = (V, E)$ where:

- $V$ is a set of **nodes** (also called *vertices*), with $|V| = n$
- $E \subseteq V \times V$ is a set of **edges** (also called *links* or *arcs*), with $|E| = m$

Each edge $e = (u, v) \in E$ connects two nodes $u, v \in V$.

This deceptively simple definition unifies the three examples from the introduction. For a molecule, $V$ is the set of atoms and $E$ is the set of covalent bonds. For a social network, $V$ is the set of users and $(u, v) \in E$ if user $u$ and user $v$ are friends. For a transaction network, $V$ is the set of bank accounts and $(u, v) \in E$ if account $u$ sent money to account $v$.

### 1.1.2 Undirected Graphs

In an **undirected graph**, edges have no direction: if $(u, v) \in E$ then $(v, u) \in E$ as well. Friendship networks are undirected — if Alice is Bob's friend, Bob is Alice's friend. The edges can be thought of as *unordered pairs* $\{u, v\}$.

Undirected graphs are the simpler case. Most of the core theory we will develop — graph Laplacians, spectral clustering, message passing — was originally derived for undirected graphs.

### 1.1.3 Directed Graphs

In a **directed graph** (or *digraph*), each edge $(u, v)$ has a *source* $u$ and a *target* $v$, and $(u, v) \in E$ does not imply $(v, u) \in E$. Web hyperlinks are directed — a page can link to another without receiving a link back. Citation networks are directed — paper A cites paper B, but not necessarily vice versa.

The distinction between directed and undirected graphs affects almost every property we will compute. Degree, connectivity, and path-finding all behave differently depending on whether edges have direction.

### 1.1.4 Notation Conventions

Throughout this textbook we use the following standard conventions:

- $n = |V|$ — number of nodes
- $m = |E|$ — number of edges
- $\mathbf{A} \in \{0,1\}^{n \times n}$ — adjacency matrix (defined formally in Section 1.2)
- $\mathbf{X} \in \mathbb{R}^{n \times d}$ — node feature matrix ($d$-dimensional feature per node)
- $\mathcal{N}(v)$ — the *neighborhood* of node $v$: the set of nodes adjacent to $v$
- $d_v$ or $\deg(v)$ — the degree of node $v$ (number of neighbors)

---

## 1.2 Graph Representations

### 1.2.1 The Adjacency Matrix

The most fundamental way to represent a graph computationally is the **adjacency matrix** $\mathbf{A} \in \{0,1\}^{n \times n}$:

$$A_{ij} = \begin{cases} 1 & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

For an undirected graph, $\mathbf{A}$ is *symmetric*: $A_{ij} = A_{ji}$ for all $i, j$. For a directed graph, $\mathbf{A}$ is generally *asymmetric*.

As a concrete example, consider a 4-node undirected path graph (nodes 1–2–3–4):

$$\mathbf{A} = \begin{pmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

Row $i$ of $\mathbf{A}$ is a binary indicator vector: entry $j$ is 1 if and only if there is an edge from node $i$ to node $j$. This means multiplying $\mathbf{A}$ by a feature vector $\mathbf{x}$ — as we saw in Chapter 0 — computes the sum of neighbor features for every node simultaneously: $(\mathbf{A}\mathbf{x})_i = \sum_{j \in \mathcal{N}(i)} x_j$.

### 1.2.2 Sparse Representations

For large real-world graphs, storing $\mathbf{A}$ as a dense $n \times n$ matrix is impractical. The ogbn-arxiv graph has 169,343 nodes; its adjacency matrix would require $169{,}343^2 \approx 28.7$ billion entries. Since real networks are sparse (average degree typically 2–100), the vast majority of those entries are zero.

The standard sparse representation is the **edge list** (or equivalently, the COO — Coordinate — sparse format): simply store the list of $(i, j)$ pairs where $A_{ij} = 1$. For an undirected graph with $m$ edges, the edge list has $2m$ entries. PyTorch Geometric stores graphs in this format as `edge_index`, a $2 \times m$ tensor of source and target node indices.

### 1.2.3 Node Feature Matrix

Beyond pure structure, most real-world graphs carry **attributes** on nodes and edges. A node feature matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ stores a $d$-dimensional feature vector in row $i$ for node $i$. In a citation network, node features might be bag-of-words representations of paper abstracts. In a molecular graph, atom features might include atomic number, charge, and hybridization.

The goal of a GNN is to compute a new representation $\mathbf{H} \in \mathbb{R}^{n \times d'}$ by combining the graph structure $\mathbf{A}$ with the node features $\mathbf{X}$ — exactly the operation $\mathbf{A}\mathbf{X}$ (and its normalized variants) that the Chapter 0 MicroSim visualized.

!!! mascot-thinking "The Adjacency Matrix Is the Backbone of Every GNN"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage explaining how the adjacency matrix connects everything">
    Everything a GNN does reduces, at its core, to multiplying some normalized version of **A** by some matrix of features. GCN uses $\mathbf{D}^{-1/2}\hat{\mathbf{A}}\mathbf{D}^{-1/2}\mathbf{X}\mathbf{W}$. GraphSAGE uses $\mathbf{D}^{-1}\mathbf{A}$. GAT replaces the uniform normalization with learned attention weights. Every chapter in Part 2 of this textbook is a variation on this theme. When you truly understand the adjacency matrix — what it represents, how sparse it is, and what multiplying by it *does* — you understand the foundation of all of graph machine learning. Connect the dots and the pattern emerges.

---

## 1.3 Degree and Degree Distributions

### 1.3.1 Node Degree

The **degree** of a node $v$ in an undirected graph is the number of edges incident to it:

$$d_v = \deg(v) = |\mathcal{N}(v)| = \sum_{j=1}^{n} A_{vj}$$

Equivalently, the degree of node $v$ is the sum of row $v$ in the adjacency matrix. For a directed graph, we distinguish:

- **In-degree** $d_v^{\text{in}} = \sum_{j} A_{jv}$: the number of edges *entering* $v$ (column sum)
- **Out-degree** $d_v^{\text{out}} = \sum_{j} A_{vj}$: the number of edges *leaving* $v$ (row sum)

For a directed graph, a node can have a high in-degree (many things point to it, like a highly-cited paper) and a low out-degree (it cites few others), or vice versa. PageRank, which we study in Chapter 3, is essentially a sophisticated measure of in-degree adjusted for the quality of the incoming edges.

The **degree matrix** $\mathbf{D}$ is a diagonal matrix with $D_{vv} = d_v$. It appears in virtually every GNN normalization: $\mathbf{D}^{-1}\mathbf{A}$ is the row-stochastic transition matrix, and $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ is the symmetric normalized adjacency used in GCN.

### 1.3.2 Degree Distribution and Scale-Free Networks

The **degree distribution** $P(k)$ is the probability that a randomly chosen node has degree exactly $k$:

$$P(k) = \frac{\text{number of nodes with degree } k}{n}$$

In a classical random graph (Erdős–Rényi model $G(n,p)$), edges are added independently with probability $p$, producing a binomial — and for large $n$, approximately Poisson — degree distribution concentrated around the mean degree $\bar{d} = (n-1)p$.

Real-world networks look nothing like this. In social networks, citation networks, and the web, a small number of nodes have enormously high degree (hubs) while the vast majority have low degree. This pattern follows a **power-law degree distribution**:

$$P(k) \propto k^{-\gamma}$$

where $\gamma$ is typically between 2 and 3 for real networks. Networks with this property are called **scale-free**: there is no characteristic scale for the degree, and the tail of the distribution is "heavy" — hubs are far more common than a Poisson distribution would predict.

Why do power-law networks arise? The **preferential attachment** mechanism (Barabási-Albert model) provides one explanation: new nodes are more likely to connect to already-popular nodes ("the rich get richer"). Twitter followers, web inlinks, and paper citations all follow this dynamic. The existence of hubs has profound consequences for GNN design — aggregating over a hub's 10,000 neighbors is computationally very different from aggregating over a typical node's 5 neighbors, which motivates the sampling strategies we cover in Chapter 20.

---

## 1.4 Paths, Distances, and Connectivity

### 1.4.1 Walks, Paths, and Cycles

A **walk** in a graph is any sequence of nodes $v_0, v_1, \ldots, v_k$ such that $(v_{i}, v_{i+1}) \in E$ for all $i$. The **length** of a walk is the number of edges traversed ($k$ here).

A **path** is a walk that visits no node more than once. Paths are the standard notion of "route" between two nodes.

A **cycle** is a closed walk (starts and ends at the same node) of length $\geq 3$ that visits no intermediate node more than once. The presence or absence of cycles is a fundamental graph property: trees are precisely the connected graphs with no cycles. Cycle detection underlies many graph algorithms.

### 1.4.2 Shortest Paths and Graph Diameter

The **shortest path** (or *geodesic*) between two nodes $u$ and $v$ is the path between them with minimum length. The length of the shortest path is called the **graph distance** $d(u, v)$.

The **diameter** of a graph is the maximum distance between any pair of nodes:

$$\text{diam}(G) = \max_{u, v \in V} d(u, v)$$

For disconnected graphs, the diameter is technically infinite; in practice we report the diameter of the largest connected component.

### 1.4.3 Breadth-First Search

**Breadth-First Search (BFS)** computes shortest paths from a source node to all other nodes in an unweighted graph. Starting from source $s$, BFS explores all neighbors at distance 1, then all nodes at distance 2, and so on, using a queue.

BFS is the algorithmic basis for computing graph distances, finding connected components, and — crucially — defining the *receptive field* of a GNN. A $k$-layer GNN aggregates information from the $k$-hop BFS neighborhood of each node.

### 1.4.4 Depth-First Search

**Depth-First Search (DFS)** explores as far as possible along each branch before backtracking, using a stack. DFS is well-suited for detecting cycles, computing topological orderings of directed acyclic graphs, and finding strongly connected components (via Tarjan's or Kosaraju's algorithm).

### 1.4.5 Connected Components

A graph is **connected** if there is a path between every pair of nodes. A **connected component** is a maximal connected subgraph — the largest set of nodes that are all reachable from each other.

Real-world networks often have one **giant component** containing most nodes, and many small isolated components. The existence of the giant component is a phase transition: in the Erdős–Rényi model $G(n, c/n)$, a giant component emerges when $c > 1$.

For directed graphs, connectivity is more nuanced:

- A **strongly connected component (SCC)** is a maximal set of nodes where every node is reachable from every other node *following the direction of edges*.
- A **weakly connected component (WCC)** is a maximal set of nodes that are connected when edge directions are ignored.

The web has a famous "bow-tie" structure: a large SCC at the center, a set of nodes that can reach the SCC but not be reached from it, a set that can be reached from the SCC but cannot reach it, and tendrils connecting these groups.

---

## 1.5 Special Graph Types

### 1.5.1 Bipartite Graphs

A **bipartite graph** has its nodes partitioned into two disjoint sets $U$ and $V$ such that every edge connects a node in $U$ to a node in $V$ — no edges exist within $U$ or within $V$. User-item interaction graphs (users in $U$, items in $V$, edges representing interactions) are the canonical bipartite graph in machine learning. The recommender systems in Chapter 16 (LightGCN, NGCF) operate almost exclusively on bipartite graphs.

A graph is bipartite if and only if it contains no odd-length cycles.

### 1.5.2 Heterogeneous Graphs

A **heterogeneous graph** (or *heterogeneous information network*) generalizes the standard graph by allowing multiple types of nodes and multiple types of edges. Formally, each node $v$ is assigned a type $\tau(v)$ from a type vocabulary $\mathcal{T}_V$, and each edge $e$ is assigned a type $\phi(e)$ from $\mathcal{T}_E$.

A knowledge graph — with entity nodes of types like Person, Place, Organization and relation edges of types like *born_in*, *works_for*, *located_in* — is the archetypal heterogeneous graph. Heterogeneous graphs require architectures (R-GCN, HGT) that distinguish between type-specific transformations, which we cover in Chapter 15.

### 1.5.3 Multigraphs

A **multigraph** allows multiple edges between the same pair of nodes (called *parallel edges*) and may also allow **self-loops** $(v, v) \in E$. Molecular graphs sometimes include multigraphs when different bond types (single, double, triple) exist between the same atom pair. Self-loops, while seemingly trivial, play an important practical role in GNNs: adding $\mathbf{I}$ to $\mathbf{A}$ (i.e., $\hat{\mathbf{A}} = \mathbf{A} + \mathbf{I}$) ensures each node includes its own features in aggregation, which substantially improves performance.

### 1.5.4 Weighted Graphs

A **weighted graph** assigns a scalar weight $w(u, v) \in \mathbb{R}$ to each edge $(u, v)$. Road networks use distance or travel time as weights; financial networks use transaction amounts; co-authorship networks use the number of joint papers. The weighted adjacency matrix $\mathbf{W}$ replaces the binary $A_{ij} \in \{0,1\}$ with $W_{ij} = w(i, j)$.

### 1.5.5 Attribute Graphs

An **attribute graph** (or *feature graph*) attaches feature vectors to nodes and/or edges. Most GNN benchmarks use attribute graphs — each node in the ogbn-arxiv citation network carries a 128-dimensional feature vector derived from the paper's title and abstract.

---

## 1.6 Graph Substructures

### 1.6.1 Subgraphs

A **subgraph** $G' = (V', E')$ of $G = (V, E)$ satisfies $V' \subseteq V$ and $E' \subseteq E \cap (V' \times V')$. Subgraphs arise naturally in graph ML: we often want to classify or embed not the entire graph but specific regions of it. Frequent subgraph mining (Chapter 19) searches for subgraph patterns that appear unusually often across a collection of graphs.

### 1.6.2 Ego Networks

The **ego network** of a node $v$ (also called the *1-hop neighborhood* or *local neighborhood*) is the subgraph induced by $v$ and all its immediate neighbors $\mathcal{N}(v)$. This is exactly the computation unit of a single GNN layer: layer $l$ of a GNN computes a new representation for $v$ using the ego network of $v$ in the graph of layer-$(l-1)$ representations.

The $k$-hop ego network of $v$ — the subgraph induced by all nodes within distance $k$ of $v$ — is the *receptive field* of a $k$-layer GNN.

### 1.6.3 Cliques

A **clique** is a subset of nodes $C \subseteq V$ such that every pair of nodes in $C$ is connected by an edge — a completely connected subgraph. A clique of size $k$ is called a $k$-clique. Friend groups in social networks often approximate cliques; molecules often contain ring structures that are cliques.

Detecting large cliques is NP-hard in general (the maximum clique problem), but approximate methods and GNN-based approaches are active research areas.

### 1.6.4 Cycles, Trees, and Spanning Trees

A **cycle** is a closed path $v_1, v_2, \ldots, v_k, v_1$ of length $k \geq 3$. Cycles are the fundamental "non-tree-like" feature of a graph.

A **tree** is a connected acyclic graph — a connected graph with $n-1$ edges and no cycles. Trees are the simplest connected graphs and admit linear-time algorithms for many problems that are NP-hard on general graphs. Decision trees, parse trees, and molecular scaffold trees all exploit this tractability.

A **spanning tree** of a graph $G$ is a tree that includes every node of $G$ and uses only edges from $G$. Every connected graph has at least one spanning tree. Minimum spanning trees (which minimize total edge weight) arise in network design and clustering.

### 1.6.5 Planar Graphs

A **planar graph** is one that can be drawn in the plane with no crossing edges. Many physical networks — road networks, circuit layouts — are approximately planar. Non-planar graphs include $K_5$ (the complete graph on 5 nodes) and $K_{3,3}$ (the complete bipartite graph on $3+3$ nodes); Kuratowski's theorem states that a graph is planar if and only if it contains neither as a minor.

---

## 1.7 Graph Isomorphism and Homomorphism

!!! mascot-encourage "This One Is Hard — Even for Computers"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage encouraging students through a difficult concept">
    Graph isomorphism is one of the oldest open problems in computational complexity — no polynomial-time algorithm is known, yet no proof of NP-hardness exists either. That puts it in a strange limbo. But here is why it matters for GNNs directly: in Chapter 9, we will prove that most GNNs *cannot* distinguish certain pairs of non-isomorphic graphs, and in Chapter 10 we will see how to fix this. Understanding what isomorphism means now will make those chapters click. Your neighbors have insights too — and so do the graphs they can't tell apart.

### 1.7.1 Graph Isomorphism

Two graphs $G = (V, E)$ and $G' = (V', E')$ are **isomorphic** (written $G \cong G'$) if there exists a bijection $f: V \to V'$ such that:

$$(u, v) \in E \iff (f(u), f(v)) \in E'$$

Intuitively, two graphs are isomorphic if one can be relabeled to become identical to the other — they have the same structure, just with nodes renamed. Two graphs that look different can be isomorphic (the same structure drawn differently), and two graphs that look similar can be non-isomorphic.

**Why this matters for GNNs:** A fundamental requirement of any graph representation learning method is that it should map isomorphic graphs to the same representation (isomorphism invariance) and, ideally, non-isomorphic graphs to different representations (injectivity). The Weisfeiler-Lehman (WL) graph isomorphism test, which we will study in depth in Chapter 9, characterizes exactly which pairs of graphs can and cannot be distinguished by message-passing GNNs.

### 1.7.2 Graph Homomorphism

A **graph homomorphism** from $G = (V, E)$ to $H = (V', E')$ is a function $f: V \to V'$ such that:

$$(u, v) \in E \implies (f(u), f(v)) \in E'$$

Unlike an isomorphism, a homomorphism need not be injective or surjective — multiple nodes of $G$ can map to the same node of $H$. The existence of a homomorphism from $G$ to $H$ means that the structure of $G$ can be "mapped into" $H$.

Graph homomorphism counts appear in the theory of GNN expressiveness (Chapter 9) and in the analysis of graph kernels (Chapter 2). The number of homomorphisms from a small graph (a "pattern") to a large graph is a rich descriptor of the large graph's structure.

---

## 1.8 Real-World Network Properties

Two empirical regularities appear so consistently across real-world networks that they are sometimes called "laws" of network science.

### 1.8.1 The Small-World Property

Real networks have, on average, surprisingly short paths between nodes despite being large. The **average shortest path length** $\bar{\ell}$ grows only *logarithmically* with the number of nodes $n$:

$$\bar{\ell} \approx \frac{\log n}{\log \bar{d}}$$

where $\bar{d}$ is the average degree. For the global Facebook network with ~3 billion users, the average distance between any two users is approximately 3.5 hops — the famous "six degrees of separation" is actually an overestimate.

Watts and Strogatz (1998) showed that high clustering (many triangles in the network) and low average path length can coexist — this is the **small-world model**. Real networks achieve both simultaneously through a combination of local clustering and a few long-range "shortcut" edges.

### 1.8.2 Scale-Free Structure and Hubs

As discussed in Section 1.3, real networks follow power-law degree distributions. The consequence is the existence of **hubs** — nodes with degree orders of magnitude above the average. In the web graph, Google.com has billions of inlinks while a typical page has fewer than 10. Hubs make networks robust to random failures (removing a random node rarely removes a hub) but fragile to targeted attacks (removing hubs rapidly disconnects the network).

### 1.8.3 The Clustering Coefficient

The **clustering coefficient** of a node $v$ measures what fraction of $v$'s neighbors are connected to each other:

$$C_v = \frac{\text{number of edges among } \mathcal{N}(v)}{\binom{d_v}{2}} = \frac{2 \cdot |\{(u,w) \in E : u,w \in \mathcal{N}(v)\}|}{d_v(d_v - 1)}$$

High clustering means that the "friend of my friend is also my friend" principle holds locally. Real networks have clustering coefficients 10–100× higher than random graphs of the same size and average degree.

---

## 1.9 Graph ML Tasks

Graph machine learning encompasses three levels of prediction, differing in what is predicted and what signals are used.

**Node-level tasks** predict a property of individual nodes. Node classification assigns a label (e.g., a topic category) to each node in a network. In the Karate Club graph, the task is predicting which faction each member joined after the club split — a 2-class node classification problem on 34 nodes.

**Edge-level tasks** predict properties of pairs of nodes. Link prediction asks: given the current graph structure, which pairs of currently unconnected nodes are likely to form an edge in the future? Recommender systems reduce to link prediction on bipartite user-item graphs. Edge classification assigns a type (e.g., *friend*, *colleague*, *family*) to an existing edge.

**Graph-level tasks** treat the entire graph as a single object. Graph classification assigns a global label to a graph — predicting whether a molecule is toxic, for example. Graph regression predicts a continuous value (e.g., the binding affinity of a drug molecule). Graph generation creates new graphs with desired properties (Chapter 21).

The three levels are deeply connected: a good node-level representation can be pooled to produce a graph-level representation, and link prediction can be reduced to classifying pairs of node embeddings.

---

## 1.10 Analyzing Graphs with NetworkX

!!! mascot-tip "NetworkX Is Your Graph Laboratory"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage giving a helpful tip about NetworkX">
    NetworkX is the standard Python library for graph analysis — it is not a deep learning framework (that is PyTorch Geometric's job), but it is the best tool for understanding graph structure, running graph algorithms, and exploring datasets before training. Every property in this chapter can be computed with a single NetworkX call. Use it to sanity-check your graphs before feeding them into GNNs: verifying degree distributions, checking for unexpected disconnected components, and confirming the graph is undirected when you expect it to be will save you hours of debugging.

Before we examine the code, a brief orientation: the `networkx` library represents a graph as a Python object. Nodes can be any hashable object (integers, strings, tuples). Edges are stored internally as an adjacency dictionary. The key functions we will use are `G.degree()`, `nx.shortest_path_length()`, `nx.connected_components()`, and `nx.average_clustering()`. The first time you call a function, read the docstring — NetworkX functions return various types (lists, dictionaries, generators) depending on what is most natural for the computation.

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ── Load the Karate Club graph ────────────────────────────────────────────────
# Zachary's Karate Club: 34 nodes (members), 78 edges (social ties)
# The standard benchmark for small-scale graph learning experiments.
G = nx.karate_club_graph()

print(f"Nodes  : {G.number_of_nodes()}")     # 34
print(f"Edges  : {G.number_of_edges()}")     # 78
print(f"Directed: {nx.is_directed(G)}")      # False (undirected)

# ── Degree statistics ─────────────────────────────────────────────────────────
degrees = dict(G.degree())                   # {node: degree}
degree_sequence = sorted(degrees.values(), reverse=True)

print(f"\nDegree stats:")
print(f"  Max degree  : {max(degree_sequence)}")     # 17 (node 0, the instructor)
print(f"  Min degree  : {min(degree_sequence)}")     # 1
print(f"  Mean degree : {np.mean(degree_sequence):.2f}")   # 4.59
print(f"  Median degree: {np.median(degree_sequence):.1f}")  # 3.0

# ── Adjacency matrix ─────────────────────────────────────────────────────────
A = nx.to_numpy_array(G)                     # dense numpy matrix, shape (34, 34)
print(f"\nAdjacency matrix shape: {A.shape}")
print(f"Symmetric: {np.allclose(A, A.T)}")   # True (undirected)
print(f"Non-zero entries: {int(A.sum())}")   # 156 = 2 * 78 edges

# ── Connectivity ──────────────────────────────────────────────────────────────
print(f"\nConnected: {nx.is_connected(G)}")  # True
components = list(nx.connected_components(G))
print(f"Number of components: {len(components)}")  # 1

# ── Path lengths ──────────────────────────────────────────────────────────────
avg_path = nx.average_shortest_path_length(G)
diameter = nx.diameter(G)
print(f"\nAverage shortest path length: {avg_path:.3f}")  # 2.408
print(f"Diameter: {diameter}")                              # 5

# ── Clustering ────────────────────────────────────────────────────────────────
avg_clustering = nx.average_clustering(G)
print(f"\nAverage clustering coefficient: {avg_clustering:.3f}")  # 0.571

# ── Degree distribution ───────────────────────────────────────────────────────
degree_counts = Counter(degrees.values())
k_values = sorted(degree_counts.keys())
pk_values = [degree_counts[k] / G.number_of_nodes() for k in k_values]

# The distribution shows a heavy tail: a few hubs, many low-degree nodes.
print("\nDegree distribution P(k):")
for k, pk in zip(k_values, pk_values):
    print(f"  k={k:2d}: {'█' * int(pk * 200):.<30} {pk:.3f}")

# ── BFS and shortest paths ────────────────────────────────────────────────────
# Compute all shortest paths from node 0 (the instructor)
distances_from_0 = nx.single_source_shortest_path_length(G, source=0)
print(f"\nDistances from node 0: {dict(sorted(distances_from_0.items()))}")

# ── DFS ───────────────────────────────────────────────────────────────────────
dfs_order = list(nx.dfs_preorder_nodes(G, source=0))
print(f"\nDFS order from node 0 (first 10): {dfs_order[:10]}")

# ── Subgraph and ego network ──────────────────────────────────────────────────
ego_0 = nx.ego_graph(G, 0)                   # ego network of node 0
print(f"\nEgo network of node 0:")
print(f"  Nodes: {ego_0.number_of_nodes()}")  # 17 (node 0 + 16 neighbors)
print(f"  Edges: {ego_0.number_of_edges()}")  # 54

# ── Bipartite graph example ───────────────────────────────────────────────────
# Construct a simple user-item bipartite graph
B = nx.Graph()
users = ["Alice", "Bob", "Carol"]
items = ["Item1", "Item2", "Item3"]
B.add_nodes_from(users, bipartite=0)         # set user node group
B.add_nodes_from(items, bipartite=1)         # set item node group
B.add_edges_from([("Alice","Item1"), ("Alice","Item2"),
                  ("Bob","Item1"), ("Bob","Item3"),
                  ("Carol","Item2"), ("Carol","Item3")])

from networkx.algorithms import bipartite
print(f"\nIs bipartite: {bipartite.is_bipartite(B)}")  # True

# ── Clique detection ──────────────────────────────────────────────────────────
cliques = list(nx.find_cliques(G))
max_clique = max(cliques, key=len)
print(f"\nLargest clique in Karate Club: {max_clique} (size {len(max_clique)})")

# ── Spanning tree ─────────────────────────────────────────────────────────────
T = nx.minimum_spanning_tree(G)
print(f"\nMinimum spanning tree: {T.number_of_nodes()} nodes, {T.number_of_edges()} edges")
# A spanning tree of n nodes always has exactly n-1 edges: 33 here.
```

The output reveals classic small-world properties: an average path length of just 2.4 hops in a 34-node network, and an average clustering coefficient of 0.571 — nearly 58% of each node's neighbors know each other. This combination of short paths and high clustering is the hallmark of real-world social networks.

---

## 1.11 MicroSim: Graph Property Explorer

The interactive simulation below lets you build your own graph and watch properties update in real time. Add nodes and edges by clicking; remove them by right-clicking. The panel on the right updates the degree distribution, average path length, clustering coefficient, and connected components as you modify the graph.

#### Diagram: Graph Property Explorer


<iframe src="../../sims/ch01-graph-explorer/main.html" width="100%" height="572px" scrolling="no"></iframe>
[Run Graph Property Explorer Fullscreen](../../sims/ch01-graph-explorer/main.html)

---

## 1.12 Benchmark: Properties of Key Graph Datasets

The table below summarizes the properties of graph datasets that appear repeatedly throughout this textbook, from the small Karate Club benchmark to large-scale OGB datasets used for competitive evaluation.

| Dataset | Type | Nodes | Edges | Avg. Degree | Avg. Path | Clustering | Task |
|---------|------|------:|------:|------------:|----------:|-----------:|------|
| Karate Club | Social | 34 | 78 | 4.59 | 2.41 | 0.571 | Node class. (2) |
| Cora | Citation | 2,708 | 5,429 | 4.01 | 6.31 | 0.241 | Node class. (7) |
| CiteSeer | Citation | 3,327 | 4,732 | 2.74 | — | 0.141 | Node class. (6) |
| ogbn-arxiv | Citation | 169,343 | 1,166,243 | 13.78 | — | — | Node class. (40) |
| ogbn-products | Co-purchase | 2,449,029 | 61,859,140 | 50.5 | — | — | Node class. (47) |
| ogbg-molhiv | Molecular | ~26/graph | ~27/graph | ~2.2 | — | — | Graph class. (binary) |
| ogbl-collab | Collaboration | 235,868 | 1,285,465 | 10.9 | — | — | Link prediction |

*Sources: Kipf & Welling (2017) for Cora/CiteSeer statistics; Hu et al. (2020) for OGB statistics. "—" indicates not commonly reported for this dataset. Avg. path not computed for large sparse graphs due to computational cost.*

**Key observations:** (1) Citation networks have very low clustering compared to social networks. (2) The jump from Cora (2,708 nodes) to ogbn-arxiv (169,343 nodes) motivates the scalable training methods in Chapter 20 — a GNN that runs fine on Cora may need days on ogbn-arxiv without sampling. (3) Average degree varies enormously: from 2.74 (CiteSeer) to 50.5 (ogbn-products), which affects every normalization choice.

---

## 1.13 Common Pitfalls

!!! mascot-warning "Six Mistakes That Will Silently Break Your Graph Code"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning about common graph coding mistakes">
    These errors do not always raise exceptions — they produce wrong results quietly, which makes them far more dangerous. Commit the list below to memory; each one has cost practitioners real debugging time.

**1. Treating a directed graph as undirected (or vice versa)**

When loading a graph from an edge list, many loaders default to creating a directed graph. If you compute degree as `G.in_degree() + G.out_degree()` instead of the simple undirected degree, the numbers will be doubled for undirected edges represented as two directed arcs. Always confirm directionality with `nx.is_directed(G)` after loading.

**2. 0-indexed vs. 1-indexed node labels**

NetworkX uses 0-indexed nodes by default for most generators, but many real-world datasets use 1-indexed IDs. Loading a 1-indexed graph into a 0-indexed system shifts every node ID by 1, creating an extra phantom node 0 and leaving one node disconnected. Always check `min(G.nodes())` after loading and re-index if necessary with `nx.convert_node_labels_to_integers(G)`.

**3. Forgetting to add reverse edges for undirected graphs**

When constructing a graph from an edge list by calling `G.add_edge(u, v)` in a `DiGraph`, only the directed edge $(u \to v)$ is added. For undirected semantics you must add both $(u, v)$ and $(v, u)$, or use a `Graph` instead of `DiGraph`. The adjacency matrix will be asymmetric without this — and your GNN will only aggregate from one direction.

**4. Confusing the adjacency matrix with the transition matrix**

$\mathbf{A}$ and $\mathbf{D}^{-1}\mathbf{A}$ look similar but mean very different things. $\mathbf{A}$ sums neighbor features; $\mathbf{D}^{-1}\mathbf{A}$ averages them. Plugging the wrong matrix into a GNN equation produces the wrong aggregation — and since both can give reasonable-looking loss curves, the error may not surface until you compare benchmark numbers.

**5. Ignoring isolated nodes**

An isolated node has degree 0 and is its own connected component. Several NetworkX functions — `average_shortest_path_length`, `diameter` — raise exceptions or return incorrect values for disconnected graphs. Always check connectivity and handle isolated nodes before running path-length or diameter computations.

**6. Equating graph isomorphism with equality**

Two non-isomorphic graphs can have the same number of nodes, edges, and even the same degree sequence. If you use only degree sequences to check "are these the same graph?", you will get false positives. Proper isomorphism checking requires `nx.is_isomorphic(G, H)` — which runs in exponential time in the worst case but is fast in practice for typical graph sizes.

---

## 1.14 Chapter Summary

| Concept | Definition | GNN Relevance |
|---------|-----------|--------------|
| Graph $G=(V,E)$ | $n$ nodes, $m$ edges | The input object for all graph ML |
| Adjacency matrix $\mathbf{A}$ | $A_{ij}=1$ iff $(i,j)\in E$ | Core operation: $\mathbf{A}\mathbf{X}$ = neighborhood aggregation |
| Degree $d_v$ | $\|\mathcal{N}(v)\|$ | Normalization: $\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ |
| Power-law $P(k)\propto k^{-\gamma}$ | Heavy-tailed degree distribution | Motivates sampling (Ch 20), hub handling |
| Shortest path $d(u,v)$ | Min-length path between $u$ and $v$ | Defines GNN receptive field via BFS |
| Connected component | Maximal connected subgraph | Affects train/val/test split validity |
| Ego network $\mathcal{N}(v)$ | 1-hop neighborhood of $v$ | The computation unit of one GNN layer |
| Graph isomorphism $G\cong G'$ | Structure-preserving bijection | Bounds GNN expressiveness (Ch 9) |
| Small-world property | Low $\bar{\ell}$, high $C$ | Enables information diffusion in few hops |
| Bipartite graph | Two disjoint node sets | Recommender systems (Ch 16) |

---

## 1.15 Exercises

*12 exercises across all six levels of Bloom's Taxonomy.*

### Remember

**1.1.** Write the formal definition of a graph $G$. State what $n$, $m$, $V$, $E$, and $\mathcal{N}(v)$ denote, and give one real-world example of each of the following graph types: undirected, directed, weighted, bipartite, heterogeneous.

**1.2.** List the three levels of graph ML tasks (node, edge, graph). For each level, name one concrete prediction task from a different application domain (social networks, molecular biology, recommender systems).

### Understand

**1.3.** The adjacency matrix of an undirected graph is symmetric. Explain *why* this is the case and describe what a non-symmetric adjacency matrix tells you about the underlying graph. Why does symmetry matter for eigendecomposition (recall Chapter 0)?

**1.4.** Explain the small-world property in plain language: what two properties must a network have simultaneously, and why is it surprising that real networks exhibit both? Use the Karate Club graph's statistics (average path 2.41, clustering 0.571) to illustrate your answer.

### Apply

**1.5.** Using NetworkX, load the Karate Club graph and compute: (a) the degree of every node, (b) the largest clique, (c) the ego network of the node with the highest degree, and (d) the three nodes with the highest betweenness centrality. Report your results and interpret what the highest-betweenness node represents socially.

**1.6.** Construct the $6 \times 6$ adjacency matrix for the following graph by hand, then verify your result with `nx.to_numpy_array`: a cycle $C_6$ (nodes 1–2–3–4–5–6–1). Is the matrix symmetric? What is the degree of each node? What is the diameter?

### Analyze

**1.7.** Compare BFS and DFS for computing the shortest path between two nodes in an unweighted, undirected graph. Which algorithm is guaranteed to find the shortest path and why? Describe a case where DFS would explore more nodes than BFS before finding the target.

**1.8.** A social network has 1 million users, average degree 50, and follows a power-law degree distribution with $\gamma = 2.1$. Estimate: (a) the expected number of edges, (b) the expected number of users with degree $\geq 1000$, and (c) the expected average path length (use the $\log n / \log \bar{d}$ approximation). What does the heavy tail imply about neighbor sampling strategies for GNNs on this network?

### Evaluate

**1.9.** A team proposes to represent a knowledge graph (entities: people, places, organizations; relations: *lives_in*, *works_for*, *founded*) as a simple undirected graph by ignoring edge types. Evaluate the consequences: what information is lost, and which GNN architectures would be appropriate for the original heterogeneous representation vs. the simplified undirected representation? When might the simplification be acceptable?

**1.10.** The clustering coefficient and average path length are both summary statistics of a graph. Evaluate their limitations as *discriminative* features for graph classification. Construct two non-isomorphic graphs that have the same clustering coefficient and average path length. What does this imply for approaches that use only these statistics as graph-level features?

### Create

**1.11.** Design a graph representation for a university course prerequisite system with at least 10 courses. Specify: (a) the node and edge types, (b) whether the graph should be directed or undirected and why, (c) what node features would be useful for a GNN that predicts whether a student will pass a course, and (d) what graph ML task you would formulate the passing-prediction problem as. Implement your graph in NetworkX.

**1.12.** Implement a function `graph_report(G)` in Python that accepts any NetworkX graph and prints a comprehensive one-page summary including: number of nodes and edges, directed/undirected, degree statistics (mean, median, max, min), the top-5 highest-degree nodes, number of connected components, average clustering coefficient, diameter (if connected), and a text-art degree distribution histogram. Test it on the Karate Club graph and on a random Barabási-Albert graph generated with `nx.barabasi_albert_graph(100, 2)`. Compare and interpret the difference in their degree distributions.

---

## 1.16 Further Reading

**[1] Barabási & Albert (1999) — Emergence of Scaling in Random Networks.** *Science 286(5439).*
The paper that introduced the preferential attachment model and coined the term "scale-free network." Reading this alongside a plot of real-world degree distributions makes the power-law phenomenon concrete. The paper is short (~3 pages) and accessible without deep mathematical background.

**[2] Watts & Strogatz (1998) — Collective Dynamics of Small-World Networks.** *Nature 393(6684).*
The landmark paper introducing the small-world model, demonstrating that interpolating between a regular lattice and a random graph produces networks with both high clustering and low path length. The Karate Club graph is a prototypical small-world network.

**[3] Zachary (1977) — An Information Flow Model for Conflict and Fission in Small Groups.** *Journal of Anthropological Research 33(4).*
The original paper describing the Karate Club dataset that appears in virtually every GNN paper and tutorial. Worth reading to understand what the node labels actually mean — the data predates graph neural networks by decades.

**[4] Hamilton (2020) — Graph Representation Learning (book), Chapters 1–2.** *Free PDF online.*
The most accessible graduate-level textbook on graph ML. Chapter 1 introduces graph theory concisely; Chapter 2 covers background graph statistics (degree distributions, clustering, etc.) at the same level as this chapter, providing useful supplementary exercises.

**[5] Barabási (2016) — Network Science (book), Chapters 1–4.** *Free online at networksciencebook.com.*
A visually rich introduction to network science with interactive figures. Chapters 2–4 cover random graphs, scale-free networks, and the Barabási-Albert model in depth. An excellent companion for the network-science half of this chapter.

**[6] Hu et al. (2020) — Open Graph Benchmark: Datasets for Machine Learning on Graphs.** *NeurIPS 2020.*
Introduces the OGB benchmark suite that we use for performance evaluation throughout the textbook. Section 2 describes the dataset construction, split design, and the reasoning behind the choice of evaluation metrics. Understanding the split design (especially for link prediction) is essential for fair evaluation.

**[7] Hagberg, Schult & Swart (2008) — Exploring Network Structure, Dynamics, and Function using NetworkX.** *Proceedings of the 7th Python in Science Conference.*
The original NetworkX paper. Readable in under 30 minutes and explains the design philosophy behind the library — in particular, why nodes can be arbitrary Python objects and why the adjacency is stored as a dictionary of dictionaries. This design decision enables flexibility at the cost of raw performance compared to adjacency matrix representations.

!!! mascot-celebration "Chapter 1 Complete — You Speak Graph Now!"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating chapter completion with graph-node confetti">
    You have just learned the vocabulary that the rest of this textbook is written in. Nodes, edges, adjacency matrices, degree distributions, paths, components, subgraphs, isomorphism — these are no longer abstract terms, they are tools. In Chapter 2, we will start computing features on these structures that traditional machine learning models can use, before building the GNNs that make feature engineering mostly unnecessary. Your neighborhood has more to offer — let's keep aggregating.

[See Annotated References](./references.md)