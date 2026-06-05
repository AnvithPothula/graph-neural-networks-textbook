# Quiz: Introduction to Graphs and Networks

Test your understanding of graph fundamentals, network properties, and the language of graph theory.

---

#### Question 1

In a directed graph (digraph), what is the difference between in-degree and out-degree of a node?

<div class="upper-alpha" markdown>
1. In-degree is the number of edges arriving at the node; out-degree is the number of edges leaving it
2. In-degree counts self-loops; out-degree counts edges to other nodes
3. In-degree counts unweighted edges; out-degree counts weighted edges
4. In-degree and out-degree are identical for all nodes in a strongly connected graph
</div>

??? question "Show Answer"
    The correct answer is **A**. In a directed graph, each edge has a source and a destination. The in-degree of a node v is the count of edges pointing *into* v (edges with v as destination), while the out-degree is the count of edges pointing *out* of v (edges with v as source). In social networks, in-degree often measures popularity (how many people follow you) and out-degree measures activity (how many people you follow).

    **Concept Tested:** In-Degree, Out-Degree

---

#### Question 2

Which property best describes a power-law degree distribution in a real-world network?

<div class="upper-alpha" markdown>
1. A small number of hub nodes have extremely high degree while most nodes have very low degree
2. Node degrees follow a Gaussian (bell-curve) distribution centered at the average degree
3. Every node has exactly the same degree (a perfectly regular network)
4. Degrees increase linearly with the node's distance from the center of the network
</div>

??? question "Show Answer"
    The correct answer is **A**. In a power-law (scale-free) network, the fraction of nodes with degree \(k\) follows \(P(k) \propto k^{-\gamma}\) for some exponent \(\gamma > 1\). This heavy-tailed distribution means a tiny number of hubs (like Google or major Twitter accounts) have millions of connections while the vast majority of nodes have just a few. The Internet, citation networks, and protein interaction networks all exhibit this property.

    **Concept Tested:** Power-Law Degree Distribution

---

#### Question 3

What is the diameter of a connected graph?

<div class="upper-alpha" markdown>
1. The total number of edges in the graph
2. The average shortest path length across all pairs of nodes
3. The length of the longest shortest path between any two nodes in the graph
4. The degree of the highest-degree node in the graph
</div>

??? question "Show Answer"
    The correct answer is **C**. The diameter is defined as \(\max_{u,v} d(u, v)\) — the maximum over all node pairs of the shortest path distance between them. It captures the "worst-case separation" in the network. The famous "six degrees of separation" result claims that the diameter of the human social network is approximately 6, meaning any two people are connected by at most six acquaintances.

    **Concept Tested:** Graph Diameter

---

#### Question 4

A bipartite graph has two disjoint node sets U and V. Which statement is always true about a bipartite graph?

<div class="upper-alpha" markdown>
1. Every node in U has exactly the same degree as every node in V
2. Every edge connects a node in U to a node in V; no edges exist within U or within V
3. The graph contains no cycles of any length
4. The graph must be directed, with all edges pointing from U to V
</div>

??? question "Show Answer"
    The correct answer is **B**. By definition, a bipartite graph partitions its nodes into two sets such that every edge crosses the partition — no edge connects two nodes within the same set. User-item graphs in recommender systems are classic bipartite graphs: users form one set, items form the other, and edges represent interactions. Bipartite graphs can contain even-length cycles and can be either directed or undirected.

    **Concept Tested:** Bipartite Graph

---

#### Question 5

The adjacency matrix \(A\) of an undirected graph with \(N\) nodes has the entry \(A_{ij} = 1\). What does this indicate?

<div class="upper-alpha" markdown>
1. Node \(i\) is at distance 1 from node \(j\) in the shortest-path metric
2. There is an edge between node \(i\) and node \(j\)
3. Node \(i\) has exactly one more neighbor than node \(j\)
4. The subgraph induced by nodes \(i\) and \(j\) is a clique
</div>

??? question "Show Answer"
    The correct answer is **B**. The adjacency matrix encodes connectivity: \(A_{ij} = 1\) if and only if there is an edge between node \(i\) and node \(j\), and 0 otherwise. For undirected graphs, \(A\) is symmetric (\(A_{ij} = A_{ji}\)). The row sum of row \(i\) equals the degree of node \(i\). The matrix entry \((A^2)_{ij}\) counts the number of paths of length 2 from \(i\) to \(j\) — a fact used extensively in spectral graph theory.

    **Concept Tested:** Adjacency Matrix

---

#### Question 6

What is an ego network centered on node v?

<div class="upper-alpha" markdown>
1. The clique of maximum size that includes node v
2. The shortest path tree rooted at v spanning the entire connected component
3. The set of all nodes reachable from v within k hops for any k ≥ 2
4. The subgraph consisting of v, all of v's immediate neighbors, and all edges among them
</div>

??? question "Show Answer"
    The correct answer is **D**. The ego network of v (also called the 1-hop neighborhood subgraph) includes v itself (the "ego"), all nodes directly connected to v (the "alters"), and every edge among the alters. Ego networks are used to analyze local structure: clustering coefficient, role detection, and feature engineering for node classification all rely on ego network properties. In GNNs, a single message-passing layer aggregates over exactly the ego network.

    **Concept Tested:** Ego Network

---

#### Question 7

Two graphs G₁ and G₂ are isomorphic. What does this mean?

<div class="upper-alpha" markdown>
1. G₁ and G₂ have the same number of nodes and the same total edge weight
2. G₁ is a subgraph of G₂ and G₂ is a subgraph of G₁
3. There exists a bijection between their node sets that preserves all edge relationships
4. G₁ and G₂ can be embedded in the same 2D plane without edge crossings
</div>

??? question "Show Answer"
    The correct answer is **C**. Graph isomorphism means there exists a bijection f: V(G₁) → V(G₂) such that (u, v) is an edge in G₁ if and only if (f(u), f(v)) is an edge in G₂. Isomorphic graphs are structurally identical — just with different node labels. Deciding whether two graphs are isomorphic is a computationally hard problem, and the inability of GNNs to distinguish certain non-isomorphic graphs is the central topic of Chapter 9.

    **Concept Tested:** Graph Isomorphism

---

#### Question 8

What distinguishes a strongly connected component (SCC) from a weakly connected component in a directed graph?

<div class="upper-alpha" markdown>
1. In an SCC, every node can reach every other node by following directed edges; weak connectivity only requires connectivity when edge directions are ignored
2. An SCC requires all nodes to have equal in-degree and out-degree
3. An SCC contains no cycles, while a weakly connected component always contains at least one cycle
4. An SCC is a subgraph with exactly one source node and one sink node
</div>

??? question "Show Answer"
    The correct answer is **A**. Strong connectivity is a stricter condition: every node in the SCC must be reachable from every other node *following the direction of edges*. Weak connectivity is satisfied if the underlying undirected graph (ignoring directions) is connected. A directed graph may have multiple SCCs even if it is weakly connected — for example, a directed path A→B→C forms three SCCs (each node alone) but one weakly connected component.

    **Concept Tested:** Strongly Connected Component

---

#### Question 9

The four main graph machine learning tasks are node classification, link prediction, graph classification, and graph generation. Which task involves predicting properties of an entire molecule given its atom-bond structure?

<div class="upper-alpha" markdown>
1. Node classification
2. Link prediction
3. Graph classification
4. Graph generation
</div>

??? question "Show Answer"
    The correct answer is **C**. Graph classification assigns a label to an entire graph, treating the whole structure as a single input. Predicting whether a molecule is toxic or active against a given target (as in ogbg-molhiv) is graph classification — the graph represents the molecule and the label is a molecular property. Node classification would instead assign labels to individual atoms; link prediction would predict missing bonds.

    **Concept Tested:** Graph (Undirected), Graph (Directed)

---

#### Question 10

The Zachary Karate Club graph is a canonical benchmark used throughout this textbook. Which property makes it ideal for illustrating GNN concepts in the early chapters?

<div class="upper-alpha" markdown>
1. It has 1 million nodes, making it suitable for scalability benchmarks
2. Its degree distribution follows a perfect Gaussian, making mathematical analysis tractable
3. It is the only publicly available graph with labeled node features
4. It has two known ground-truth communities that emerged from a real social split, and it is small enough (34 nodes) to visualize completely
</div>

??? question "Show Answer"
    The correct answer is **D**. The Karate Club graph (Zachary 1977) has 34 nodes and 78 edges representing friendships in a university karate club. After a dispute, the club split into two factions — providing ground-truth community labels. Its small size allows complete visualization, yet it exhibits rich structure (community structure, bridging edges, degree heterogeneity) that makes it an ideal teaching graph for Chapters 1–9.

    **Concept Tested:** Node (Vertex), Connected Component
