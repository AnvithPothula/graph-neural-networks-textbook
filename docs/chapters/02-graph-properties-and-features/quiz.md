# Quiz: Graph Properties and Traditional ML Features

Test your understanding of network models, centrality measures, and classical graph feature engineering.

---

#### Question 1

What is the local clustering coefficient of a node v?

<div class="upper-alpha" markdown>
1. The ratio of v's degree to the maximum degree in the graph
2. The number of triangles passing through v divided by the graph diameter
3. The average shortest path length from v to all other nodes
4. The fraction of pairs among v's neighbors that are also connected to each other
</div>

??? question "Show Answer"
    The correct answer is **D**. The local clustering coefficient \(C(v) = \text{(edges among } v\text{'s neighbors)} / \binom{k}{2}\), where \(k\) is \(v\)'s degree. It measures how "clique-like" the neighborhood of \(v\) is — a value of 1 means all neighbors are connected to each other, while 0 means no neighbors are connected. High clustering is a hallmark of social networks, where friends of friends tend to also be friends.

    **Concept Tested:** Local Clustering Coefficient

---

#### Question 2

The Barabási–Albert model generates scale-free networks through a mechanism called preferential attachment. What does preferential attachment mean?

<div class="upper-alpha" markdown>
1. New nodes are added uniformly at random and connect to existing nodes with equal probability
2. Existing nodes preferentially delete edges to low-degree nodes over time
3. New nodes preferentially connect to high-degree nodes, making rich nodes get richer
4. The network grows by duplicating existing nodes along with all their connections
</div>

??? question "Show Answer"
    The correct answer is **C**. In the Barabási–Albert model, when a new node joins the network it attaches to existing node \(i\) with probability proportional to \(i\)'s current degree: \(P(\text{attach to } i) \propto k_i\). This "rich-get-richer" mechanism causes a small number of early or high-degree nodes to accumulate far more connections than average, producing the power-law degree distribution observed in the Web, citation networks, and protein interaction networks.

    **Concept Tested:** Preferential Attachment, Barabási–Albert Model

---

#### Question 3

Betweenness centrality of a node v measures which of the following?

<div class="upper-alpha" markdown>
1. The fraction of all shortest paths between other node pairs that pass through v
2. The reciprocal of the average shortest path distance from v to all other nodes
3. The eigenvector component of v in the leading eigenvector of the adjacency matrix
4. The number of triangles that include v as a vertex
</div>

??? question "Show Answer"
    The correct answer is **A**. Betweenness centrality \(\text{BC}(v) = \sum_{s \neq v \neq t} \sigma(s,t|v) / \sigma(s,t)\), where \(\sigma(s,t)\) is the total number of shortest paths from \(s\) to \(t\) and \(\sigma(s,t|v)\) is the number of those paths passing through \(v\). High betweenness indicates a node that acts as a bridge or bottleneck — removing it would disrupt many shortest paths. In the Karate Club graph, the instructor and club president have high betweenness because they bridge the two communities.

    **Concept Tested:** Betweenness Centrality

---

#### Question 4

The Weisfeiler-Lehman (WL) graph kernel compares two graphs by iteratively refining node color labels. What determines the final color of a node at each iteration?

<div class="upper-alpha" markdown>
1. The node's original label only, ignoring neighbors entirely
2. The betweenness centrality of the node at that iteration
3. The hash of the node's current label combined with the multiset of its neighbors' labels
4. The average embedding vector of the node and all nodes within two hops
</div>

??? question "Show Answer"
    The correct answer is **C**. At each WL iteration, a node's new label is computed as \(\text{hash}(\text{current-label},\ \text{sort}(\text{neighbor-labels}))\). This aggregates the neighborhood's color signature into a single value, propagating structural information outward with each step. After \(k\) iterations, two nodes have the same color if and only if their \(k\)-hop subtree structures are identical. The WL kernel counts matching color histograms across the two graphs — directly foreshadowing how GNN message passing works.

    **Concept Tested:** Weisfeiler-Lehman Kernel

---

#### Question 5

What is a network motif?

<div class="upper-alpha" markdown>
1. A small subgraph pattern that appears significantly more often in a real network than in a random graph with the same degree sequence
2. The set of all triangles containing a specific focal node
3. The largest connected component that emerges during Erdős–Rényi random graph generation
4. A single high-degree hub node in a scale-free network
</div>

??? question "Show Answer"
    The correct answer is **A**. Network motifs are recurring, statistically over-represented structural patterns. A subgraph counts as a motif if its frequency in the real network is significantly higher than its expected frequency in a random graph with the same degree sequence (measured via Z-score). Feed-forward loops in gene regulatory networks and bi-fan patterns in neural circuits are classic examples. Motifs capture the functional "building blocks" of a network.

    **Concept Tested:** Network Motif

---

#### Question 6

Eigenvector centrality assigns high scores to nodes that are connected to other high-scoring nodes. Which real-world algorithm is a direct application of this recursive definition?

<div class="upper-alpha" markdown>
1. The Louvain community detection algorithm
2. PageRank (used by Google to rank web pages)
3. Dijkstra's shortest-path algorithm
4. K-means clustering applied to node features
</div>

??? question "Show Answer"
    The correct answer is **B**. PageRank is essentially a damped version of eigenvector centrality applied to directed web graphs: a page's rank is proportional to the sum of ranks of pages that link to it, with a teleportation term to handle dangling nodes. The PageRank vector is the dominant eigenvector of the modified adjacency matrix. Both PageRank and eigenvector centrality capture the idea that being connected to important nodes makes you important.

    **Concept Tested:** Eigenvector Centrality

---

#### Question 7

In the Erdős–Rényi \(G(n, p)\) model, what happens to the giant component as \(p\) increases past the threshold \(p \approx 1/n\)?

<div class="upper-alpha" markdown>
1. The graph becomes a perfect tree with no cycles
2. The graph splits into exactly two equally sized components
3. All nodes simultaneously reach the same degree
4. A giant connected component suddenly emerges containing a constant fraction of all nodes
</div>

??? question "Show Answer"
    The correct answer is **D**. The Erdős–Rényi model exhibits a sharp phase transition at \(p = 1/n\) (equivalently, average degree = 1). Below this threshold, the graph consists of small isolated components. Above it, a giant component emerges that contains \(\Theta(n)\) nodes — a macroscopic fraction of the whole network. This percolation-like transition is one of the most celebrated results in random graph theory and has practical implications for network robustness.

    **Concept Tested:** Giant Component, Erdős–Rényi Random Graph

---

#### Question 8

A graphlet is best described as:

<div class="upper-alpha" markdown>
1. A compressed representation of the adjacency matrix using bit vectors
2. A small rooted connected non-isomorphic induced subgraph pattern
3. A node embedding produced by averaging neighbor features over k hops
4. The subgraph formed by the top-k highest-degree nodes
</div>

??? question "Show Answer"
    The correct answer is **B**. Graphlets are small (typically 2–5 node) connected non-isomorphic induced subgraph patterns. There are 2 graphlets on 2 nodes, 2 on 3 nodes, 6 on 4 nodes, and so on. For each node, the Graphlet Degree Vector (GDV) counts how many times the node participates in each graphlet at each orbiting position. GDVs provide rich structural fingerprints for node classification and graph comparison without any learning.

    **Concept Tested:** Graphlet, Graphlet Degree Vector

---

#### Question 9

Katz similarity between nodes u and v is computed as a weighted sum of the number of paths of all lengths between them. Why are shorter paths weighted more heavily?

<div class="upper-alpha" markdown>
1. Shorter paths are computationally cheaper to count
2. Longer paths are excluded entirely because they create cycles in the similarity matrix
3. A decay factor \(\beta < 1\) is raised to the power of the path length, exponentially down-weighting longer paths
4. The adjacency matrix is normalized so that longer paths receive zero weight automatically
</div>

??? question "Show Answer"
    The correct answer is **C**. Katz similarity is defined as \(K(u,v) = \sum_{\ell=1}^{\infty} \beta^\ell [A^\ell]_{uv}\), where \(\beta \in (0, 1/\lambda_1)\) is a decay factor and \(A^\ell\) counts paths of length \(\ell\). Each additional hop multiplies the contribution by \(\beta\), so a path of length 3 contributes \(\beta^3 \approx (0.1)^3 = 0.001\) of the weight of a length-1 path when \(\beta = 0.1\). This ensures convergence of the infinite sum and reflects the intuition that nearby neighbors are more relevant than distant ones.

    **Concept Tested:** Katz Similarity

---

#### Question 10

Small-world networks exhibit two properties simultaneously: high clustering and short average path lengths. Which graph model captures both properties, unlike Erdős–Rényi (low clustering) and regular lattices (long paths)?

<div class="upper-alpha" markdown>
1. The Barabási–Albert preferential attachment model
2. The Watts–Strogatz model, which interpolates between a ring lattice and a random graph via random rewiring
3. A complete graph (clique) with every node connected to every other
4. A random geometric graph where nodes connect if they are within distance r
</div>

??? question "Show Answer"
    The correct answer is **B**. The Watts–Strogatz model starts with a regular ring lattice (high clustering, long paths) and rewires each edge with probability p. Even small p creates random shortcuts that drastically reduce average path length while preserving most of the local clustering. This captures the "six degrees of separation" phenomenon: social networks have tight friend circles (high clustering) yet any two people are just a few handshakes apart (short paths).

    **Concept Tested:** Small-World Network, Transitivity
