# Quiz: Frequent Subgraph Mining

Test your understanding of subgraph isomorphism, frequent pattern mining, order embeddings, and SPMiner.

---

#### Question 1

Subgraph isomorphism asks whether a query graph Q is isomorphic to some subgraph of a target graph G. What is the computational complexity of this problem?

<div class="upper-alpha" markdown>
1. \(O(N \log N)\) — efficiently solvable using topological sort
2. \(O(N^2 + E)\) — solvable in polynomial time using BFS/DFS
3. NP-complete — no known polynomial-time algorithm exists for the general case
4. \(O(N^3)\) — solvable in cubic time using dynamic programming
</div>

??? question "Show Answer"
    The correct answer is **C**. Subgraph isomorphism is NP-complete: deciding whether a query graph Q appears as a subgraph in a target graph G is at least as hard as graph isomorphism, and the special case of finding Hamiltonian paths (an NP-complete problem) reduces to it. In practice, algorithms like VF2 and Ullmann's use backtracking with pruning and are fast for many practical cases, but worst-case runtime is exponential in |V(Q)|. This computational hardness motivates approximate neural approaches like SPMiner.

    **Concept Tested:** Subgraph Isomorphism

---

#### Question 2

Frequent subgraph mining searches for patterns that appear in at least min_support graphs in a database. The anti-monotonicity property is key to making this tractable. What does anti-monotonicity state?

<div class="upper-alpha" markdown>
1. If a subgraph pattern S is infrequent, then all supergraphs of S (graphs containing S as a subgraph) are also infrequent
2. If a subgraph pattern S is infrequent (below min_support), then all graphs that contain S as a subgraph are also infrequent
3. Larger subgraphs are always more frequent than smaller ones in dense graphs
4. A subgraph's frequency decreases monotonically as the minimum support threshold increases
</div>

??? question "Show Answer"
    The correct answer is **A**. Anti-monotonicity (also called the downward closure property): if pattern S is frequent, then every subgraph of S must also be frequent (if you can find pattern S in many graphs, you can certainly find all of S's smaller subpatterns in at least as many graphs). Equivalently, if S is infrequent, no supergraph of S can be frequent. This enables level-wise search (Apriori-style): enumerate patterns of size k only from frequent patterns of size k−1, pruning any candidate whose sub-patterns are already infrequent.

    **Concept Tested:** Frequent Subgraph Mining

---

#### Question 3

Order embeddings represent the partial order relationship "Q is a subgraph of G" in embedding space. What geometric property must the embedding satisfy?

<div class="upper-alpha" markdown>
1. Embeddings of isomorphic graphs must be the same vector
2. Embeddings of query and target graphs must lie in orthogonal subspaces
3. The cosine similarity between Q and G embeddings must equal the subgraph frequency
4. If Q is a subgraph of G, then Q's embedding must be element-wise less than or equal to G's embedding: \(\mathbf{z}_Q \leq \mathbf{z}_G\) (component-wise)
</div>

??? question "Show Answer"
    The correct answer is **D**. Order embeddings map graphs to vectors in \(\mathbb{R}^d\) such that \(Q \subseteq G\) (Q is a subgraph of G) implies \(\mathbf{z}_Q \leq \mathbf{z}_G\) component-wise (Q's embedding is dominated by G's embedding in all dimensions). This partial order in the original space is preserved as coordinate dominance in the embedding space. The penalty for violating this order is \(E(Q, G) = \|\max(0, \mathbf{z}_Q - \mathbf{z}_G)\|^2\) — zero when the order is satisfied, positive otherwise. This allows subgraph relationship queries to be answered in \(O(d)\) time via embedding lookup.

    **Concept Tested:** Order Embedding

---

#### Question 4

SPMiner is a neural method for frequent subgraph mining that uses GNN encoders and order embeddings. What is the advantage of SPMiner over classical combinatorial methods like gSpan?

<div class="upper-alpha" markdown>
1. SPMiner always finds the exact set of all frequent patterns, while gSpan finds only approximate patterns
2. SPMiner replaces combinatorial search with embedding-space traversal, enabling frequency estimation in milliseconds rather than hours and scaling to large graphs where exhaustive search is infeasible
3. SPMiner requires no training data and generalizes zero-shot to any graph database
4. SPMiner uses parallel search across multiple pattern candidates simultaneously, reducing wall-clock time
</div>

??? question "Show Answer"
    The correct answer is **B**. Classical methods like gSpan perform exhaustive search with pruning — still exponential in the worst case. For a large graph database with complex patterns, gSpan can take hours. SPMiner trains a GNN to embed graphs in order embedding space, then uses a greedy traversal in embedding space to identify high-frequency regions — no explicit subgraph matching required. Frequency estimation reduces to counting how many target graph embeddings dominate a query embedding, executable in milliseconds with nearest-neighbor search.

    **Concept Tested:** SPMiner, Frequent Subgraph Mining

---

#### Question 5

Why is exact subgraph frequency counting intractable for large networks even when approximate isomorphism checks are available?

<div class="upper-alpha" markdown>
1. The number of candidate subgraph embeddings grows exponentially with the size of the query pattern and the graph, making enumeration infeasible for patterns larger than a few nodes
2. Exact counting requires computing the graph's eigenspectrum, which is \(O(N^3)\)
3. Large networks cannot be stored in adjacency matrix format, preventing exact computation
4. Exact counting requires all node labels to be unique, which is not satisfied in real networks
</div>

??? question "Show Answer"
    The correct answer is **A**. For a query \(Q\) with \(k\) nodes, the number of candidate node mappings in target \(G\) is \(O(N^k)\) — exponential in \(k\). Even with pruning via subgraph isomorphism checks, this blows up: counting all triangles (\(k=3\)) is \(O(N^3)\), 4-cliques are \(O(N^4)\), and general patterns of size \(k\) require checking up to \(\binom{N}{k} \times k!\) candidate mappings. This motivates both approximate counting (color-coding, sampling) and neural approaches (SPMiner) that estimate frequency without exhaustive enumeration.

    **Concept Tested:** Subgraph Isomorphism, Frequent Subgraph Mining

---

#### Question 6

In order embedding training, the loss function uses two types of examples: positive pairs (Q ⊆ G) and negative pairs (Q ⊄ G). What is the training objective?

<div class="upper-alpha" markdown>
1. Predict a binary label (is_subgraph) using a sigmoid classifier on the concatenated embeddings
2. Minimize cosine distance for positive pairs; maximize cosine distance for negative pairs
3. For positive pairs: minimize \(E(Q,G) = \|\max(0, \mathbf{z}_Q - \mathbf{z}_G)\|^2\) toward 0; for negative pairs: push \(E(Q,G)\) above a margin — ensure the order violation penalty is large when \(Q\) is NOT a subgraph of \(G\)
4. Minimize the L2 distance between embeddings of graphs with the same number of nodes
</div>

??? question "Show Answer"
    The correct answer is **C**. The order embedding loss has two components: (1) for true subgraph pairs (\(Q \subseteq G\)), minimize \(E(Q,G) = \|\max(0, \mathbf{z}_Q - \mathbf{z}_G)\|^2 \to 0\) (the embedding of \(Q\) must be dominated by \(G\)'s); (2) for non-subgraph pairs (\(Q \not\subseteq G\)), maximize \(E(Q,G)\) above a margin — ensure the embedding of \(Q\) is NOT dominated by \(G\)'s (at least one dimension of \(\mathbf{z}_Q\) exceeds \(\mathbf{z}_G\)). This contrastive loss trains the encoder to embed the subgraph partial order geometrically.

    **Concept Tested:** Order Embedding

---

#### Question 7

The canonical form of a subgraph pattern (used in gSpan) ensures each pattern is represented by a unique DFS code. Why is canonical form necessary for subgraph mining?

<div class="upper-alpha" markdown>
1. DFS codes are more memory-efficient than adjacency matrices for storing patterns
2. Without canonical form, the same structural pattern would be counted multiple times (once per different node labeling or traversal order), inflating frequency counts
3. Canonical form converts all patterns to a tree structure, enabling faster isomorphism checking
4. Canonical form prevents duplicate candidates from being generated during breadth-first expansion
</div>

??? question "Show Answer"
    The correct answer is **B**. The same 4-node path A-B-C-D can be traversed in multiple ways (A→B→C→D, D→C→B→A, B→A→C→D, etc.), each producing a different "surface form" of the pattern. Without canonical form, the mining algorithm would generate and count each surface form as a separate candidate — dramatically inflating both the search space and the frequency counts. Canonical form (minimum DFS code in gSpan) assigns a unique representation to each structural pattern class, enabling duplicate-free enumeration and accurate frequency counting.

    **Concept Tested:** Frequent Subgraph Mining, Subgraph Isomorphism

---

#### Question 8

SPMiner searches for frequent patterns by navigating the order embedding space. How does it identify a "dense region" corresponding to a frequent pattern?

<div class="upper-alpha" markdown>
1. It clusters the target graph embeddings using k-means and interprets each cluster as a frequent pattern
2. It samples random walks in embedding space and counts the number of pattern visits
3. It uses gradient ascent to find the embedding that maximizes the sum of dot products with all target embeddings
4. It places a query point \(\mathbf{z}_q\) in embedding space and counts how many target graph embeddings dominate \(\mathbf{z}_q\) (i.e., satisfy \(\mathbf{z}_Q \leq \mathbf{z}_q\)); regions with many dominating targets correspond to patterns that appear in many graphs
</div>

??? question "Show Answer"
    The correct answer is **D**. In order embedding space, a query at position \(\mathbf{z}_q\) is a subgraph of all targets G where \(\mathbf{z}_q \leq \mathbf{z}_G\) (component-wise dominance). The frequency estimate of the pattern corresponding to \(\mathbf{z}_q\) is the count of target embeddings that dominate \(\mathbf{z}_q\) — i.e., the volume of the "upper orthant" above \(\mathbf{z}_q\) occupied by target points. SPMiner searches for \(\mathbf{z}_q\) values in high-density upper orthants using a greedy graph growing strategy: start with a single node, add one node/edge at a time choosing moves that maximize domination count.

    **Concept Tested:** SPMiner, Order Embedding

---

#### Question 9

In biological networks, subgraph mining is used to identify recurring structural motifs. Why are triangles (3-cliques) particularly significant in protein-protein interaction networks?

<div class="upper-alpha" markdown>
1. Triangles appear in all biological networks by construction due to the way interaction data is collected
2. Triangles are the only subgraph pattern that can be detected in polynomial time
3. Triangles indicate that three proteins mutually interact, often forming stable protein complexes or regulatory circuits — their over-representation signals biological function beyond random network structure
4. Triangles are the only pattern type that can be discovered by the Louvain community detection algorithm
</div>

??? question "Show Answer"
    The correct answer is **C**. In protein-protein interaction (PPI) networks, a triangle means proteins A, B, and C all interact pairwise. This mutual interaction pattern is characteristic of protein complexes (all subunits interact), feedback regulatory loops (A activates B which activates C which inhibits A), and co-localized functional modules. The enrichment of triangles beyond random expectation (measured via Z-score) indicates that the network's architecture is non-random and reflects underlying biological organization — exactly what network motif analysis is designed to reveal.

    **Concept Tested:** Frequent Subgraph Mining, Network Motif (from Ch02)

---

#### Question 10

Classical frequent subgraph mining (gSpan, Gaston) uses anti-monotonicity for pruning. What is the maximum size of patterns that can be practically mined in a database of 1000 graphs each with ~100 nodes?

<div class="upper-alpha" markdown>
1. Up to 3 nodes (triangles) — patterns larger than size 3 are always intractable
2. Exactly 5 nodes — all 5-node patterns are tractable and all 6-node patterns are not
3. Unlimited — anti-monotonicity guarantees polynomial runtime for any pattern size
4. Up to 8–12 nodes in practice — with min_support pruning, the search space collapses dramatically for patterns above ~50% support, making moderately large patterns tractable within minutes
</div>

??? question "Show Answer"
    The correct answer is **D**. Anti-monotonicity pruning is extremely effective in practice: when min_support is set above ~50% of graphs, the frequent pattern set is small and patterns of size up to ~12 nodes can be mined within reasonable time. However, lowering min_support dramatically expands the search space — at 10% support, even 5-node patterns may become intractable for large graph databases. The practical limit depends on both the support threshold and the structural diversity of the database, not a fixed node count.

    **Concept Tested:** Frequent Subgraph Mining
