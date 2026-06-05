# Quiz: Link Analysis and PageRank

Test your understanding of PageRank, random walks, power iteration, and link-based authority measures.

---

#### Question 1

What is the core idea behind the PageRank formula \(r(v) = \sum_{u \to v} r(u) / d_{\text{out}}(u)\)?

<div class="upper-alpha" markdown>
1. A page's rank is proportional to the sum of rank contributed by each page linking to it, split evenly among that page's outgoing links
2. A page's rank equals the fraction of the web's total links that point to it
3. A page's rank is the sum of its outgoing links weighted by the destination's rank
4. A page's rank is determined by the number of keywords it contains divided by its out-degree
</div>

??? question "Show Answer"
    The correct answer is **A**. PageRank distributes a page's importance equally among all pages it links to. Each linking page \(u\) contributes \(r(u) / d_{\text{out}}(u)\) to \(v\) — its full rank divided by the number of outgoing links. This models a random surfer who follows links uniformly at random: the stationary probability of visiting \(v\) is exactly \(\text{PageRank}(v)\). A page is important if important pages link to it, capturing the recursive authority relationship.

    **Concept Tested:** PageRank

---

#### Question 2

What is a "spider trap" in the context of PageRank, and how does teleportation solve it?

<div class="upper-alpha" markdown>
1. A group of nodes whose outgoing links only point to each other, gradually accumulating all rank; teleportation gives every node a non-zero probability of jumping to any page
2. A node with no outgoing edges that absorbs all rank; teleportation redistributes its rank uniformly
3. A cycle of length 2 that causes power iteration to oscillate; teleportation breaks the cycle by adding self-loops
4. A node with more than 1000 outgoing links that slows convergence; teleportation caps out-degree at a fixed constant
</div>

??? question "Show Answer"
    The correct answer is **A**. A spider trap is a strongly connected subgraph with no outgoing edges to the rest of the graph. Once a random surfer enters, they can never leave, so PageRank mass accumulates indefinitely inside the trap. Teleportation — represented by the damping factor d — gives the surfer a probability (1 − d) of jumping to a uniformly random page at each step, regardless of link structure. This breaks the trap and guarantees every node receives non-zero rank.

    **Concept Tested:** Teleportation (PageRank), Spider Trap

---

#### Question 3

The PageRank vector r is the stationary distribution of a random walk on the web graph. What does the stationary distribution mean in this context?

<div class="upper-alpha" markdown>
1. The distribution over nodes that remains unchanged after one more step of the random walk — the long-run fraction of time the surfer spends at each page
2. The distribution that the random walk reaches after exactly N steps, where N is the number of nodes
3. The distribution that assigns equal probability to all nodes simultaneously
4. The final state reached when the random walk visits each edge exactly once
</div>

??? question "Show Answer"
    The correct answer is **A**. A stationary distribution \(\pi\) satisfies \(\pi = M\pi\), where \(M\) is the column-stochastic transition matrix. It represents the long-run fraction of time a random surfer spends at each node — equivalently, the probability of finding the surfer at a given page after infinitely many steps. For the PageRank Markov chain (with teleportation ensuring ergodicity), this stationary distribution exists, is unique, and equals the PageRank vector.

    **Concept Tested:** Stationary Distribution, Random Walk

---

#### Question 4

Power iteration for PageRank starts with a uniform rank vector and repeatedly applies the update \(r \leftarrow Mr\). Why does this converge to the PageRank vector?

<div class="upper-alpha" markdown>
1. Because M is a symmetric matrix, its eigenvectors form an orthonormal basis guaranteeing convergence
2. Because the column-stochastic matrix M has 1 as its largest eigenvalue, so repeated multiplication amplifies the corresponding eigenvector relative to all others
3. Because power iteration solves the PageRank linear system exactly in at most N iterations
4. Because M is sparse, allowing gradient descent to minimize the PageRank objective function efficiently
</div>

??? question "Show Answer"
    The correct answer is **B**. For an ergodic (irreducible, aperiodic) column-stochastic matrix, the Perron–Frobenius theorem guarantees a unique dominant eigenvalue of 1. All other eigenvalues have magnitude < 1. Repeated multiplication by M drives contributions from non-dominant eigenvectors to zero exponentially fast, leaving only the eigenvector corresponding to eigenvalue 1 — the stationary distribution. Convergence is typically reached in 50–100 iterations even for billion-node graphs.

    **Concept Tested:** Power Iteration

---

#### Question 5

Personalized PageRank (PPR) for a query node q differs from standard PageRank in what fundamental way?

<div class="upper-alpha" markdown>
1. PPR uses a directed graph while standard PageRank requires undirected graphs
2. PPR runs backward from q toward its predecessors instead of forward along outgoing edges
3. The teleportation distribution in PPR always returns to node q rather than to a uniformly random node
4. PPR replaces the damping factor with the clustering coefficient of q
</div>

??? question "Show Answer"
    The correct answer is **C**. In standard PageRank, teleportation jumps to any node with probability 1/N. In Personalized PageRank, teleportation always returns to the query node q (or a small seed set). This biases the stationary distribution toward nodes structurally close to q, making PPR a measure of proximity from q rather than global importance. PPR is central to local community detection and forms the basis of the propagation in GraphSAGE-style neighborhood sampling.

    **Concept Tested:** Personalized PageRank

---

#### Question 6

HITS assigns two scores to each web page: hub score and authority score. What is the key relationship between these scores?

<div class="upper-alpha" markdown>
1. A page's hub score equals the sum of hub scores of pages it links to; authority score equals hub score divided by degree
2. Hub score equals the number of inbound links and authority score equals the number of outbound links, both normalized by the maximum in the graph
3. A page's authority score is proportional to the sum of hub scores of pages linking to it; a page's hub score is proportional to the sum of authority scores of pages it links to
4. Hub and authority scores are identical for undirected graphs and differ only for directed graphs with asymmetric degree distributions
</div>

??? question "Show Answer"
    The correct answer is **C**. HITS defines a mutually reinforcing system: a good hub points to many good authorities, and a good authority is pointed to by many good hubs. Formally, \(a(v) \propto \sum_{u \to v} h(u)\) and \(h(v) \propto \sum_{v \to w} a(w)\). This system can be written as \(a = A^\top A a\) and \(h = A A^\top h\) — the leading eigenvectors of \(A^\top A\) and \(A A^\top\) respectively. HITS tends to find topic-specific authorities rather than the single global ranking of PageRank.

    **Concept Tested:** HITS Algorithm

---

#### Question 7

A random walk on an undirected graph visits node v with stationary probability proportional to its degree. Given the Karate Club graph, which node would a random walk spend the most time at?

<div class="upper-alpha" markdown>
1. An arbitrary peripheral leaf node with degree 1
2. The node at the exact geometric center of the graph
3. A node with exactly the average degree, since stationary probability is proportional to average degree
4. The highest-degree hub node (instructor or president), which has the most connections
</div>

??? question "Show Answer"
    The correct answer is **D**. For a random walk on an undirected graph, the stationary probability of visiting node \(v\) is \(\pi(v) = d(v) / (2|E|)\), proportional to its degree. In the Karate Club graph, the club instructor (node 1) and president (node 34) have the highest degrees and therefore attract the most random walk visits. This is why PageRank correlates strongly with degree in undirected graphs — though directed link structure can override this in web graphs.

    **Concept Tested:** Random Walk, Stationary Distribution

---

#### Question 8

A "dead end" node in the PageRank formulation is a node with no outgoing edges. What problem does it cause for the random walk model?

<div class="upper-alpha" markdown>
1. It makes the transition matrix non-symmetric, preventing eigenvector computation
2. It forces the random surfer into an infinite loop, causing power iteration to diverge
3. It absorbs rank without redistributing it, causing the total rank in the system to leak away to zero over iterations
4. It doubles the effective degree of all neighboring nodes, skewing rank estimates
</div>

??? question "Show Answer"
    The correct answer is **C**. A dead end node has a column of zeros in the stochastic transition matrix — when the surfer arrives, there is nowhere to go, and rank is "leaked" from the system. Without correction, the total PageRank mass across all nodes converges to zero. The standard fix is to treat dead ends as if they have outgoing edges to all N nodes (teleporting with probability 1), which corresponds to replacing zero columns with 1/N in the transition matrix.

    **Concept Tested:** Teleportation (PageRank)

---

#### Question 9

The damping factor d in PageRank is typically set to 0.85. What does this mean for the random surfer model?

<div class="upper-alpha" markdown>
1. The surfer visits each node exactly 0.85 times on average per full traversal of the graph
2. The surfer follows an outgoing link with probability 0.85 and teleports to a random page with probability 0.15
3. The surfer ranks only the top 85% of pages by degree and ignores the rest
4. The surfer's walk terminates after exactly 85 steps with probability 1
</div>

??? question "Show Answer"
    The correct answer is **B**. At each step, the random surfer model with damping factor d = 0.85 flips a biased coin: with probability 0.85 the surfer follows a uniformly random outgoing link; with probability 0.15 the surfer teleports to a uniformly random page in the entire graph. This (1 − d) = 0.15 teleportation probability simultaneously handles dead ends, breaks spider traps, and ensures the Markov chain is ergodic, guaranteeing a unique stationary distribution.

    **Concept Tested:** Teleportation (PageRank), PageRank

---

#### Question 10

Why is Personalized PageRank (PPR) useful as a neighborhood aggregation strategy in GNNs such as APPNP?

<div class="upper-alpha" markdown>
1. PPR computes the exact k-hop neighborhood for each node, avoiding over-smoothing entirely
2. PPR identifies which nodes to exclude from aggregation rather than which nodes to include
3. PPR replaces matrix multiplication with scalar addition, making GNN training 100× faster
4. PPR assigns high scores to nodes that are structurally close to the query node while exponentially discounting distant nodes, providing a principled soft neighborhood
</div>

??? question "Show Answer"
    The correct answer is **D**. PPR scores nodes by their teleportation-adjusted proximity to a query node q. High-PPR nodes are those a random surfer starting at q would visit frequently — structurally nearby and well-connected to q's neighborhood. APPNP uses PPR as the aggregation weights, giving a smooth decay over multi-hop neighborhoods that avoids the binary "include/exclude at hop k" cutoff. The teleportation term acts as a residual connection, reducing over-smoothing at large propagation depths.

    **Concept Tested:** Personalized PageRank
