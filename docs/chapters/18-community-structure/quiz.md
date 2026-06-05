# Quiz: Community Structure in Networks

Test your understanding of community detection, modularity, Louvain, Girvan-Newman, and overlapping communities.

---

#### Question 1

Modularity Q measures the quality of a network partition. What does Q > 0 indicate?

<div class="upper-alpha" markdown>
1. The number of communities is greater than zero
2. The partition has more intra-community edges than expected in a random graph with the same degree sequence, indicating genuine community structure
3. All nodes are in a single community
4. The partition violates the minimum cut criterion
</div>

??? question "Show Answer"
    The correct answer is **B**. Modularity:

    \[Q = \text{(fraction of edges within communities)} - \text{(expected fraction at random, preserving degrees)}\]

    \(Q \in (-1, 1)\); \(Q > 0\) means the observed intra-community edge density exceeds random expectation, indicating real clustering structure. \(Q = 0\) means the partition is no better than random. Community detection algorithms like Louvain maximize \(Q\) as their objective, although maximizing modularity is NP-hard and requires heuristics for large networks.

    **Concept Tested:** Modularity (Network), Community Detection

---

#### Question 2

The Girvan-Newman algorithm detects communities by iteratively removing edges with the highest betweenness centrality. What is the intuition behind why high-betweenness edges mark community boundaries?

<div class="upper-alpha" markdown>
1. High-betweenness edges are usually the longest edges in terms of path weight
2. High-betweenness edges connect high-degree hub nodes that define community centers
3. Edges on the boundary between communities lie on many shortest paths between nodes in different communities — removing them disconnects inter-community traffic
4. High-betweenness edges are added last during graph construction, so removing them first reverses the construction process
</div>

??? question "Show Answer"
    The correct answer is **C**. Within a tight community, nodes have many short internal paths. Paths between communities must pass through the few inter-community edges (bridges), making those edges lie on a huge fraction of all shortest paths — giving them high betweenness. Removing the highest-betweenness edge cuts the weakest inter-community link, progressively revealing nested community structure as a dendrogram. The algorithm is exact but \(O(E^2 N)\) — too slow for large networks, motivating faster methods like Louvain.

    **Concept Tested:** Girvan-Newman Algorithm

---

#### Question 3

The Louvain algorithm runs in two phases repeated until convergence. What are the two phases?

<div class="upper-alpha" markdown>
1. Phase 1: compute betweenness centrality; Phase 2: remove high-betweenness edges
2. Phase 1: greedily assign each node to the neighbor's community that maximizes modularity gain; Phase 2: build a "super-node" graph by collapsing each community to a single node
3. Phase 1: spectral clustering to get initial communities; Phase 2: refine boundaries using label propagation
4. Phase 1: find all triangles in the graph; Phase 2: merge triangles with high overlap into communities
</div>

??? question "Show Answer"
    The correct answer is **B**. Louvain (Blondel et al. 2008) alternates two phases. Phase 1 (local move): for each node v, try moving v to each neighbor's community; keep the move if ΔQ > 0, repeat until no improvement. This runs in nearly O(E) for sparse graphs. Phase 2 (aggregation): collapse each community to a super-node, with weighted edges representing inter-community connections. Repeat Phase 1 on the coarsened graph. The nested aggregation is what gives Louvain its hierarchical community structure and logarithmic runtime in practice.

    **Concept Tested:** Louvain Algorithm

---

#### Question 4

BigCLAM models overlapping communities where a node can belong to multiple communities simultaneously. What is the key difference from hard-partition methods like Louvain?

<div class="upper-alpha" markdown>
1. BigCLAM requires labeled nodes while Louvain is fully unsupervised
2. BigCLAM only works on bipartite graphs while Louvain handles any graph type
3. BigCLAM uses spectral methods while Louvain uses greedy optimization
4. BigCLAM assigns each node a non-negative membership strength to each community rather than a single hard assignment — a node can strongly belong to community A and weakly belong to community B simultaneously
</div>

??? question "Show Answer"
    The correct answer is **D**. In BigCLAM, each node \(v\) has a community membership vector \(\mathbf{F}_v \in \mathbb{R}^C\) where \(F_{v,c} \geq 0\) represents \(v\)'s affinity for community \(c\). The probability of an edge \((u,v)\) is \(P(\text{edge} \mid u,v) = 1 - \exp(-\mathbf{F}_u \cdot \mathbf{F}_v)\): higher overlap in membership vectors means higher edge probability. This model explains overlapping structure — a researcher might belong to both "machine learning" and "bioinformatics" communities. Hard-partition methods force every node into exactly one community, losing this overlap information.

    **Concept Tested:** BigCLAM Model, Overlapping Community

---

#### Question 5

In the Louvain algorithm, the modularity gain ΔQ for moving node v from its current community to a neighboring community c is computed efficiently. What makes the greedy node-moving approach tractable for large networks?

<div class="upper-alpha" markdown>
1. ΔQ can be computed in O(1) using precomputed degree sums without recomputing the full modularity each time
2. The algorithm only considers the top-10 highest-degree neighbors, reducing the search space
3. Louvain uses approximate ΔQ estimates via random sampling, trading accuracy for speed
4. ΔQ computation is parallelized across all nodes simultaneously using GPU operations
</div>

??? question "Show Answer"
    The correct answer is **A**. The key efficiency insight is that \(\Delta Q\) for moving \(v\) from community \(C_1\) to \(C_2\) depends only on: the sum of \(v\)'s edge weights to nodes in \(C_2\), the total degree of \(C_2\) (precomputed and updated incrementally), and \(v\)'s degree. All these can be maintained in \(O(\deg(v))\) per move, making Phase 1 run in \(O(E)\) total across all nodes. This incremental update — never recomputing global modularity from scratch — is what makes Louvain scale to billion-edge networks.

    **Concept Tested:** Louvain Algorithm, Modularity (Network)

---

#### Question 6

Fraud detection using graph methods exploits the network structure of fraudulent activity. What is the "guilt-by-association" principle in graph-based fraud detection?

<div class="upper-alpha" markdown>
1. High-degree nodes are more likely to be fraudulent because they interact with many accounts
2. Fraudsters always cluster in the same geographic location, which can be detected from GPS metadata
3. Fraudulent accounts tend to interact primarily with other fraudulent accounts — graph proximity to known fraudsters is a strong signal even without direct evidence
4. Fraud can only be detected in directed graphs where money flow creates asymmetric patterns
</div>

??? question "Show Answer"
    The correct answer is **C**. Fraud networks exhibit homophily: fraudulent accounts send transactions to each other (money mules, fake merchants, money laundering rings), communicate with each other, or were created by the same actors. Even if account A shows no individually suspicious behavior, if A transacts heavily with 3 known fraudsters, A is likely fraudulent too. Label propagation (Chapter 5) and GNNs naturally operationalize this guilt-by-association signal through their neighborhood aggregation mechanism.

    **Concept Tested:** Fraud Detection (Graph), Social Network Analysis

---

#### Question 7

Resolution limit is a known weakness of modularity maximization. What does it mean?

<div class="upper-alpha" markdown>
1. Modularity cannot distinguish communities with different density levels
2. Modularity can only detect communities with more than 100 nodes
3. Modularity fails to detect communities smaller than \(O(\sqrt{E})\) — small communities in large networks merge into larger ones because their internal edges are statistically indistinguishable from random fluctuations at the graph scale
4. Modularity assigns all isolated nodes to the same "noise" community
</div>

??? question "Show Answer"
    The correct answer is **C**. Fortunato & Barthélemy (2007) showed that modularity has a resolution limit: communities smaller than \(O(\sqrt{E})\) may be merged even if they are well-separated, because at the scale of the whole graph, their internal edges look statistically similar to random edges. A clique of 5 nodes in a graph with a million edges has an internal edge density that modularity cannot distinguish from noise. This motivates multi-resolution community detection methods and local community detection algorithms that focus on a query node's neighborhood rather than partitioning the entire graph.

    **Concept Tested:** Modularity (Network)

---

#### Question 8

Social network analysis (SNA) uses structural metrics to understand roles in a network. What does a node with high betweenness centrality but low clustering coefficient typically represent?

<div class="upper-alpha" markdown>
1. A highly popular node in the center of the network with dense local connections
2. A community hub that is connected to every other node in its community
3. An isolated node with few connections and no community membership
4. A bridge or broker node connecting otherwise separate communities — it lies on many shortest paths but its neighbors are not well-connected to each other
</div>

??? question "Show Answer"
    The correct answer is **D**. High betweenness = many shortest paths pass through this node (it is a bottleneck). Low clustering = its neighbors are not directly connected to each other (no triangles). Together these indicate a structural hole broker: a node that bridges two communities whose members don't know each other. Broker nodes are powerful in social networks because they control information flow between communities. In the Karate Club graph, the instructor and president are brokers between the two factions.

    **Concept Tested:** Social Network Analysis, Community Detection

---

#### Question 9

The Louvain algorithm's output is a hierarchical community structure. What is the significance of the hierarchy?

<div class="upper-alpha" markdown>
1. The hierarchy provides multiple scales of community organization simultaneously — fine-grained communities at the bottom and broad super-communities at the top
2. The hierarchy determines the order in which nodes are processed during Phase 1
3. The hierarchy is used only for visualization; the final partition is always the finest level
4. The hierarchy enables Louvain to recover communities that Girvan-Newman cannot find
</div>

??? question "Show Answer"
    The correct answer is **A**. Each aggregation phase in Louvain produces communities at a coarser resolution. The result is a dendrogram of communities: at the finest level (original nodes), communities are small and tightly knit; at the coarsest level (single super-node), all nodes are in one community. Intermediate levels provide communities at different scales of organization — useful for understanding hierarchical structure in biological networks (genes → pathways → biological processes) or social networks (individuals → cliques → organizations → industries).

    **Concept Tested:** Louvain Algorithm, Community Detection

---

#### Question 10

BigCLAM's generative model assumes that edge probability increases with the overlap of community membership vectors. Under this model, what does a node with near-zero membership strength in all communities represent?

<div class="upper-alpha" markdown>
1. A hub node that connects all communities
2. A node that the model failed to process due to numerical issues
3. A node that belongs equally to all communities
4. A peripheral or background node: its edges are mostly random noise, not explained by community structure
</div>

??? question "Show Answer"
    The correct answer is **D**. In BigCLAM, \(P(\text{edge} \mid u,v) = 1 - \exp(-\mathbf{F}_u \cdot \mathbf{F}_v)\). If \(F_{v,c} \approx 0\) for all communities \(c\), then \(\mathbf{F}_v \cdot \mathbf{F}_u \approx 0\) for any \(u\), giving \(P(\text{edge}) \approx 0\). Such nodes have almost no community-driven connectivity — they are structural outsiders or noise nodes that the community model assigns low membership everywhere. This is analogous to "background clutter" in image segmentation. Well-designed community models include an explicit background term to handle these nodes gracefully.

    **Concept Tested:** BigCLAM Model, Overlapping Community
