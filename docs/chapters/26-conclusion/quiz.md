# Quiz: Conclusion — The GNN Design Space and Open Problems

A synthesis quiz spanning all six parts of the textbook. Test your ability to connect concepts across chapters and reason about the GNN design space as a whole.

---

#### Question 1

The GNN design space treats every architectural choice as an independent dimension. Which combination of design choices defines the minimal "standard GCN" configuration?

<div class="upper-alpha" markdown>
1. Mean aggregation + no activation + residual connections + inductive training + attention weights
2. Max aggregation + sigmoid activation + jumping knowledge connections + mini-batch training + self-loops
3. Sum aggregation + ReLU activation + no skip connections + transductive training + degree-normalized adjacency
4. Attention aggregation + tanh activation + DiffPool readout + spectral initialization + edge dropout
</div>

??? question "Show Answer"
    The correct answer is **C**. Standard GCN (Kipf & Welling, 2017) uses: (1) sum aggregation with symmetric degree normalization \(\tilde{D}^{-1/2} \tilde{A}\, \tilde{D}^{-1/2}\); (2) ReLU activation between layers; (3) no skip connections (pure propagation); (4) transductive training on the full graph; (5) renormalization trick (\(\tilde{A} = A + I\) adds self-loops). Each of these is the "default" setting in the GNN design space. Modifying any dimension — switching to GAT attention, adding residuals, enabling inductive mini-batch training — moves to a different point in the design space.

    **Concept Tested:** Graph Convolutional Network (Ch06), GNN Expressiveness (Ch09)

---

#### Question 2

Looking across the textbook, which two chapters address the same fundamental tension — the trade-off between expressiveness and computational cost — at different scales?

<div class="upper-alpha" markdown>
1. Chapter 9 (GNN Theory: WL Expressiveness) and Chapter 20 (Scaling GNNs)
2. Chapter 3 (PageRank) and Chapter 16 (Recommender Systems)
3. Chapter 12 (KG Embeddings) and Chapter 22 (Temporal Graphs)
4. Chapter 5 (Label Propagation) and Chapter 14 (KG Foundation Models)
</div>

??? question "Show Answer"
    The correct answer is **A**. Chapter 9 shows that more expressive GNNs (k-WL, subgraph GNNs) require exponentially more computation to distinguish finer structural patterns. Chapter 20 shows that more accurate training (full-batch GCN) requires exponentially more memory as the receptive field grows, while cheaper sampling reduces accuracy. Both chapters define the same expressiveness-vs-cost frontier at different levels of abstraction: Chapter 9 at the architecture level, Chapter 20 at the training algorithm level. Understanding both curves is essential for principled GNN design.

    **Concept Tested:** GNN Expressiveness (Ch09), Scaling GNNs (Ch20)

---

#### Question 3

The textbook's running example of ogbn-arxiv spans Chapters 6–20. Why was this specific dataset chosen as the benchmark for Part 2–5 content rather than a smaller graph?

<div class="upper-alpha" markdown>
1. With 170K nodes and 1.1M edges, ogbn-arxiv is large enough to reveal scaling challenges (Chapter 20), is a standard OGB benchmark enabling fair comparison, has rich text features enabling LLM integration (Chapter 23), and is small enough to train on a single GPU in reasonable time
2. ogbn-arxiv is the only graph dataset with labeled nodes, making supervised evaluation possible
3. ogbn-arxiv has a known community structure that was verified by human experts, making it ideal for community detection benchmarks
4. ogbn-arxiv is the only graph where GIN outperforms GCN by more than 5% on the OGB leaderboard
</div>

??? question "Show Answer"
    The correct answer is **A**. ogbn-arxiv was chosen as the primary running example because it satisfies multiple criteria simultaneously: (1) large enough to motivate neighbor sampling, cluster-GCN, and SIGN (Chapter 20); (2) OGB-standardized with fair train/val/test splits enabling reproducible comparisons; (3) text-attributed (each node has a paper abstract) enabling the LLM+GNN content in Chapter 23; (4) trainable on a single GPU within minutes. The Karate Club graph (Chapters 1–9) complements it as a small visualization-friendly example for conceptual explanation.

    **Concept Tested:** Graph Neural Network (Ch06), Scaling GNNs (Ch20)

---

#### Question 4

Across the textbook, three chapters derive methods from random walk theory: Chapter 3 (PageRank), Chapter 4 (node2vec), and Chapter 10 (APPNP). What unifying insight connects all three?

<div class="upper-alpha" markdown>
1. All three compute the leading eigenvector of some graph matrix, connecting random walks to spectral analysis
2. All three define node importance or node proximity as the stationary distribution of a carefully designed random walk on the graph, differing in how the walk is biased and what quantity is used downstream
3. All three methods require computing the full adjacency matrix, limiting them to small graphs
4. All three methods produce identical results on regular graphs where all nodes have the same degree
</div>

??? question "Show Answer"
    The correct answer is **B**. The unifying insight: a random walk's stationary distribution encodes structural proximity and importance. PageRank computes the stationary distribution of a teleporting random walk, ranking pages by long-run visit probability. node2vec biases the walk to interpolate between BFS (community) and DFS (structural role), using visit co-occurrence as a proximity measure. APPNP uses the PPR stationary distribution as the propagation weights for feature diffusion, with teleportation providing a residual connection. All three are "random walk meets learning objective" combinations.

    **Concept Tested:** PageRank (Ch03), node2vec (Ch04), APPNP (Ch10)

---

#### Question 5

The six-chapter structure of Part 3 (Knowledge Graphs) maps onto a progression of query complexity. What is the ordering from simplest to most complex KG query type?

<div class="upper-alpha" markdown>
1. Matrix factorization → spectral decomposition → tensor factorization → neural decomposition
2. Community detection → link prediction → entity alignment → knowledge base completion
3. Entity lookup → relation type inference → schema construction → ontology reasoning
4. Triple scoring → multi-hop chains → conjunctive queries → inductive generalization
</div>

??? question "Show Answer"
    The correct answer is **D**. The KG progression: (1) **Triple scoring** (Chapter 12): given (h, r, ?), score all possible tails — a 1-hop lookup; (2) **Multi-hop chains** (Chapter 13, path queries): follow a chain of k relations from an anchor; (3) **Conjunctive queries** (Chapter 13, 2i/3i): AND multiple chain constraints from different anchors; (4) **Inductive generalization** (Chapter 14): transfer reasoning to new entities and new KGs via structural representations that don't require entity-specific embeddings. Each level requires strictly more expressive machinery than the one before.

    **Concept Tested:** KG Completion (Ch12), Multi-Hop Query (Ch13), Inductive KG Reasoning (Ch14)

---

#### Question 6

Over-smoothing (Chapter 9) and over-squashing (Chapter 9) both limit deep GNNs, but for different reasons. Which mitigation strategy addresses both problems simultaneously?

<div class="upper-alpha" markdown>
1. Increasing the embedding dimension from 64 to 512 dimensions provides more capacity to prevent both phenomena
2. Reducing the number of message passing layers from 5 to 2 eliminates both over-smoothing and over-squashing
3. Adding graph rewiring (extra edges between distant nodes) helps over-squashing, while skip connections help over-smoothing — using both together addresses both limitations
4. Applying batch normalization prevents both by normalizing representations at each layer
</div>

??? question "Show Answer"
    The correct answer is **C**. Over-smoothing and over-squashing have different causes requiring different fixes. Over-smoothing: too many layers cause all representations to converge to a similar vector → fix with skip/residual connections (preserve the original representation) and PairNorm. Over-squashing: bottleneck edges force exponentially many nodes' information through a narrow cut → fix with graph rewiring (adding edges from bottleneck nodes to increase graph expansion/Cheeger constant). Since both co-occur in deep GNNs, using both mitigation techniques together (rewiring + skip connections) is the most robust approach.

    **Concept Tested:** Over-Squashing (Ch09), Skip Connection (Ch07), Cheeger Constant (Ch09)

---

#### Question 7

Chapters 8 (training) and 20 (scaling) both discuss mini-batch training, but from different perspectives. What is the key distinction between "neighbor sampling variance" (Chapter 8) and "neighborhood explosion" (Chapter 20)?

<div class="upper-alpha" markdown>
1. Neighbor sampling variance is a training problem; neighborhood explosion is an inference problem
2. Neighbor sampling variance refers to the statistical noise in gradient estimates from sampling a subset of neighbors; neighborhood explosion refers to the exponential growth of the computational graph size that makes full-batch training infeasible — sampling solves explosion at the cost of introducing variance
3. Neighbor sampling variance only affects shallow GNNs; neighborhood explosion only affects deep GNNs
4. They are the same problem described from different perspectives in different chapters
</div>

??? question "Show Answer"
    The correct answer is **B**. These are two sides of the same design decision. Neighborhood explosion (Ch20): without sampling, a 3-layer GNN on a 10-degree graph requires \(O(10^3) = 1000\) nodes per training example, making large-graph training infeasible. Neighbor sampling solves explosion by capping the computational graph size. But sampling introduces variance: \(k=5\) samples from 100 neighbors is a noisy estimate of the true gradient. These two perspectives frame the tradeoff from different angles: one as an accuracy concern, the other as a scalability engineering challenge.

    **Concept Tested:** Neighbor Sampling (Ch08/Ch20), Graph Sampling Strategy (Ch20)

---

#### Question 8

The textbook covers both "transductive" and "inductive" learning settings in multiple chapters. In which setting is a GNN more likely to be deployed in a real production system, and why?

<div class="upper-alpha" markdown>
1. Transductive, because production systems have fixed graphs that never change
2. Neither — production systems use shallow embeddings because GNNs are too slow for real-time inference
3. Both equally — production systems are evenly split between static and dynamic graphs
4. Inductive, because production systems continuously add new users, items, documents, or entities that must be embedded without retraining the entire model
</div>

??? question "Show Answer"
    The correct answer is **D**. Real production systems are almost always inductive: new users sign up (Pinterest, LinkedIn), new papers are published (semantic scholar), new transactions occur (fraud detection), new molecules are synthesized (drug discovery). A transductive model trained on the current graph must be fully retrained to embed any new entity — impractical at scale. Inductive methods (GraphSAGE, GNNs with node features) generalize to new nodes by running the shared aggregation function on their neighborhoods. This is why PinSage, APPNP, and graph foundation models all use inductive architectures.

    **Concept Tested:** Inductive Learning (Ch04), GraphSAGE (Ch07), Transfer Learning (Ch14)

---

#### Question 9

Identify the single most important "bridge concept" that connects the classical graph methods (Part 1) to the modern GNN methods (Part 2). Which concept from Part 1 directly foreshadows the GNN message passing framework of Part 2?

<div class="upper-alpha" markdown>
1. The WL color refinement kernel (Chapter 2), because its iterative hash-based neighborhood aggregation is structurally identical to GNN message passing, directly motivating the WL-GNN connection in Chapter 9
2. PageRank (Chapter 3), because its iterative rank propagation is the direct precursor of GNN message passing — both propagate information from neighbors iteratively, with learnable weights replacing the fixed damping factor
3. DeepWalk (Chapter 4), because its random walk sampling generates the training data for the first GNN layers
4. Label propagation (Chapter 5), because its iterative label spreading is the exact mathematical limit of a GNN with infinite layers
</div>

??? question "Show Answer"
    The correct answer is **A**. The WL color refinement algorithm (Chapter 2) iteratively assigns each node a new color hash(current_color, multiset_of_neighbor_colors) — this is exactly what a GNN message passing layer computes (aggregate neighbor representations, hash/MLP to new representation). Chapter 9 formalizes this connection rigorously: standard MPNNs are upper-bounded by 1-WL, and GIN achieves this bound by matching the WL algorithm's injectivity property. This makes WL the bridge concept that connects the classical feature engineering perspective of Chapter 2 to the deep learning perspective of Chapters 6–10.

    **Concept Tested:** Weisfeiler-Lehman Kernel (Ch02), GNN Expressiveness (Ch09)

---

#### Question 10

Looking at the full arc of the textbook, what single prediction can be made about the next 5 years of graph machine learning research, based on the trajectories established across the 26 chapters?

<div class="upper-alpha" markdown>
1. GNNs will be replaced by pure attention models (Transformers) that eliminate message passing entirely
2. Classical graph algorithms (PageRank, community detection) will prove sufficient for all practical applications, making GNNs obsolete
3. Graph foundation models that combine LLM-scale pre-training with GNN structural reasoning will emerge as the dominant paradigm, enabling zero-shot transfer across graph domains — the KG work (Chapters 12–14), LLM+GNN integration (Chapter 23), and in-context learning (Chapter 24) all point toward this convergence
4. The field will converge on a single architecture (GPS or similar) that is universally optimal for all graph ML tasks
</div>

??? question "Show Answer"
    The correct answer is **C**. Multiple chapters point to the same convergence: ULTRA (Ch14) showed KGs can be reasoned over inductively with zero-shot transfer; OFA (Ch23) showed cross-domain node classification using LLM-unified feature spaces; PRODIGY (Ch24) showed in-context learning for graphs. The trajectory is clear — graph ML is moving toward foundation models that pre-train on large diverse graph corpora and transfer to new tasks and domains at inference time, combining LLMs' language understanding with GNNs' structural reasoning. The key remaining challenges (identified in this conclusion chapter) are cross-domain feature alignment, inductive structural encoding, and scale.

    **Concept Tested:** Graph Foundation Model (Ch23), In-Context Learning (Ch24), Transfer Learning (Ch14)
