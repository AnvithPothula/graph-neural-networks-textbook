# Quiz: Node Embeddings: DeepWalk and node2vec

Test your understanding of shallow node embeddings, the encoder-decoder framework, and biased random walks.

---

#### Question 1

In the encoder-decoder framework for node embeddings, what do the encoder and decoder represent?

<div class="upper-alpha" markdown>
1. The encoder compresses the adjacency matrix into a sparse format; the decoder reconstructs it exactly
2. The encoder learns edge weights from node features; the decoder assigns community labels
3. The encoder applies PCA to node features; the decoder reverses the transformation
4. The encoder maps a node to a low-dimensional vector; the decoder scores how well pairs of vectors predict graph structure
</div>

??? question "Show Answer"
    The correct answer is **D**. In the encoder-decoder view, \(\text{ENC}(v) = \mathbf{z}_v\) maps each node to a \(d\)-dimensional vector. The decoder \(\text{DEC}(\mathbf{z}_u, \mathbf{z}_v)\) predicts some graph-structural relationship — typically co-occurrence in random walks or edge existence. Training minimizes reconstruction loss, pushing embeddings of similar nodes together and dissimilar nodes apart in embedding space.

    **Concept Tested:** Encoder-Decoder Framework

---

#### Question 2

DeepWalk applies the Skip-Gram model to random walk sequences. What does Skip-Gram learn to predict?

<div class="upper-alpha" markdown>
1. The probability that a context node co-occurs within a fixed window of the center node in random walks
2. The degree sequence of the entire graph given a single node's embedding
3. The shortest path distance between nodes from their embedding vectors
4. The cluster assignment of each node based on the angle between embedding vectors
</div>

??? question "Show Answer"
    The correct answer is **A**. Skip-Gram treats random walks as "sentences" and nodes as "words." For each center node \(v\) and context node \(u\) within window radius \(w\), the model maximizes \(\log P(u \mid v) = \log \sigma(\mathbf{z}_u \cdot \mathbf{z}_v)\). Nodes co-occurring frequently in walks end up with high dot-product similarity. The resulting embeddings capture community structure and proximity automatically from walk statistics alone.

    **Concept Tested:** DeepWalk, Skip-Gram Model

---

#### Question 3

The node2vec hyperparameter q controls a biased random walk. What structural property does q < 1 encourage?

<div class="upper-alpha" markdown>
1. Returning immediately to the previous node, producing short local walks
2. Moving outward and exploring diverse parts of the graph (DFS-like), capturing global structural roles
3. Staying confined to the immediate 1-hop neighborhood
4. Moving only along the highest-degree nodes in the graph
</div>

??? question "Show Answer"
    The correct answer is **B**. When q < 1, the walk is biased toward nodes farther from the previous position (DFS-like exploration). This samples globally diverse neighborhoods, making co-occurring nodes structurally equivalent regardless of where they sit in the graph — useful for role-based tasks like detecting all "bridge nodes." High q > 1 produces BFS-like walks that stay in the local neighborhood, emphasizing community membership instead.

    **Concept Tested:** DFS Strategy (node2vec), Biased Random Walk

---

#### Question 4

Shallow embeddings like DeepWalk and node2vec are transductive. What key limitation does this impose?

<div class="upper-alpha" markdown>
1. They cannot generate embeddings for new nodes added after training without re-running the full optimization
2. They cannot model weighted edges or multi-relational graphs
3. They can only be applied to graphs with fewer than 10,000 nodes
4. They require labeled data to train, making them supervised methods
</div>

??? question "Show Answer"
    The correct answer is **A**. Shallow methods maintain a separate embedding vector per node, learned during training on the fixed graph. When a new node arrives, it has no corresponding vector — the model must be fully retrained on the expanded graph to embed it. This transductive limitation is a major motivation for GNNs: an inductive GNN uses a shared aggregation function that can embed unseen nodes by aggregating their features and neighborhood, without any retraining.

    **Concept Tested:** Transductive Learning, Inductive Learning

---

#### Question 5

Negative sampling in Skip-Gram training is used because the full softmax over all N nodes is too expensive. What does negative sampling approximate?

<div class="upper-alpha" markdown>
1. The eigenvalue decomposition of the graph Laplacian using sampled eigenvectors
2. The softmax normalization constant by replacing it with a few randomly sampled "negative" (non-co-occurring) node pairs
3. The random walk transition probabilities by subsampling edges proportional to weight
4. The batch gradient by averaging gradients over a random subset of training examples
</div>

??? question "Show Answer"
    The correct answer is **B**. The full Skip-Gram objective requires computing \(\exp(\mathbf{z}_u \cdot \mathbf{z}_v)\) for all \(N\) nodes in the denominator — \(O(N)\) per update. Negative sampling replaces this with a binary classification: the model must distinguish true co-occurring pairs from \(k\) randomly sampled "negative" pairs. The training objective becomes maximizing \(\log \sigma(\mathbf{z}_u \cdot \mathbf{z}_v) + \sum_i \log \sigma(-\mathbf{z}_{n_i} \cdot \mathbf{z}_v)\), making each update \(O(k) \ll O(N)\).

    **Concept Tested:** Negative Sampling

---

#### Question 6

Matrix factorization for graph embeddings expresses node embeddings as the factorization of which matrix?

<div class="upper-alpha" markdown>
1. The degree matrix D raised to the power −1/2
2. A similarity or proximity matrix \(S\) (such as the adjacency matrix or Katz similarity matrix) into \(Z \cdot Z^\top\)
3. The graph Laplacian L = D − A
4. The node feature matrix X concatenated with the edge list
</div>

??? question "Show Answer"
    The correct answer is **B**. Matrix factorization methods approximate a node similarity matrix \(S \approx Z Z^\top\), where \(Z \in \mathbb{R}^{N \times d}\) is the embedding matrix. Different choices of \(S\) yield different methods: \(S = A\) gives adjacency factorization, \(S = \text{Katz}\) gives Katz-based embeddings. DeepWalk was shown to be equivalent to factorizing a matrix of pointwise mutual information (PMI) computed from random walk statistics, establishing the deep connection between random-walk methods and linear algebra.

    **Concept Tested:** Matrix Factorization (Graph)

---

#### Question 7

Homophily and structural equivalence are two different properties that node embeddings can capture. What is the distinction?

<div class="upper-alpha" markdown>
1. Homophily captures closeness in the graph (nearby nodes get similar embeddings); structural equivalence captures similar structural roles (e.g., both are hub centers) regardless of proximity
2. Homophily means two nodes share a feature label; structural equivalence means they share the same embedding dimension
3. Homophily applies to undirected graphs and structural equivalence to directed graphs only
4. Homophily requires supervised labels while structural equivalence is fully unsupervised
</div>

??? question "Show Answer"
    The correct answer is **A**. Homophily means connected or nearby nodes tend to be similar (e.g., friends share interests) — BFS-biased walks capture this. Structural equivalence means nodes playing the same structural role are similar (e.g., two hub nodes at opposite ends of the graph) — DFS-biased walks capture this. node2vec uses p and q to interpolate between these two regimes, making it flexible for diverse downstream tasks.

    **Concept Tested:** Homophily, Structural Equivalence

---

#### Question 8

The node2vec hyperparameter p controls the return probability. When p > 1, the walk is less likely to return to the previous node. What does p < 1 encourage?

<div class="upper-alpha" markdown>
1. The walk explores new nodes farther from its starting point
2. The walk terminates earlier, reducing the total walk length
3. The walk tends to revisit the previous node, producing locally focused BFS-like exploration
4. The walk ignores edge weights and samples neighbors uniformly
</div>

??? question "Show Answer"
    The correct answer is **C**. When p < 1, the unnormalized transition weight back to the previous node t is 1/p > 1 (relatively high), making return transitions more likely. This keeps the walk confined near its recent positions, producing locally-dense, BFS-like sampling that emphasizes the immediate neighborhood. Combined with high q, this creates embeddings that strongly capture community membership and homophily — similar nodes in the same cluster get similar vectors.

    **Concept Tested:** BFS Strategy (node2vec), Biased Random Walk

---

#### Question 9

In the encoder-decoder framework, what is the key difference between a shallow embedding method and a GNN-based encoder?

<div class="upper-alpha" markdown>
1. Shallow methods use continuous embeddings; GNN encoders use discrete one-hot representations
2. Shallow methods minimize cross-entropy loss; GNN encoders always use mean squared error
3. Shallow methods are only for node tasks; GNN encoders only support graph-level tasks
4. Shallow methods have one embedding vector per node with no parameter sharing; GNN encoders use shared aggregation parameters that can generalize to unseen nodes
</div>

??? question "Show Answer"
    The correct answer is **D**. A shallow encoder is simply a lookup table: \(\text{ENC}(v) = Z \mathbf{e}_v\) (the \(v\)-th column of a trainable matrix). It stores \(O(N \times d)\) parameters with no sharing across nodes. A GNN encoder computes \(\mathbf{z}_v = \text{GNN}(\mathbf{x}_v, \text{neighborhood})\) using shared weight matrices — the same parameters produce embeddings for all nodes, including new ones at test time. This parameter sharing is the key to inductive generalization.

    **Concept Tested:** Shallow Embedding, Inductive Learning

---

#### Question 10

LINE (Large-scale Information Network Embedding) differs from DeepWalk by explicitly modeling both first-order and second-order proximity. What do these two proximities capture?

<div class="upper-alpha" markdown>
1. First-order proximity captures self-loops; second-order proximity captures edges to non-neighbor nodes
2. First-order proximity applies to directed graphs; second-order proximity applies to undirected graphs only
3. First-order proximity captures direct edge connections between nodes; second-order proximity captures similarity via shared neighborhoods
4. First-order proximity uses the degree matrix; second-order proximity uses the Laplacian
</div>

??? question "Show Answer"
    The correct answer is **C**. LINE's first-order proximity preserves direct connections: nodes linked by an edge should have similar embeddings. Second-order proximity preserves shared neighborhood structure: two nodes that share many neighbors (even without a direct edge) should also be similar, since they occupy analogous positions in the graph. Combining both orders captures both local connectivity and broader structural similarity, making LINE particularly effective for large social and information networks.

    **Concept Tested:** LINE Embedding
