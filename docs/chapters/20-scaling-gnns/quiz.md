# Quiz: Scaling GNNs to Billion-Node Graphs

Test your understanding of neighbor sampling, cluster-based batching, GraphSAINT, SIGN, and GNN scalability strategies.

---

#### Question 1

The "neighborhood explosion" problem is the central scalability challenge for deep GNNs on large graphs. What causes it?

<div class="upper-alpha" markdown>
1. Deep GNNs require computing the full graph Laplacian, which takes \(O(N^3)\) time
2. The number of trainable parameters grows exponentially with the number of layers
3. GNN training requires loading the full adjacency matrix into GPU memory regardless of batch size
4. The receptive field of each node grows as \(O(d^L)\) — exponential in depth \(L\) — because each layer expands to one more hop and hubs have high degree \(d\)
</div>

??? question "Show Answer"
    The correct answer is **D**. Each GNN layer expands a node's receptive field by one hop. For a 3-layer GNN on a graph with average degree \(d = 10\), each training node requires \(O(d^3) = O(1000)\) nodes in its computational graph. For a hub node with degree 100, this is \(O(10^6)\). Loading all these nodes, their features, and the subgraph adjacency into a mini-batch makes training infeasible for large graphs without sampling. This exponential growth is the neighborhood explosion problem.

    **Concept Tested:** Neighbor Sampling, Graph Sampling Strategy

---

#### Question 2

GraphSAGE's neighbor sampling fixes the neighborhood size to k per layer. What trade-off does choosing k = 5 (vs. k = 25) make?

<div class="upper-alpha" markdown>
1. k = 5 is more accurate but slower; k = 25 is an approximation with higher throughput
2. k = 5 uses less memory and is faster but produces higher-variance gradient estimates; k = 25 reduces variance and is more accurate but uses more memory per batch
3. k = 5 only samples from the first hop; k = 25 samples from all L hops
4. k = 5 applies attention weighting to selected neighbors; k = 25 uses uniform aggregation
</div>

??? question "Show Answer"
    The correct answer is **B**. With k = 5, only 5 of a node's potentially hundreds of neighbors are sampled — the mini-batch is small and fast, but each gradient estimate uses a very noisy neighborhood approximation. With k = 25, the sample better approximates the full neighborhood, producing lower-variance gradients at the cost of 5× more neighborhood nodes to load and process. GraphSAGE's original paper used k = [25, 10] for 2-layer models: 25 neighbors in the outer hop, 10 in the inner hop.

    **Concept Tested:** Neighbor Sampling

---

#### Question 3

Cluster-GCN partitions the graph into clusters and trains on one cluster at a time. What property of graph clusters makes this a good approximation to full-graph GCN?

<div class="upper-alpha" markdown>
1. Clustering removes all hub nodes from training, preventing neighborhood explosion
2. Graph clusters have exactly the same number of nodes, ensuring balanced batches
3. Clusters are always trees, making exact GCN computation \(O(N)\) rather than \(O(N^2)\)
4. High-quality clusters (produced by METIS or Louvain) concentrate most edges within clusters — intra-cluster edges dominate inter-cluster edges, so within-cluster message passing approximates full-graph aggregation well
</div>

??? question "Show Answer"
    The correct answer is **D**. Good graph partitioning algorithms (METIS, Louvain) minimize inter-cluster edges while maximizing intra-cluster density. When a cluster is used as a mini-batch, GCN message passing within the cluster sees almost all the edges that would have been seen in full-graph training for those nodes — the missing edges are the small fraction that cross cluster boundaries. The approximation quality is measured by the "edge cut ratio" (inter-cluster edges / total edges); lower is better.

    **Concept Tested:** Cluster-GCN, Graph Partitioning

---

#### Question 4

GraphSAINT samples subgraphs at the graph level rather than expanding neighborhoods node by node. What is GraphSAINT's key theoretical contribution?

<div class="upper-alpha" markdown>
1. It provides importance-weighted normalization that makes the mini-batch estimator for the loss function unbiased, regardless of the sampling distribution
2. It proves that subgraph sampling is always more accurate than neighbor sampling for L ≥ 3 layers
3. It uses attention to select subgraphs that contain the most informative edges for gradient computation
4. It eliminates all randomness in sampling by using a deterministic round-robin subgraph schedule
</div>

??? question "Show Answer"
    The correct answer is **A**. GraphSAINT samples a connected subgraph G_s and trains GCN on it. Naively, nodes and edges at the boundary are undersampled relative to their frequency in the full graph. GraphSAINT derives normalization coefficients for both node features and edge contributions based on the probability that each node/edge is included in the sampled subgraph. These importance weights make the loss function's expectation equal to the full-graph loss — an unbiased estimator that converges to the same solution regardless of sampling strategy.

    **Concept Tested:** GraphSAINT, Graph Sampling Strategy

---

#### Question 5

SIGN (Scalable Inception Graph Neural Network) avoids stochastic sampling entirely. How does it achieve scalable training?

<div class="upper-alpha" markdown>
1. By using hash functions to aggregate neighborhood features without storing adjacency information
2. By compressing the adjacency matrix into a low-rank approximation before GNN training
3. By pre-computing multi-hop aggregations offline (\(S^k X\) for \(k = 1, \ldots, K\)) before training, so the online training loop involves only a standard MLP with no message passing
4. By replacing GNN layers with Transformer attention applied to a random sample of global node pairs
</div>

??? question "Show Answer"
    The correct answer is **C**. SIGN pre-computes the feature aggregations \(S^k X = (\tilde{D}^{-1/2} \tilde{A}\, \tilde{D}^{-1/2})^k X\) for \(k = 0, 1, \ldots, K\) — each requiring only sparse matrix-vector multiplications. These pre-computed features are stored as augmented node features. During training, the model is just an MLP (or other classifier) applied to the concatenated pre-computed features — no message passing occurs at training time. This decouples the expensive graph aggregation (done once offline) from the training loop, enabling standard mini-batch SGD without any neighborhood sampling.

    **Concept Tested:** SIGN Architecture, Historical Embeddings (SIGN)

---

#### Question 6

LADIES (Layer-Dependent Importance-based Neighbor Sampling) improves over uniform neighbor sampling. What is its key innovation?

<div class="upper-alpha" markdown>
1. LADIES samples only from nodes within 2 hops of the target, ignoring all longer-range connections
2. LADIES samples neighbors proportional to their degree: high-degree nodes are sampled more often
3. LADIES uses different sampling rates per graph layer: more samples in shallow layers, fewer in deep layers
4. LADIES samples the set of neighbors for all nodes in the current layer jointly, using importance sampling weights based on the influence of each candidate node on the current-layer nodes
</div>

??? question "Show Answer"
    The correct answer is **D**. Standard neighbor sampling (GraphSAGE) independently samples k neighbors per node, leading to redundancy: the same popular node may be sampled multiple times across different target nodes in the mini-batch. LADIES samples a fixed budget of \(n_l\) nodes for each layer \(l\) jointly, with importance weights proportional to their squared adjacency values to current-layer nodes. This importance-weighted layer-wise sampling reduces redundancy and ensures the most influential nodes in each layer are included, lowering variance at fixed computational cost.

    **Concept Tested:** LADIES Sampler, Layer-Wise Sampling

---

#### Question 7

Graph partitioning using METIS is a pre-processing step for Cluster-GCN. What does METIS optimize and why is recursive bisection efficient?

<div class="upper-alpha" markdown>
1. METIS minimizes the number of nodes per partition to ensure balanced memory usage
2. METIS minimizes the edge cut (number of inter-partition edges) subject to balanced partition sizes using recursive bisection — splitting the graph in half repeatedly until the desired number of partitions is reached
3. METIS maximizes intra-partition modularity using the same objective as Louvain
4. METIS uses spectral clustering to find partitions aligned with the Laplacian eigenvectors
</div>

??? question "Show Answer"
    The correct answer is **B**. METIS partitions a graph G into k balanced parts while minimizing the number of edges crossing partition boundaries (the edge cut). It uses a multi-level approach: coarsen the graph by collapsing edges, partition the coarsened graph (easy because it's small), then project and refine the partition back to the original graph. Recursive bisection splits the graph into 2 parts, then splits each part into 2, and so on until k partitions are formed. This multi-level strategy is near-linear in the number of edges, making METIS practical even for billion-edge graphs.

    **Concept Tested:** Graph Partitioning, Cluster-GCN

---

#### Question 8

Historical embeddings (used in systems like VR-GCN) reduce variance in neighbor sampling by caching intermediate representations. What is the key challenge with this approach?

<div class="upper-alpha" markdown>
1. Cached embeddings require storing \(O(N \times d)\) values which is too large for billion-node graphs
2. Historical embeddings cannot be used with momentum-based optimizers like Adam
3. Cached embeddings become stale as parameters update during training — the historical embedding of a node was computed with old parameters, introducing a bias-variance tradeoff
4. Caching requires the graph to be static, preventing online learning on streaming graphs
</div>

??? question "Show Answer"
    The correct answer is **C**. Historical embeddings store the last computed representation of each node and reuse it when the node appears as a neighbor in subsequent mini-batches. This eliminates the need to recompute deep neighborhoods (reducing variance), but introduces staleness: the cached embedding was computed using parameters \(\theta_{t-k}\) from \(k\) steps ago, not the current \(\theta_t\). As learning rate is high and parameters change rapidly, stale embeddings introduce bias. Methods like VR-GCN use control variates to correct for staleness, trading off accuracy vs. computational savings.

    **Concept Tested:** Historical Embeddings (SIGN)

---

#### Question 9

Why is SIGN's pre-computation approach limited to fixed propagation matrices like \(\tilde{D}^{-1/2} \tilde{A}\, \tilde{D}^{-1/2}\)? What types of GNN architectures cannot benefit from SIGN-style pre-computation?

<div class="upper-alpha" markdown>
1. Pre-computation requires the aggregation to be independent of the node's own learned representation — adaptive architectures like GAT (where attention weights depend on learned features) cannot be pre-computed because the aggregation depends on the result being learned
2. SIGN requires symmetric matrices, so directed graphs cannot use pre-computation
3. Pre-computation is limited to graphs with fewer than 10 million edges due to memory constraints
4. SIGN cannot pre-compute more than 3 hops without numerical underflow in the propagation matrix
</div>

??? question "Show Answer"
    The correct answer is **A**. SIGN's pre-computation works because \(S^k X = \hat{S}^k X\) is a fixed linear operator applied to fixed features \(X\) — it doesn't depend on learned parameters. GAT computes attention \(\alpha_{vu} = \text{softmax}(\text{LeakyReLU}(\mathbf{a}^\top[W\mathbf{h}_v \| W\mathbf{h}_u]))\) which depends on the current learned \(W\) — different parameter values give different attention weights and therefore different aggregations. Since the aggregation function changes during training, it cannot be pre-computed. Only architectures with fixed, parameter-independent aggregation (like GCN, SGC) benefit from pre-computation.

    **Concept Tested:** SIGN Architecture, Layer-Wise Sampling

---

#### Question 10

For training on the ogbn-arxiv benchmark (170K nodes, 1.1M edges), which scaling approach requires the least GPU memory during training?

<div class="upper-alpha" markdown>
1. SIGN with pre-computed 3-hop features — training is a standard MLP mini-batch with no adjacency matrix needed in the GPU training loop
2. Full-batch GCN — all nodes are processed simultaneously for efficient GPU utilization
3. Neighbor sampling with k=25 — the large sample size ensures accurate gradient estimates
4. Cluster-GCN with 1500 clusters — each mini-batch loads one cluster with ~113 nodes
</div>

??? question "Show Answer"
    The correct answer is **A**. SIGN pre-computes \(S^0 X, S^1 X, S^2 X, S^3 X\) offline (requiring only sparse-dense matrix products and CPU/disk storage). During GPU training, the input to the MLP is the pre-computed feature matrix stored as a standard tensor — no adjacency matrix, no neighborhood sampling, no message passing. GPU memory is \(O(\text{batch-size} \times K \times d)\) — independent of graph size. Full-batch GCN requires the full \(N \times d\) features plus the adjacency on GPU, while Cluster-GCN still needs in-cluster adjacency. SIGN has the smallest GPU footprint during training.

    **Concept Tested:** SIGN Architecture, Graph Sampling Strategy
