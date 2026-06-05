# Quiz: Designing Powerful Encoders: GIN and Beyond

Test your understanding of GIN, SGC, APPNP, PNA, and architectures that go beyond the 1-WL expressiveness bound.

---

#### Question 1

Simple Graph Convolution (SGC) removes all non-linearities from GCN and collapses L layers into a single matrix. What is the resulting operation?

<div class="upper-alpha" markdown>
1. An attention-weighted aggregation over 1-hop neighbors
2. A single linear transformation of the \(L\)-th power of the normalized adjacency matrix applied to node features: \(\hat{Y} = \text{softmax}(S^L X W)\)
3. A spectral convolution using the \(L\) largest eigenvectors of the graph Laplacian
4. A random walk that aggregates features from \(L\) randomly sampled nodes
</div>

??? question "Show Answer"
    The correct answer is **B**. SGC replaces GCN's \(L\) layers (each with a non-linearity) with a single step: first compute \(S^L X\) (propagating features \(L\) hops via repeated multiplication by the normalized adjacency), then apply a single linear classifier \(W\). This reduces training to a standard logistic regression problem on pre-computed smoothed features. Wu et al. (2019) showed SGC matches GCN performance on many node classification benchmarks at a fraction of the computational cost.

    **Concept Tested:** Simple Graph Convolution (SGC)

---

#### Question 2

APPNP separates feature transformation from propagation. What specific advantage does this provide over standard GCN?

<div class="upper-alpha" markdown>
1. APPNP can process directed graphs while GCN cannot
2. APPNP eliminates the need for node features by using only edge weights for classification
3. APPNP achieves higher-than-1-WL expressiveness by using second-order neighborhood statistics
4. APPNP allows deeper propagation (more hops) without over-smoothing, because it uses Personalized PageRank propagation with a teleportation term that acts as a residual connection
</div>

??? question "Show Answer"
    The correct answer is **D**. APPNP first transforms features with a standard MLP: \(H = \text{MLP}(X)\), then propagates with \(k\) steps of PPR-like diffusion: \(H^{(t+1)} = (1-\alpha)\hat{S} H^{(t)} + \alpha H\), where \(\alpha\) is the teleportation probability and \(\hat{S}\) is the normalized adjacency. The teleportation term \(\alpha H\) injects the original (pre-propagation) features at every step, preventing over-smoothing. This allows APPNP to use 10+ propagation steps — far more than GCN can use — without representations collapsing.

    **Concept Tested:** APPNP

---

#### Question 3

Principal Neighbourhood Aggregation (PNA) uses multiple aggregators and degree scalers together. What problem does combining multiple aggregators solve?

<div class="upper-alpha" markdown>
1. It reduces training time by parallelizing aggregation across multiple GPUs
2. It converts the GNN to a permutation-sensitive model, improving expressiveness beyond 1-WL
3. Different aggregators (sum, mean, max, standard deviation) capture different statistical moments of the neighborhood, and no single aggregator captures all relevant structural information
4. It prevents over-smoothing by applying each aggregator to a random subset of neighbors
</div>

??? question "Show Answer"
    The correct answer is **C**. PNA aggregates with multiple functions simultaneously — sum (count-sensitive), mean (scale-invariant), max (peak-sensitive), and standard deviation (spread-sensitive) — and scales each by degree-dependent scalers. Each aggregator captures a different statistical property of the neighborhood distribution. Concatenating these statistics gives a richer neighborhood summary than any single aggregator alone, empirically matching or exceeding GIN on many molecular graph benchmarks.

    **Concept Tested:** Principal Neighbourhood Aggregation (PNA)

---

#### Question 4

ID-GNN (Identity-Aware GNN) extends message passing by injecting node identity information. What specific limitation of standard GNNs does this overcome?

<div class="upper-alpha" markdown>
1. Standard GNNs cannot process graphs with more than 10,000 nodes
2. Standard GNNs require labeled nodes for all training examples
3. Standard GNNs cannot represent edge-level features
4. Standard GNNs are permutation equivariant but lack cycle detection — they cannot distinguish whether a center node is part of a cycle from its own position, not just from neighbors' views
</div>

??? question "Show Answer"
    The correct answer is **D**. Standard MPNNs are permutation equivariant: they assign the same representation to structurally equivalent nodes. ID-GNN colors each node's ego subgraph with a unique identity marker (indicating "this is the center node") before running message passing. This lets the GNN compute different representations for the center node vs. its neighbors, enabling cycle detection and counting. ID-GNN strictly expands the class of graphs that can be distinguished beyond 1-WL.

    **Concept Tested:** Identity-Aware GNN (ID-GNN)

---

#### Question 5

Subgraph GNNs (like ESAN) achieve higher expressiveness than 1-WL by operating on collections of subgraphs. What is the core idea?

<div class="upper-alpha" markdown>
1. For each node v, they extract a subgraph (e.g., the ego network or node-deleted subgraph) and run a base GNN on the whole collection, then aggregate across subgraphs
2. They partition the graph into k equal-sized subgraphs and apply independent GNNs to each
3. They randomly sample subgraphs during training and apply 1-WL testing on each sample
4. They replace the graph's nodes with subgraphs, treating each subgraph as a "super-node"
</div>

??? question "Show Answer"
    The correct answer is **A**. ESAN (Equivariant Subgraph Aggregation Networks) decomposes a graph G into a "bag" of subgraphs — one per node (node-deleted or ego subgraph) — then runs a standard GNN over each subgraph and aggregates the resulting representations. Because each subgraph has a different structure (the deleted/center node breaks symmetry), the model can distinguish graphs that fool 1-WL on the full graph. The cost is proportional to N times the base GNN cost, which is tractable for small-to-medium graphs.

    **Concept Tested:** Subgraph GNN (ESAN / DS-GNN)

---

#### Question 6

Stochastic Depth for GNNs randomly drops entire layers during training. How does this differ from standard Dropout?

<div class="upper-alpha" markdown>
1. Stochastic Depth applies only to edge features; Dropout applies only to node features
2. Stochastic Depth drops entire graph layers (skipping all aggregation in that layer for the whole batch) while Dropout randomly zeroes individual feature dimensions for individual nodes
3. Stochastic Depth removes random nodes from the graph; Dropout removes random edges
4. Stochastic Depth is applied only during inference; Dropout is applied only during training
</div>

??? question "Show Answer"
    The correct answer is **B**. Standard Dropout randomly zeroes individual activations within a layer — a node's feature dimensions are independently masked. Stochastic Depth randomly skips an entire GNN layer with some probability during training, effectively training an ensemble of GNNs with different depths. At inference, the expected output (averaging over all depth configurations) is used. For deep GNNs, stochastic depth acts as a regularizer and can reduce over-smoothing by sometimes propagating through fewer hops.

    **Concept Tested:** Stochastic Depth (GNN)

---

#### Question 7

k-GNN operates on k-tuples of nodes instead of individual nodes. A 2-GNN assigns representations to pairs (u, v). What makes 2-GNN strictly more expressive than 1-WL?

<div class="upper-alpha" markdown>
1. 2-GNN can process both directed and undirected edges simultaneously
2. 2-GNN can directly detect and count triangles and cycles, which are invisible to 1-WL on regular graphs
3. 2-GNN uses attention weights that 1-WL does not, enabling soft neighborhood selection
4. 2-GNN can distinguish graphs of different sizes, while 1-WL is size-agnostic
</div>

??? question "Show Answer"
    The correct answer is **B**. 1-WL on a k-regular graph assigns identical colors to all nodes and fails to detect any structural differences. 2-GNN operates on node pairs, tracking the shared neighborhood of pair (u, v). This allows it to count common neighbors, detect triangles (3-cliques), and identify cycle structure — information that is structurally invisible to single-node message passing. It has been proven that 2-WL/2-GNN is strictly more powerful than 1-WL.

    **Concept Tested:** k-GNN and δ-k-GNN, GNN Expressiveness

---

#### Question 8

The Jumping Knowledge Network (JK-Net) aggregates representations across all layers. When is the "max" jump aggregation preferred over "concatenate"?

<div class="upper-alpha" markdown>
1. Max aggregation produces a fixed-size output regardless of the number of layers, making it preferable when memory is constrained and the number of layers is large
2. Max aggregation is preferred when graphs have highly imbalanced node degrees
3. Max aggregation is preferred for directed graphs; concatenation for undirected graphs
4. Max aggregation achieves higher WL expressiveness; concatenation does not improve expressiveness
</div>

??? question "Show Answer"
    The correct answer is **A**. Concatenating L layer outputs produces a representation of size L × d, which grows with network depth and requires a downstream layer that accepts variable-length input. Max aggregation selects the maximum value per dimension across all L layers, always producing a d-dimensional output regardless of depth. This makes max JK-Net more parameter-efficient for deep networks. Concatenation is richer but requires a fixed number of layers; LSTM aggregation offers a middle ground with fixed output size and sequence-aware aggregation.

    **Concept Tested:** Jumping Knowledge Network (JK-Net)

---

#### Question 9

SGC's theoretical justification is that non-linearities between GCN layers are "unnecessary" for many tasks. Under what condition does this hold?

<div class="upper-alpha" markdown>
1. When the downstream task is linearly separable in the space of L-hop smoothed features
2. When the graph is a tree with no cycles
3. When the graph has a regular degree distribution
4. When the node feature matrix X is low-rank
</div>

??? question "Show Answer"
    The correct answer is **A**. SGC pre-computes the \(L\)-hop feature diffusion \(S^L X\) and then trains a linear classifier on top. This is effective when the optimal decision boundary in the smoothed feature space is (approximately) linear. For many homophilous node classification benchmarks (Cora, Citeseer), this condition holds — nodes in the same class cluster linearly in the diffused feature space. For heterophilous graphs or tasks requiring non-linear feature interactions, non-linearities remain important.

    **Concept Tested:** Simple Graph Convolution (SGC)

---

#### Question 10

APPNP's propagation rule \(H^{(t+1)} = (1-\alpha)\hat{S} H^{(t)} + \alpha H_0\) converges to a closed-form solution as the number of propagation steps approaches infinity. What does this closed form correspond to?

<div class="upper-alpha" markdown>
1. The stationary distribution of a random walk on the graph starting from node features \(H_0\)
2. The \(L_2\)-normalized version of \(H_0\) projected onto the dominant graph frequency
3. The Personalized PageRank (PPR) matrix applied to \(H_0\): \(H^* = (I - (1-\alpha)\hat{S})^{-1} \alpha H_0\)
4. The first eigenvector of the graph Laplacian scaled by \(\alpha\)
</div>

??? question "Show Answer"
    The correct answer is **C**. APPNP's iterative diffusion \(H^{(t+1)} = (1-\alpha)\hat{S} H^{(t)} + \alpha H_0\) is exactly the power iteration for computing the PPR vector from each node. At convergence, \(H^* = \alpha (I - (1-\alpha)\hat{S})^{-1} H_0\) — the PPR-weighted feature propagation. This means each node's final representation is the PPR-weighted combination of all nodes' initial (MLP-transformed) features, with the PPR weights providing principled multi-hop smoothing grounded in the random-walk interpretation of PageRank.

    **Concept Tested:** APPNP, Personalized PageRank
