# Quiz: GNN Training, Augmentation, and Practical Tips

Test your understanding of GNN training techniques, regularization, data augmentation, and self-supervised learning.

---

#### Question 1

Cross-entropy loss for node classification is computed as \(L = -\sum_i \sum_k y_{ik} \log(\hat{y}_{ik})\). What does the \(y_{ik}\) term represent?

<div class="upper-alpha" markdown>
1. The predicted probability that node i belongs to class k
2. The true label indicator: 1 if node i belongs to class k, 0 otherwise
3. The attention weight assigned to node i's k-th neighbor
4. The degree of node i normalized by the number of classes k
</div>

??? question "Show Answer"
    The correct answer is **B**. In multi-class cross-entropy, \(y_{ik}\) is the ground-truth one-hot indicator — 1 for the true class and 0 for all other classes. \(\hat{y}_{ik} = \text{softmax}(\text{logit})_k\) is the GNN's predicted probability. The loss penalizes low predicted probability for the true class. For binary classification, this simplifies to binary cross-entropy: \(L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]\).

    **Concept Tested:** Cross-Entropy Loss, Binary Cross-Entropy

---

#### Question 2

The transductive split and inductive split differ in how test nodes relate to training. Which statement correctly describes the transductive setting?

<div class="upper-alpha" markdown>
1. Test nodes are drawn from a different graph distribution than training nodes
2. Test nodes are not seen during training; the model must generalize to completely new graphs
3. Test nodes are present in the training graph (their edges are visible), but their labels are withheld until evaluation
4. Test nodes have no edges to training nodes, forming an isolated subgraph
</div>

??? question "Show Answer"
    The correct answer is **C**. In the transductive setting (standard for node classification on citation networks like Cora), all nodes including test nodes are in the training graph. The model can use their structural position and features when computing embeddings, but their labels are masked during training. This contrasts with the inductive setting (Chapter 4 revisited), where test nodes — and their neighborhoods — are withheld entirely until evaluation time.

    **Concept Tested:** Transductive Split, Inductive Split

---

#### Question 3

Over-smoothing is a well-known problem in deep GNNs. What does over-smoothing mean, and why does adding more layers cause it?

<div class="upper-alpha" markdown>
1. The model parameters become too large, causing memory overflow on GPU
2. Edge features are overwritten by node features after several aggregation rounds
3. The loss function plateaus early, preventing further gradient-based improvement
4. Node representations converge toward the same vector as depth increases, because repeated averaging over neighbors destroys local distinctions
</div>

??? question "Show Answer"
    The correct answer is **D**. With L message passing layers, each node aggregates information from its L-hop neighborhood. As L grows large, the receptive fields of all nodes eventually overlap — every node "sees" the entire graph. Repeated averaging pushes representations toward the graph's mean feature vector, losing local structural distinctions. Formally, node representations converge to a function of node degree alone. Techniques like DropEdge, PairNorm, and skip connections (Chapter 7) mitigate this.

    **Concept Tested:** Batch Normalization (GNN), Layer Normalization (GNN)

---

#### Question 4

DropEdge randomly removes edges from the graph during training. Why does this help reduce over-smoothing?

<div class="upper-alpha" markdown>
1. By randomly removing edges, it reduces message flow between nodes, slowing the convergence of representations toward a global average
2. It prevents high-degree hub nodes from receiving too many messages, balancing degree distributions
3. It reduces the number of parameters in the network, acting as standard weight dropout
4. It converts the graph to a tree structure, which has no cycles and therefore no over-smoothing
</div>

??? question "Show Answer"
    The correct answer is **A**. DropEdge stochastically removes each edge with probability p during training, creating sparser versions of the graph at each iteration. This restricts information propagation: a node with 10 neighbors might only aggregate from 7 in a given step. Sparser propagation slows the convergence of representations across the whole graph, acting as a regularizer against over-smoothing. It also acts as data augmentation, exposing the model to different graph topologies.

    **Concept Tested:** DropEdge

---

#### Question 5

PairNorm normalizes node representations such that the sum of pairwise squared distances is constant. What problem does this directly address?

<div class="upper-alpha" markdown>
1. It prevents exploding gradients during backpropagation through many layers
2. It directly counteracts over-smoothing by ensuring node representations remain spread out rather than collapsing toward the mean
3. It normalizes the graph adjacency matrix to have unit spectral radius
4. It prevents feature dimensions with large variance from dominating the aggregation
</div>

??? question "Show Answer"
    The correct answer is **B**. PairNorm first centers node representations (subtract mean), then scales them so that \(\sum_i \sum_j \|\mathbf{h}_i - \mathbf{h}_j\|^2 = s \cdot N^2\) for a chosen scale constant \(s\). This directly prevents over-smoothing: if all representations collapsed toward the mean, pairwise distances would all approach zero — violating the PairNorm constraint. The normalization forces the model to maintain diversity among node representations regardless of network depth.

    **Concept Tested:** PairNorm

---

#### Question 6

Neighbor sampling in mini-batch GNN training (as in GraphSAGE) fixes the computational graph size by sampling k neighbors per node per layer. What is the main trade-off?

<div class="upper-alpha" markdown>
1. Exact gradients, but requires loading the entire graph into memory per batch
2. Reduced over-smoothing, but requires more layers to achieve the same receptive field
3. Faster convergence, but cannot be applied to heterogeneous graphs
4. Lower memory cost per batch, but gradient estimates become stochastic (high variance for small k)
</div>

??? question "Show Answer"
    The correct answer is **D**. Neighbor sampling caps the computational graph at \(k^L\) nodes per training example, making memory cost independent of graph size. However, by sampling only a subset of neighbors, the gradient estimate is a noisy approximation of the full-graph gradient — variance increases as \(k\) decreases. Small \(k\) (e.g., \(k=2\)) is very efficient but noisy; large \(k\) (e.g., \(k=25\)) approaches full-neighborhood accuracy but increases memory. This trade-off is the central design tension in scalable GNN training.

    **Concept Tested:** Neighbor Sampling (Mini-Batch), Mini-Batch GNN Training

---

#### Question 7

Deep Graph Infomax (DGI) trains a GNN encoder without any node labels. What is the training objective?

<div class="upper-alpha" markdown>
1. Predict the class labels of masked nodes from their structural neighborhood
2. Minimize the sum of pairwise distances between node embeddings in the same community
3. Reconstruct the adjacency matrix from node embeddings using a sigmoid decoder
4. Maximize mutual information between node-level representations and a summary of the whole graph, while minimizing mutual information with representations from corrupted graphs
</div>

??? question "Show Answer"
    The correct answer is **D**. DGI uses a discriminator that must distinguish between (positive) pairs of (node representation, global graph summary) from the real graph and (negative) pairs where the graph summary comes from a corrupted graph (shuffled node features). The encoder is trained to produce node representations that are maximally informative about the graph's global structure. This self-supervised objective yields embeddings that transfer well to downstream classification without any labels.

    **Concept Tested:** Deep Graph Infomax (DGI)

---

#### Question 8

Graph Contrastive Learning (GraphCL) generates two views of a graph through augmentation. Which of the following is NOT a standard GraphCL augmentation?

<div class="upper-alpha" markdown>
1. Graph label flipping (randomly reassigning graph-level class labels)
2. Edge dropping (randomly removing a fraction of edges)
3. Node feature masking (randomly zeroing out some feature dimensions)
4. Subgraph sampling (taking a random connected subgraph)
</div>

??? question "Show Answer"
    The correct answer is **A**. Graph label flipping is not an augmentation strategy in GraphCL — it destroys the ground truth signal and would corrupt supervised learning. The three standard GraphCL augmentations are: (1) node feature masking, (2) edge dropping/addition, and (3) subgraph sampling/node dropping. Two augmented views of the same graph form a positive pair; views from different graphs are negative pairs. The model maximizes agreement between representations of positive pairs while pushing negative pairs apart.

    **Concept Tested:** Graph Contrastive Learning (GraphCL)

---

#### Question 9

Batch normalization is applied to GNN layers to stabilize training. How is it adapted for graph-structured data compared to standard batch norm in CNNs?

<div class="upper-alpha" markdown>
1. Graph batch norm applies a separate normalization per edge, ensuring message magnitudes are unit-scale before aggregation
2. Graph batch norm normalizes across the edge dimension rather than the node dimension
3. Graph batch norm normalizes node feature vectors across all nodes and graphs in a mini-batch, using the batch statistics of node embeddings
4. Graph batch norm is applied after the readout function, not within message passing layers
</div>

??? question "Show Answer"
    The correct answer is **C**. Batch normalization in GNNs normalizes the feature vector of each dimension across all nodes in the mini-batch: for feature dimension \(j\), it computes \(\mu_j = \text{mean over all node embeddings } h_{i,j}\) and \(\sigma_j^2 = \text{variance}\), then normalizes. This is analogous to how CNN batch norm normalizes across the spatial dimensions. It stabilizes the distribution of node representations as they pass through layers, mitigating internal covariate shift.

    **Concept Tested:** Batch Normalization (GNN)

---

#### Question 10

The ROC-AUC score is often preferred over accuracy for evaluating GNN link prediction on highly imbalanced graphs. Why?

<div class="upper-alpha" markdown>
1. ROC-AUC is always higher than accuracy, making papers look better
2. Accuracy is undefined for regression tasks; ROC-AUC works for both classification and regression
3. In sparse graphs, the number of non-edges vastly outnumbers edges, so a classifier predicting "no edge" everywhere achieves high accuracy but zero utility; ROC-AUC measures ranking quality across all thresholds
4. ROC-AUC directly penalizes false negatives more than false positives, which is appropriate for link prediction
</div>

??? question "Show Answer"
    The correct answer is **C**. For a graph with N = 10,000 nodes and E = 50,000 edges, there are ~50 million possible edges — 99.9% negative. A trivial classifier predicting "no edge" everywhere achieves 99.9% accuracy but is useless. ROC-AUC measures the probability that a randomly chosen positive pair (true edge) is ranked higher than a randomly chosen negative pair — a threshold-independent quality measure that is not inflated by class imbalance.

    **Concept Tested:** ROC-AUC Score
