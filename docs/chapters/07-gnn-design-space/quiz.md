# Quiz: GNN Design Space: GraphSAGE and GAT

Test your understanding of GraphSAGE, graph attention networks, pooling strategies, and the GNN design space.

---

#### Question 1

GraphSAGE differs from GCN in two key ways. What are they?

<div class="upper-alpha" markdown>
1. GraphSAGE uses spectral convolution; GCN uses spatial aggregation
2. GraphSAGE applies attention weights to edges while GCN uses uniform weights
3. GraphSAGE requires edge features while GCN only uses node features
4. GraphSAGE supports inductive learning (new nodes without retraining) and concatenates the node's own embedding with the aggregated neighbor embedding, rather than replacing it
</div>

??? question "Show Answer"
    The correct answer is **D**. GCN is transductive (the full adjacency matrix is used during training), while GraphSAGE samples neighborhoods and uses a learned aggregation function that generalizes to unseen nodes inductively. Additionally, GraphSAGE concatenates the node's own previous representation with the aggregated neighborhood: \(\mathbf{h}_v^{(l+1)} = \sigma\!\left(W \cdot \text{CONCAT}\!\left(\mathbf{h}_v^{(l)},\ \text{AGG}(\{\mathbf{h}_u : u \in \mathcal{N}(v)\})\right)\right)\), preserving the node's identity rather than blending it with neighbors.

    **Concept Tested:** GraphSAGE

---

#### Question 2

In a Graph Attention Network (GAT), how is the attention weight \(\alpha_{vu}\) between nodes \(v\) and \(u\) computed?

<div class="upper-alpha" markdown>
1. \(\alpha_{vu} = \text{softmax}_u\!\left(\text{LeakyReLU}(\mathbf{a}^\top [W\mathbf{h}_v \| W\mathbf{h}_u])\right)\), assigning learned importance to each neighbor
2. \(\alpha_{vu} = \cos(\mathbf{h}_v, \mathbf{h}_u) / \sum_{w \in \mathcal{N}(v)} \cos(\mathbf{h}_v, \mathbf{h}_w)\) (cosine similarity normalized)
3. \(\alpha_{vu} = 1 / d_v\), where \(d_v\) is the degree of \(v\) (uniform normalization like GCN)
4. \(\alpha_{vu} = \exp(-d(v,u))\) where \(d(v,u)\) is the shortest path distance between \(v\) and \(u\)
</div>

??? question "Show Answer"
    The correct answer is **A**. GAT computes raw attention scores \(e_{vu} = \text{LeakyReLU}(\mathbf{a}^\top [W\mathbf{h}_v \| W\mathbf{h}_u])\) for each neighbor \(u\) using a learnable attention vector \(\mathbf{a}\). These are then softmax-normalized over \(v\)'s neighborhood: \(\alpha_{vu} = \exp(e_{vu}) / \sum_{w \in \mathcal{N}(v)} \exp(e_{vw})\). The final aggregation is \(\mathbf{h}_v^{(l+1)} = \sigma\!\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu}\, W\mathbf{h}_u\right)\). This allows the GNN to learn which neighbors matter more for a given task.

    **Concept Tested:** Graph Attention Network (GAT), Attention Mechanism (Graph)

---

#### Question 3

Multi-head attention in GAT runs K independent attention mechanisms in parallel and then concatenates (or averages) the results. What is the primary benefit?

<div class="upper-alpha" markdown>
1. It reduces the number of trainable parameters by sharing weights across heads
2. It stabilizes training and allows the model to attend to different types of structural relationships simultaneously
3. It eliminates the need for the LeakyReLU non-linearity in attention computation
4. It guarantees that attention weights are symmetric (\(\alpha_{vu} = \alpha_{uv}\)) for undirected graphs
</div>

??? question "Show Answer"
    The correct answer is **B**. With K heads, each head learns a distinct attention pattern — one head might focus on high-degree hub connections, another on same-community neighbors. The K outputs are concatenated (or averaged in the last layer), giving the model a richer representation. Multi-head attention also stabilizes training via gradient averaging across heads, much like how ensemble models are more robust than single models.

    **Concept Tested:** Multi-Head Attention (Graph)

---

#### Question 4

A skip connection in a GNN adds the input of a layer to its output: \(\mathbf{h}^{(l+1)} = F(\mathbf{h}^{(l)}) + \mathbf{h}^{(l)}\). What specific problem does this address in deep GNNs?

<div class="upper-alpha" markdown>
1. It prevents the weight matrices from becoming sparse during training
2. It converts the GNN from inductive to transductive learning mode
3. It doubles the receptive field of each layer without adding extra message passing steps
4. It allows gradients to flow backward through the network without passing through every non-linearity, mitigating vanishing gradients and over-smoothing
</div>

??? question "Show Answer"
    The correct answer is **D**. Skip connections create gradient highways: during backpropagation, gradients can flow through the identity path \(\partial \mathbf{h}^{(l)} / \partial \mathbf{h}^{(l)} = 1\), bypassing the non-linear transformation. This prevents vanishing gradients in deep networks and also mitigates over-smoothing by ensuring that at least some of the original node representation survives after many layers of aggregation. ResNet (for images) and Jumping Knowledge Networks (for graphs) use this principle.

    **Concept Tested:** Skip Connection (GNN), Residual Connection (GNN)

---

#### Question 5

Jumping Knowledge Networks address the problem that different nodes in the same graph may benefit from different aggregation depths. How does JK-Net solve this?

<div class="upper-alpha" markdown>
1. It trains a separate GNN for each possible depth and ensembles their predictions
2. It computes representations at every layer and then selects or aggregates across all layer outputs for the final node representation
3. It applies attention weights that increase with layer depth, emphasizing deeper aggregations
4. It prunes edges in the graph based on attention weights before each layer
</div>

??? question "Show Answer"
    The correct answer is **B**. JK-Net collects the representations \(\mathbf{h}_v^{(1)}, \mathbf{h}_v^{(2)}, \ldots, \mathbf{h}_v^{(L)}\) from all \(L\) layers and applies a jump aggregation (max-pooling, LSTM, or concatenation) to combine them: \(\mathbf{h}_v^{\text{final}} = \text{AGG}(\mathbf{h}_v^{(1)}, \ldots, \mathbf{h}_v^{(L)})\). This allows the model to adaptively use shallow representations (capturing local structure) for peripheral nodes and deep representations (capturing global context) for central nodes — all within a single network.

    **Concept Tested:** Jumping Knowledge Network

---

#### Question 6

For a graph-level classification task, a graph readout function must aggregate all node representations into a single vector. What is the key advantage of sum pooling over mean pooling for this purpose?

<div class="upper-alpha" markdown>
1. Sum pooling is faster to compute because it avoids division
2. Sum pooling automatically normalizes features to the unit sphere
3. Sum pooling is differentiable while mean pooling is not
4. Sum pooling preserves graph size information (a graph with 100 nodes sums differently from one with 10 nodes even if average features are identical)
</div>

??? question "Show Answer"
    The correct answer is **D**. Mean pooling divides by the number of nodes, making graphs of different sizes indistinguishable if they have the same average node feature. For molecular classification, a molecule with 10 carbon atoms is fundamentally different from one with 100 carbon atoms, even if their average feature is the same. Sum pooling encodes both the distribution of features and the count, preserving this size information. This is why GIN (Chapter 10) uses sum as its readout.

    **Concept Tested:** Global Sum Pooling, Global Mean Pooling, Graph-Level Readout

---

#### Question 7

DiffPool learns a differentiable soft cluster assignment at each layer to hierarchically coarsen a graph. What are the two learned matrices it produces?

<div class="upper-alpha" markdown>
1. A graph adjacency matrix and a degree normalization matrix
2. A node feature matrix and an edge feature matrix
3. A soft assignment matrix S (which cluster each node belongs to) and an embedding matrix Z (updated node representations)
4. A query matrix and a key matrix for attention-based pooling
</div>

??? question "Show Answer"
    The correct answer is **C**. DiffPool trains two GNNs in parallel: \(\text{GNN}_{\text{pool}}\) produces \(S \in \mathbb{R}^{N \times C}\) (soft assignment of \(N\) nodes to \(C\) clusters, row-normalized via softmax), and \(\text{GNN}_{\text{embed}}\) produces \(Z \in \mathbb{R}^{N \times d}\) (node embeddings). The pooled graph has coarsened node features \(X_{\text{new}} = S^\top Z \in \mathbb{R}^{C \times d}\) and coarsened adjacency \(A_{\text{new}} = S^\top A S \in \mathbb{R}^{C \times C}\). This is applied hierarchically, collapsing the graph at each stage until a single graph-level representation is obtained.

    **Concept Tested:** DiffPool

---

#### Question 8

For link prediction, a common approach is to score candidate edges using the trained node embeddings. What scoring function is typically used?

<div class="upper-alpha" markdown>
1. The dot product or sigmoid of the dot product: \(\text{score}(u, v) = \sigma(\mathbf{z}_u \cdot \mathbf{z}_v)\)
2. The cosine distance: \(\text{score}(u, v) = 1 - \cos(\mathbf{z}_u, \mathbf{z}_v)\)
3. The L2 distance: \(\text{score}(u, v) = \|\mathbf{z}_u - \mathbf{z}_v\|_2\) (lower is more likely an edge)
4. The maximum element in the element-wise product: \(\text{score}(u, v) = \max(\mathbf{z}_u \odot \mathbf{z}_v)\)
</div>

??? question "Show Answer"
    The correct answer is **A**. Link prediction typically scores edge \((u, v)\) as \(\sigma(\mathbf{z}_u \cdot \mathbf{z}_v)\), where \(\sigma\) is the sigmoid function. The training objective is binary cross-entropy: positive edges (true links) should have score \(\to 1\) and negative samples (non-edges) should have score \(\to 0\). The dot product is used because it rewards embeddings that are aligned (large positive dot product) for nodes that should be connected — directly reflecting the co-occurrence maximization used in node2vec.

    **Concept Tested:** Link Prediction, Edge-Level Task

---

#### Question 9

A virtual node is a special node added to the graph that connects to all real nodes. What property does this augmentation give the GNN?

<div class="upper-alpha" markdown>
1. It increases the effective depth of the network by adding an additional aggregation layer
2. It converts undirected graphs to directed graphs for more expressive aggregation
3. It provides every node with a global "summary" representation, enabling all-pairs information sharing in just two message passing steps
4. It prevents over-smoothing by introducing a high-degree bottleneck node
</div>

??? question "Show Answer"
    The correct answer is **C**. The virtual node aggregates information from all real nodes (acting as a global readout), and then every real node receives a message from the virtual node in the next step. This creates a two-hop path between any pair of real nodes via the virtual node — effectively providing global context in two message passing steps regardless of graph diameter. It is particularly useful in molecular property prediction where graph diameter can vary widely.

    **Concept Tested:** Virtual Node Augmentation

---

#### Question 10

The GNN design space paper (You et al., 2020) empirically evaluated thousands of GNN configurations. What was the surprising finding about the relative importance of design choices?

<div class="upper-alpha" markdown>
1. Attention mechanisms (GAT-style) always outperformed simple sum aggregation by a large margin
2. The choice of aggregation function was the single most important design decision, dominating all other choices
3. No single architecture consistently dominated — the best configuration depended on the dataset, with aggregation type, depth, skip connections, and batch normalization all mattering comparably
4. Deeper GNNs (10+ layers) always outperformed shallow ones across all benchmark datasets
</div>

??? question "Show Answer"
    The correct answer is **C**. The design space study found that no single "best" GNN exists across all tasks and datasets. Different combinations of aggregation (sum/mean/max), depth (2–8 layers), skip connections, batch normalization, and activation function were optimal for different settings. The practical implication is that GNN design should be treated as a systematic search problem, not a "use the latest architecture" problem — a finding that motivates the principled comparisons in this chapter.

    **Concept Tested:** Graph Attention Network (GAT), GraphSAGE, Node Classification
