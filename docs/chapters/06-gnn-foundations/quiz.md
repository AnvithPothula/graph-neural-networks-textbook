# Quiz: GNN Foundations: Message Passing and GCN

Test your understanding of message passing, GCN derivation, and the spectral vs. spatial views of graph convolutions.

---

#### Question 1

In the message passing neural network (MPNN) framework, a single layer consists of three operations. Which sequence is correct?

<div class="upper-alpha" markdown>
1. Aggregate → Message → Update
2. Message → Aggregate → Update
3. Update → Aggregate → Message
4. Message → Update → Aggregate
</div>

??? question "Show Answer"
    The correct answer is **B**. In one MPNN layer: (1) **Message**: each neighbor \(u\) sends a message \(m_{u \to v} = \text{MSG}(\mathbf{h}_u, \mathbf{h}_v, \mathbf{e}_{uv})\) to node \(v\); (2) **Aggregate**: \(v\) collects all incoming messages into a single vector \(\mathbf{a}_v = \text{AGG}(\{m_{u \to v} : u \in \mathcal{N}(v)\})\); (3) **Update**: \(v\) updates its representation \(\mathbf{h}_v^{(l+1)} = \text{UPD}(\mathbf{h}_v^{(l)}, \mathbf{a}_v)\). This sequence — compute, collect, update — is repeated for \(L\) layers, allowing information to propagate \(L\) hops.

    **Concept Tested:** Message Passing Neural Net, Message Function, Aggregation Function, Update Function

---

#### Question 2

The GCN layer propagation rule is \(H^{(l+1)} = \sigma\!\left(\tilde{D}^{-1/2} \tilde{A}\, \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)\). What is the purpose of the \(\tilde{A} = A + I\) term?

<div class="upper-alpha" markdown>
1. It normalizes the adjacency matrix so all edge weights sum to 1
2. It symmetrizes the adjacency matrix for directed graphs
3. It removes isolated nodes that would have a zero row in A
4. It adds self-loops so that each node includes its own features when aggregating neighbor information
</div>

??? question "Show Answer"
    The correct answer is **D**. Adding the identity matrix \(I\) to \(A\) (\(\tilde{A} = A + I\)) inserts a self-loop for every node. Without self-loops, a node's updated representation would be computed entirely from its neighbors' previous representations, excluding its own features. The renormalization trick (\(\tilde{A}\), \(\tilde{D}\)) then ensures the combined self-and-neighbor aggregation is properly normalized. Kipf & Welling (2017) showed this simple fix dramatically improves performance on semi-supervised node classification.

    **Concept Tested:** Graph Convolutional Network

---

#### Question 3

GCN derives from spectral graph convolutions via a first-order Chebyshev approximation. What does the Chebyshev approximation step achieve?

<div class="upper-alpha" markdown>
1. It replaces global spectral filtering (requiring full eigendecomposition) with a local \(k\)-hop polynomial filter, making computation \(O(E)\) instead of \(O(N^2)\)
2. It converts the graph from directed to undirected before applying convolution
3. It quantizes the graph Laplacian eigenvalues to integers, reducing numerical error
4. It replaces the adjacency matrix with the degree matrix to simplify gradient computation
</div>

??? question "Show Answer"
    The correct answer is **A**. The spectral convolution requires computing \(U\), \(\Lambda\), \(U^\top\) (eigendecomposition of the Laplacian) and applying a filter \(g(\Lambda)\) — an \(O(N^2)\) operation. ChebConv approximates \(g(\Lambda)\) by a degree-\(K\) Chebyshev polynomial in \(\Lambda\), which translates to \(K\) applications of the sparse matrix \((2\tilde{L}/\lambda_{\max} - I)\), a \(K\)-localized filter computable in \(O(K|E|)\) time. GCN further simplifies to \(K = 1\) with the first-order approximation, yielding the two-matrix product \(\tilde{D}^{-1/2} \tilde{A}\, \tilde{D}^{-1/2}\).

    **Concept Tested:** Chebyshev Polynomial Conv, Spectral Graph Convolution

---

#### Question 4

The normalized graph Laplacian is defined as \(L_{\text{sym}} = D^{-1/2} L D^{-1/2}\). What property of \(L_{\text{sym}}\) makes it suitable for spectral analysis?

<div class="upper-alpha" markdown>
1. \(L_{\text{sym}}\) is always invertible, making linear system solving straightforward
2. \(L_{\text{sym}}\) has eigenvalues in \([0, 2]\), is positive semi-definite, and its eigenvectors form an orthonormal basis
3. \(L_{\text{sym}}\) is equivalent to the adjacency matrix \(A\) for regular graphs
4. \(L_{\text{sym}}\) preserves edge directions, making it applicable to directed graphs
</div>

??? question "Show Answer"
    The correct answer is **B**. The symmetric normalized Laplacian \(L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}\) is positive semi-definite (all eigenvalues \(\geq 0\)) with eigenvalues bounded in \([0, 2]\). Its smallest eigenvalue is 0 (with multiplicity equal to the number of connected components), and its eigenvectors are orthonormal. These properties make it well-behaved for Chebyshev polynomial approximation and spectral clustering: its eigenvectors define a natural graph Fourier basis.

    **Concept Tested:** Normalized Graph Laplacian, Spectral Domain (Graph)

---

#### Question 5

After L GNN layers, what is the receptive field of a node v?

<div class="upper-alpha" markdown>
1. The set of all nodes within L hops of v, since each layer expands the neighborhood by one hop
2. The set of all nodes in the graph, since GNNs use global attention
3. The set of v's direct neighbors only, regardless of layer depth
4. The set of L randomly sampled nodes drawn during mini-batch training
</div>

??? question "Show Answer"
    The correct answer is **A**. Each message passing layer aggregates information from 1-hop neighbors. After \(L\) layers, node \(v\)'s representation depends on all nodes within \(L\) hops — its \(L\)-hop neighborhood (also called its computational graph). With \(L = 2\) and average degree \(d\), the receptive field has \(\sim d^2\) nodes. This exponential growth creates scalability challenges for large graphs and is also the source of over-smoothing when \(L\) is large.

    **Concept Tested:** Receptive Field (GNN), K-Hop Neighborhood, Layer Depth (GNN)

---

#### Question 6

In the spatial interpretation of GCN, the normalized propagation \(\tilde{D}^{-1/2} \tilde{A}\, \tilde{D}^{-1/2} H\) applies to node features. What does this operation compute for each node?

<div class="upper-alpha" markdown>
1. The maximum feature value among all neighbors, scaled by the node's degree
2. A degree-normalized weighted average of the node's own features and its neighbors' features
3. The element-wise product of the node's feature vector with the adjacency row
4. The spectral filter response in the Fourier domain of the graph
</div>

??? question "Show Answer"
    The correct answer is **B**. The \((i,j)\) entry of \(\tilde{D}^{-1/2} \tilde{A}\, \tilde{D}^{-1/2}\) is \(1/\sqrt{\tilde{d}_i \tilde{d}_j}\), where \(\tilde{d}_i\) is the degree with self-loop. Multiplying by \(H\) computes, for each node \(i\), the weighted average of its own features and its neighbors' features, where edge \((i,j)\) is weighted by \(1/\sqrt{\tilde{d}_i \tilde{d}_j}\). This symmetric normalization ensures high-degree nodes don't dominate the aggregation — a direct spatial interpretation of what the spectral filter is doing.

    **Concept Tested:** Neighborhood Aggregation, Spatial Domain (Graph)

---

#### Question 7

Why might mean aggregation be less expressive than sum aggregation for distinguishing graph structures?

<div class="upper-alpha" markdown>
1. Mean aggregation loses information about neighborhood size: a node with 2 neighbors averaging to [1,0] looks identical to one with 200 neighbors averaging to [1,0]
2. Mean aggregation requires more memory than sum aggregation during backpropagation
3. Mean aggregation cannot handle weighted edges, while sum aggregation can
4. Mean aggregation applies only to graphs with homogeneous node features
</div>

??? question "Show Answer"
    The correct answer is **A**. Mean aggregation divides by the degree, discarding neighborhood size information. Two nodes with 2 and 200 neighbors respectively, all with the same feature vector [1, 0], produce the same mean aggregation output [1, 0] — even though their neighborhoods are fundamentally different in size. Sum aggregation preserves this count: the first yields [2, 0] and the second [200, 0]. This distinction is why GIN (Chapter 10) uses sum aggregation to achieve maximum expressiveness.

    **Concept Tested:** Mean Aggregation, Sum Aggregation

---

#### Question 8

Spectral clustering uses the eigenvectors of the graph Laplacian to partition nodes. What property of the Laplacian's eigenvectors enables this?

<div class="upper-alpha" markdown>
1. The eigenvectors are always sparse, making cluster boundaries computationally cheap to find
2. The leading eigenvector concentrates all its mass on the highest-degree hub nodes
3. The eigenvectors are binary (0 or 1), directly encoding community membership
4. The Fiedler vector (second-smallest eigenvector) reveals the graph's minimum normalized cut, since nodes on opposite sides of the cut have opposite signs
</div>

??? question "Show Answer"
    The correct answer is **D**. The Fiedler vector (eigenvector of the second-smallest eigenvalue of the Laplacian) is the relaxed solution to the normalized cut problem. Nodes with positive Fiedler values tend to be in one community; nodes with negative values in another. The eigenvalue itself measures the strength of the cut: a near-zero second eigenvalue indicates a nearly disconnected graph with a natural partition. K-way spectral clustering uses the first K eigenvectors as node features and applies k-means to find K communities.

    **Concept Tested:** Spectral Clustering, Normalized Cut

---

#### Question 9

What is the difference between the spectral and spatial approaches to graph convolution?

<div class="upper-alpha" markdown>
1. Spectral convolution uses multiplication; spatial convolution uses addition
2. Spectral methods use edge features; spatial methods use only node features
3. Spectral methods define convolution in the eigenvector domain of the Laplacian; spatial methods define convolution by directly aggregating neighbor features in the graph domain
4. Spectral methods only work on directed graphs; spatial methods only on undirected graphs
</div>

??? question "Show Answer"
    The correct answer is **C**. Spectral convolution defines a graph filter \(g\) as a function of the Laplacian's eigenvalues: \((g \star \mathbf{x})_{\text{spectral}} = U\, g(\Lambda)\, U^\top \mathbf{x}\). This requires computing eigenvectors and is graph-specific (the filter learned on one graph cannot transfer to another). Spatial methods directly aggregate neighbor features using learnable functions in the graph domain — no eigendecomposition needed. GCN bridges the two views: its spatial formula is the first-order approximation of a specific spectral filter.

    **Concept Tested:** Spectral Domain (Graph), Spatial Domain (Graph)

---

#### Question 10

The `MessagePassing` base class in PyTorch Geometric provides hooks: `message()`, `aggregate()`, and `update()`. Which argument to `propagate()` controls which neighborhood is used for aggregation?

<div class="upper-alpha" markdown>
1. The `x` tensor, which specifies node features to aggregate
2. The `aggr` string parameter, which selects between "add", "mean", and "max"
3. The `edge_index` tensor of shape [2, E], which defines the source-destination pairs for message passing
4. The `batch` tensor, which groups nodes by graph in mini-batch training
</div>

??? question "Show Answer"
    The correct answer is **C**. In PyG, `self.propagate(edge_index, x=x)` dispatches message passing along the edges specified in `edge_index`. Row 0 contains source node indices (where messages come from) and row 1 contains destination node indices (where messages go). The `message()` hook receives features of source nodes, `aggregate()` collects them per destination, and `update()` computes the final representation. The `aggr` string selects the aggregation strategy but does not define the graph structure.

    **Concept Tested:** PyTorch Geometric (PyG), Message Passing Neural Net
