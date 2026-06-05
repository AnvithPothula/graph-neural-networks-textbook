# Quiz: Graph Transformers

Test your understanding of Graph Transformers, positional encodings, and structural encodings.

---

#### Question 1

Standard Transformers apply full self-attention between all token pairs. When applied directly to a graph, what fundamental limitation does this create?

<div class="upper-alpha" markdown>
1. Transformers cannot process variable-length sequences, so graphs with different node counts cannot be handled
2. Self-attention is permutation invariant — without additional positional information, the Transformer cannot distinguish the graph's topology, treating all nodes as an unordered set
3. The quadratic attention complexity \(O(N^2)\) is smaller than the \(O(E)\) cost of message passing for dense graphs
4. Transformers cannot compute gradients through the attention softmax, making training infeasible on graphs
</div>

??? question "Show Answer"
    The correct answer is **B**. A standard Transformer applies the same computation to all pairs of nodes regardless of whether they are connected. Without positional or structural encodings, the model has no way to distinguish a path graph from a star graph of the same size — both look like an unordered set of N nodes. This is the core problem Graph Transformers address: injecting structural information (graph topology) that vanilla attention ignores.

    **Concept Tested:** Graph Transformer

---

#### Question 2

Graphormer introduces centrality encoding by adding a node's degree as a bias to its initial embedding. What does this achieve?

<div class="upper-alpha" markdown>
1. It makes Graphormer equivalent to GCN by normalizing attention scores by degree
2. It allows the model to distinguish hub nodes from peripheral nodes at the input level, before any attention layers are applied
3. It removes the need for positional encoding because degree uniquely identifies each node
4. It converts the graph from directed to undirected by symmetrizing in-degree and out-degree
</div>

??? question "Show Answer"
    The correct answer is **B**. Graphormer's centrality encoding adds learnable degree embeddings to node features: \(\mathbf{h}_v \leftarrow \mathbf{h}_v + \mathbf{z}_{\deg^-(v)} + \mathbf{z}_{\deg^+(v)}\). This injects structural information — hub nodes with high degree receive a different embedding boost than leaf nodes — before any attention is applied. Combined with spatial bias (shortest-path distances) and edge encodings, Graphormer achieved state-of-the-art on OGB-LSC graph-level tasks at time of publication.

    **Concept Tested:** Graphormer, Centrality Encoding

---

#### Question 3

Laplacian Positional Encoding (LapPE) uses eigenvectors of the graph Laplacian as node position features. What is the "sign ambiguity" problem with LapPE?

<div class="upper-alpha" markdown>
1. The Laplacian eigenvectors may not be real-valued for graphs with negative edge weights
2. Eigenvectors of different graphs have different dimensionalities, preventing batch training
3. Each eigenvector can be multiplied by \(-1\) (sign flip) and remain a valid eigenvector, so the same node can receive either \(+\varphi\) or \(-\varphi\) as its positional encoding depending on the eigensolver used
4. The Laplacian has too many eigenvectors for large graphs, making it computationally intractable
</div>

??? question "Show Answer"
    The correct answer is **C**. If \(\varphi\) is an eigenvector of the Laplacian with eigenvalue \(\lambda\), then so is \(-\varphi\). Different eigensolvers, or the same solver on different hardware, may return either sign. This means two nodes in identical structural positions could receive \(+\varphi\) or \(-\varphi\) as their positional encoding — making LapPE non-deterministic and breaking the consistency that positional encoding is supposed to provide. Sign-invariant networks (SignNet) address this by learning functions that produce the same output for both \(\varphi\) and \(-\varphi\).

    **Concept Tested:** Laplacian Positional Encoding, Sign-Invariant Network

---

#### Question 4

Random Walk Structural Encoding (RWSE) uses landing probabilities of random walks as node features. What structural information does it capture that LapPE does not?

<div class="upper-alpha" markdown>
1. RWSE captures the global degree distribution while LapPE only captures local connectivity
2. RWSE is strictly a function of the node itself (diagonal entries of powers of the random walk matrix), making it sign-invariant by construction and capturing cycle and symmetry information
3. RWSE captures edge directions while LapPE only works on undirected graphs
4. RWSE uses the full random walk matrix while LapPE uses only the top-k eigenvectors
</div>

??? question "Show Answer"
    The correct answer is **B**. RWSE computes \(p_k(v,v) = [M^k]_{vv}\) — the probability of a \(k\)-step random walk returning to node \(v\). This is a purely diagonal, node-specific quantity that requires no eigenvector computation. Crucially, it is deterministic and sign-invariant by construction. The sequence \((p_1(v,v), p_2(v,v), \ldots, p_K(v,v))\) captures structural properties like whether \(v\) is part of cycles, triangles, or other local substructures — information that complements the global positioning provided by LapPE.

    **Concept Tested:** Random Walk Struct Encoding

---

#### Question 5

GPS (General Powerful Scalable GNN) combines local message passing with global attention. What is the key architectural insight?

<div class="upper-alpha" markdown>
1. GPS runs local MPNN and global Transformer attention in parallel at each layer and sums their outputs, capturing both local structural patterns and global long-range dependencies simultaneously
2. GPS alternates between spectral and spatial convolutions in successive layers
3. GPS replaces positional encoding with a learned degree matrix, reducing computation from \(O(N^2)\) to \(O(E)\)
4. GPS uses the GPS algorithm from combinatorial optimization to find optimal node orderings before applying attention
</div>

??? question "Show Answer"
    The correct answer is **A**. Each GPS layer computes two parallel streams: a local MPNN aggregation (capturing neighborhood structure) and a global Transformer self-attention (capturing long-range correlations), then combines them: \(\mathbf{h}_v^{(l+1)} = \text{MLP}\!\left(\text{MPNN}(\mathbf{h}_v^{(l)}) + \text{Attn}(\mathbf{h}_v^{(l)}, \mathbf{H}^{(l)})\right)\). This architecture is both powerful (each component is independently expressive) and flexible (the MPNN and Transformer components can be swapped independently). GPS achieves competitive results on diverse benchmarks with a well-understood inductive bias.

    **Concept Tested:** GPS (General Powerful GNN)

---

#### Question 6

Graphormer's spatial bias adds the learnable scalar \(\varphi(\text{spd}(u,v))\) to the attention logit between nodes \(u\) and \(v\), where \(\text{spd}(u,v)\) is the shortest path distance. What does this enable?

<div class="upper-alpha" markdown>
1. It makes attention weights proportional to edge weights rather than learned query-key products
2. It replaces the standard attention computation, reducing complexity from \(O(N^2)\) to \(O(N \log N)\)
3. It prevents attention from being computed between nodes in different connected components
4. It biases attention so that nodes closer in graph distance receive higher attention scores, injecting graph topology into the attention mechanism
</div>

??? question "Show Answer"
    The correct answer is **D**. The spatial bias \(\varphi(d(u,v))\) is a learnable scalar that depends only on the shortest-path distance between \(u\) and \(v\). Adding it to the attention logit before softmax biases the model toward attending more to structurally nearby nodes (low \(d\)) and less to distant nodes (high \(d\)). The function \(\varphi\) is learned, so the model can also invert this and attend to distant nodes if that helps the task. This replaces the rigid local/non-local distinction of MPNNs with a soft, learned distance-aware attention.

    **Concept Tested:** Spatial Bias (Graph Attention)

---

#### Question 7

SAN (Spectral Attention Network) uses a learned combination of the graph Laplacian's eigenvectors as positional encoding. What distinguishes SAN from simply using the top-k LapPE eigenvectors?

<div class="upper-alpha" markdown>
1. SAN learns to attend over all N eigenvectors (both low and high frequency), with a separate attention mechanism selecting the relevant frequency components for the task
2. SAN computes eigenvectors of the adjacency matrix rather than the Laplacian
3. SAN normalizes eigenvectors by their eigenvalues, while standard LapPE uses unnormalized vectors
4. SAN uses complex-valued eigenvectors for directed graphs; LapPE is restricted to real-valued encodings
</div>

??? question "Show Answer"
    The correct answer is **A**. Standard LapPE takes only the k eigenvectors corresponding to the k smallest eigenvalues (smoothest frequency components). SAN instead uses an attention mechanism over all N eigenvectors, allowing the model to selectively attend to relevant frequency bands for the specific task. High-frequency eigenvectors (large eigenvalues) encode fine-grained local structure like cycle membership, which may be important for some molecular tasks but irrelevant for others. Learned attention over the full spectrum is more expressive than a fixed-k truncation.

    **Concept Tested:** SAN (Spectral Attention Net)

---

#### Question 8

BasisNet addresses the "basis ambiguity" problem in LapPE. When does basis ambiguity arise and why is it harder to fix than sign ambiguity?

<div class="upper-alpha" markdown>
1. Basis ambiguity arises from floating-point errors in the eigensolver, making results non-reproducible
2. Basis ambiguity means the Laplacian has too many eigenvectors to process in a single forward pass
3. Basis ambiguity occurs only for bipartite graphs, which have paired eigenvalues \(\pm\lambda\)
4. When an eigenvalue has multiplicity > 1 (repeated eigenvalues), the corresponding eigenspace has multiple valid orthonormal bases — any rotation within that eigenspace is equally valid, so the encoding is non-unique up to an arbitrary orthogonal transformation
</div>

??? question "Show Answer"
    The correct answer is **D**. Sign ambiguity (\(\varphi\) vs. \(-\varphi\)) is a special case where the "rotation" is just a sign flip — a 1D issue. Basis ambiguity arises when an eigenvalue has multiplicity \(m > 1\): any \(m \times m\) orthogonal rotation of the \(m\) eigenvectors in that eigenspace is equally valid. A node's positional encoding could be any point in an \(m\)-dimensional rotational orbit. BasisNet learns functions that are invariant to all orthogonal transformations within each eigenspace, a much harder constraint to enforce than simple sign invariance.

    **Concept Tested:** Basis-Invariant Network

---

#### Question 9

Edge encoding in Graphormer injects features along the shortest path between nodes u and v into the attention score. Why use path features rather than single edge features?

<div class="upper-alpha" markdown>
1. Single edge features are not available for molecular graphs, which only have node atom types
2. Edge features between non-adjacent nodes are undefined, so path features are used as a proxy for direct connection
3. The relationship between two non-adjacent nodes is captured by the intermediate edges on the path between them — path edge encoding provides context for long-range dependencies
4. Path encoding reduces the attention computation from \(O(N^2)\) to \(O(N)\) per layer
</div>

??? question "Show Answer"
    The correct answer is **C**. When computing attention between nodes u and v that are not directly connected, their relationship is mediated by the edges along their shortest path. In molecular graphs, the bond types (single, double, aromatic) along the path between two atoms carry chemical information about the relationship. Graphormer's edge encoding averages the learned representations of all edges on the shortest path and adds this to the attention logit, providing richer pairwise context than either "no edge" or "adjacent edge" alone.

    **Concept Tested:** Edge Encoding (Transformer)

---

#### Question 10

Position-Aware GNNs (P-GNNs) assign position-aware representations to nodes by computing distances to a set of anchor nodes. What is the advantage of this approach over Laplacian eigenvector-based encodings?

<div class="upper-alpha" markdown>
1. P-GNN encodings are always unique (no sign or basis ambiguity) because shortest-path distances are scalar and deterministic
2. P-GNN encodings capture higher-frequency graph properties than eigenvectors can represent
3. P-GNN encodings require no pre-computation because distances are computed during the forward pass
4. P-GNN encodings work for directed graphs while eigenvector encodings do not
</div>

??? question "Show Answer"
    The correct answer is **A**. The shortest-path distance from node v to anchor node a is a non-negative integer — it has no sign or basis ambiguity. Different nodes at different graph distances from the anchors receive provably distinct positional fingerprints (with high probability when anchors are randomly sampled). This deterministic uniqueness is a practical advantage over LapPE, which requires either sign correction (SignNet) or basis correction (BasisNet) to be well-defined.

    **Concept Tested:** Position-Aware GNN
