# Quiz: Heterogeneous Graphs

Test your understanding of heterogeneous GNNs, R-GCN, HGT, meta-paths, and multi-relational message passing.

---

#### Question 1

A heterogeneous graph differs from a homogeneous graph in what fundamental way?

<div class="upper-alpha" markdown>
1. Heterogeneous graphs have multiple node types and/or multiple edge types, requiring type-specific processing
2. Heterogeneous graphs are weighted while homogeneous graphs are unweighted
3. Heterogeneous graphs have directed edges while homogeneous graphs have undirected edges
4. Heterogeneous graphs require separate adjacency matrices while homogeneous graphs use a single matrix
</div>

??? question "Show Answer"
    The correct answer is **A**. A heterogeneous graph \(G = (V, E, \tau, \phi)\) has a node type mapping \(\tau: V \to \mathcal{A}\) and an edge type mapping \(\phi: E \to \mathcal{R}\), where \(|\mathcal{A}| + |\mathcal{R}| > 2\). For example, an academic network has node types {paper, author, venue} and edge types {writes, cites, publishes_in}. This type structure means a single weight matrix cannot handle all interactions — messages from an "author" node to a "paper" node carry different semantics than messages between two "paper" nodes.

    **Concept Tested:** Heterogeneous GNN

---

#### Question 2

R-GCN (Relational GCN) handles multiple edge types by learning a separate weight matrix W_r per relation type. What is the scalability problem this creates, and how does basis decomposition solve it?

<div class="upper-alpha" markdown>
1. Separate matrices cannot be applied in parallel; basis decomposition enables batch matrix multiplication
2. Separate matrices require more GPU memory; basis decomposition quantizes them to 8-bit precision
3. With \(R\) relation types, \(R\) weight matrices create \(O(R)\) parameters that can overfit or become infeasible for large \(R\); basis decomposition expresses each \(W_r\) as a linear combination of \(B\) shared basis matrices, reducing parameters from \(O(Rd^2)\) to \(O(Bd^2 + RB)\)
4. Separate matrices prevent gradient flow between relation types; basis decomposition shares gradients
</div>

??? question "Show Answer"
    The correct answer is **C**. For a KG with thousands of relation types and \(d\)-dimensional embeddings, maintaining \(R\) separate \(d \times d\) weight matrices requires \(O(Rd^2)\) parameters — easily tens of millions for large \(R\). Basis decomposition writes \(W_r = \sum_b a_{r,b} V_b\), where \(B \ll R\) basis matrices \(V_b\) are shared across all relation types and \(a_{r,b}\) are scalar mixing coefficients. This reduces parameters to \(O(Bd^2)\) for bases plus \(O(RB)\) for mixing coefficients, enabling R-GCN to scale to multi-relational KGs.

    **Concept Tested:** R-GCN (Relational GCN), Basis Decomposition (R-GCN)

---

#### Question 3

A meta-path in a heterogeneous graph defines a sequence of node and edge types. What does the meta-path "Author → Paper → Author" (via writes) capture?

<div class="upper-alpha" markdown>
1. The path from an author to all papers they cited
2. Co-authorship: two authors are connected via this meta-path if they co-authored at least one paper
3. The citation relationship between authors mediated by a shared advisor
4. The influence path from one venue's papers to another venue's authors
</div>

??? question "Show Answer"
    The correct answer is **B**. Following "Author -writes→ Paper -written_by→ Author" connects two authors when they are both listed as authors of the same paper — co-authorship. Meta-paths define typed connectivity patterns that have domain-specific semantics. In academic network analysis, common meta-paths include APA (co-authorship), APVPA (authors sharing a venue), and APCPA (authors sharing a citation). HAN uses separate attention networks per meta-path and combines them to learn which relationship types matter for a given task.

    **Concept Tested:** Meta-Path

---

#### Question 4

The Heterogeneous Graph Transformer (HGT) extends the Transformer architecture to heterogeneous graphs. What are the three type-specific components it introduces?

<div class="upper-alpha" markdown>
1. Type-specific adjacency, type-specific readout, and type-specific loss function
2. Type-specific positional encodings, type-specific layer normalization, and type-specific feed-forward networks
3. Type-specific learning rates, type-specific dropout rates, and type-specific batch normalization
4. Type-specific projection matrices for queries/keys/values, type-specific attention weights, and type-specific message functions — one set per source-type × edge-type × target-type combination
</div>

??? question "Show Answer"
    The correct answer is **D**. HGT computes attention for each source node \(s \to\) target node \(t\) of types \(\tau(s)\) and \(\tau(t)\) via edge type \(\phi(e)\) using: (1) type-specific linear projections \(K^{\tau(s)}\), \(Q^{\tau(t)}\) for keys and queries; (2) a type-specific attention matrix \(W^{\phi(e)}\) that scores how much source type \(\tau(s)\) attends to target type \(\tau(t)\) via edge \(\phi(e)\); (3) type-specific value projections \(V^{\tau(s)}\). This gives each (source\_type, edge\_type, target\_type) triplet its own semantics, capturing the heterogeneous relational structure that homogeneous attention ignores.

    **Concept Tested:** Heterogeneous Graph Transformer (HGT), Type-Specific Projection

---

#### Question 5

HAN (Heterogeneous Attention Network) aggregates information in two stages. What are the two levels of attention?

<div class="upper-alpha" markdown>
1. Node-level attention (over neighbors of the same type) and edge-level attention (over edge weights)
2. Node-level attention (over neighbors within a meta-path neighborhood) and semantic-level attention (over different meta-paths, weighting their relative importance)
3. Layer-level attention (over GNN layers) and graph-level attention (over multiple graphs)
4. Intra-type attention (within a node type) and inter-type attention (across node types)
</div>

??? question "Show Answer"
    The correct answer is **B**. HAN's two-level attention hierarchy: (1) node-level attention weights individual neighbors within a specific meta-path neighborhood (e.g., which co-authors are most informative for this paper?); (2) semantic-level attention weights the different meta-paths themselves (e.g., is co-authorship or citation more informative for predicting paper topic?). The semantic-level weights are a soft meta-path selector, learned end-to-end. The final representation is a weighted sum of meta-path-specific aggregations.

    **Concept Tested:** HAN (Heterogeneous Attention Network)

---

#### Question 6

In R-GCN for link prediction on KGs, the decoder and encoder play separate roles. What is each responsible for?

<div class="upper-alpha" markdown>
1. Encoder handles undirected edges; decoder handles directed edges
2. Encoder predicts link existence; decoder produces node embeddings
3. Encoder computes entity representations by aggregating relational neighborhoods; decoder scores (h, r, t) triples using the entity representations with a function like DistMult or TransE
4. Encoder runs during training; decoder runs only during inference
</div>

??? question "Show Answer"
    The correct answer is **C**. R-GCN uses a GNN encoder to compute entity embeddings \(\mathbf{z}_h\) and \(\mathbf{z}_t\) by aggregating messages from multi-relational neighborhoods, then a decoder (e.g., DistMult: \(f(h,r,t) = \sum_i z_{h,i} \cdot r_i \cdot z_{t,i}\)) to score triples. This encoder-decoder split allows R-GCN to leverage graph structure (missing in pure embedding methods like TransE) while still using efficient bilinear scoring. The encoder is particularly powerful because it propagates information from multi-hop relational neighborhoods.

    **Concept Tested:** R-GCN (Relational GCN), Relation-Specific Weights

---

#### Question 7

Type-specific projection in HGT maps node features of different types into a shared representation space. Why is this projection necessary?

<div class="upper-alpha" markdown>
1. Type-specific projection normalizes all node features to unit norm for numerical stability
2. It converts discrete type labels into continuous embeddings for the attention mechanism
3. It prevents gradient flow between different node types during backpropagation
4. Different node types often have features of different dimensions or semantics, requiring separate linear maps to a common d-dimensional space before attention can be computed
</div>

??? question "Show Answer"
    The correct answer is **D**. In a heterogeneous academic graph, papers might have TF-IDF features (1000-dim), authors might have affiliation embeddings (256-dim), and venues might have acceptance-rate scalars. Without type-specific projection, these cannot be concatenated or compared. Each type's projection matrix \(W^\tau \in \mathbb{R}^{d_\tau \times d}\) maps from the type's native feature dimension \(d_\tau\) to the common \(d\)-dimensional GNN space. This projection also learns which features of each type are relevant for the task.

    **Concept Tested:** Type-Specific Projection

---

#### Question 8

The academic knowledge graph DBLP is commonly used to benchmark heterogeneous GNNs. It has four node types: paper, author, venue, and term. A meta-path "Author → Paper → Term → Paper → Author" connects authors via shared keywords. What structural phenomenon does this capture?

<div class="upper-alpha" markdown>
1. Authors whose papers share common terminology or research keywords, capturing topic similarity beyond direct co-authorship
2. Co-authorship mediated by the venue where the paper was published
3. Authors who cite each other's work on the same topic
4. Authors who reviewed each other's papers in the same venue
</div>

??? question "Show Answer"
    The correct answer is **A**. The APTPA meta-path (Author-Paper-Term-Paper-Author) connects two authors when they write papers sharing a common index term (keyword). This captures semantic similarity: two authors working on "graph neural networks" are connected even if they never co-authored a paper. This is a richer signal than co-authorship (APA) for tasks like author similarity or community detection, since topic affinity persists even for researchers who never directly collaborate.

    **Concept Tested:** Meta-Path

---

#### Question 9

R-GCN handles the self-loop contribution separately from relational messages. Why is the self-loop given its own weight matrix W₀?

<div class="upper-alpha" markdown>
1. The self-loop prevents isolated nodes from having zero representations
2. W₀ implements batch normalization for the node's own features before aggregation
3. The self-loop corresponds to a special relation type (reflexive), and its influence should be learned independently from how the node's features are updated by incoming relational messages
4. W₀ is applied only during inference, not training, for computational efficiency
</div>

??? question "Show Answer"
    The correct answer is **C**. Without a separate self-loop weight W₀, a node's own features would be mixed with its relational neighbors under some relation type r — making the node's identity representation dependent on an arbitrary relation choice. The dedicated W₀ allows the model to learn how much of the node's current representation to preserve vs. update, independently from how messages flow across each relation. This is analogous to the skip connection in ResNets: the self term carries the "what I already know" signal.

    **Concept Tested:** R-GCN (Relational GCN), Relation-Specific Weights

---

#### Question 10

When converting a relational database to a heterogeneous graph for GNN processing, which database elements typically become node types and which become edge types?

<div class="upper-alpha" markdown>
1. Tables become edge types; rows become node types; foreign keys become node features
2. All database elements become a single homogeneous node set with typed node features
3. Column names become node types; rows become edges; primary keys become node features
4. Tables become node types (one node per row, one node type per table); foreign key relationships become edge types connecting nodes across tables
</div>

??? question "Show Answer"
    The correct answer is **D**. In the relational-to-graph transformation: each table \(T\) with schema \((\text{col}_1, \ldots, \text{col}_k)\) becomes a node type, each row \(t \in T\) becomes a node of type \(T\) with features \((\text{col}_1(t), \ldots, \text{col}_k(t))\), and each foreign key constraint \(\text{FK}(T_1.\text{col} \to T_2.\text{pk})\) becomes a directed edge type between node types \(T_1\) and \(T_2\). This relational graph construction is the foundation of Relational Deep Learning, and heterogeneous GNNs are then applied to this multi-table graph structure.

    **Concept Tested:** Heterogeneous GNN, Meta-Path
