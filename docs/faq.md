# Graph Neural Networks — Frequently Asked Questions

---

## Getting Started

### What is this textbook about?

This textbook is a rigorous, intuition-first treatment of machine learning on graphs. It covers the full arc from classical graph algorithms — PageRank, label propagation, community detection — through modern **Graph Neural Networks (GNNs)** such as GCN, GraphSAGE, GAT, and GIN, to frontier topics including Graph Transformers, Knowledge Graph Foundation Models, and LLM+GNN integration. Every chapter pairs mathematical derivation with complete, runnable PyTorch code and interactive MicroSim visualizations. The goal is a single self-contained resource that takes a motivated reader from prerequisite mathematics all the way to the research frontier.

### Who is this textbook designed for?

The primary audiences are: graduate students in CS, data science, computational biology, or any field with relational data; advanced undergraduates who have finished an introductory ML course; practitioners building recommendation engines, fraud detection systems, or drug discovery pipelines; and researchers entering the graph ML field who want a unified reference. If you already know what a GNN is but struggle to explain *why* sum aggregation outperforms mean aggregation, or how to choose between GCN and GIN for your task, this textbook is for you.

### What prerequisites do I need?

**Required:** Python proficiency, NumPy/Pandas familiarity, linear algebra (matrix multiplication, eigenvalues, SVD), probability and statistics, and ML fundamentals (gradient descent, backpropagation). **Not required upfront:** PyTorch, graph theory, or PyTorch Geometric — these are taught from scratch in Chapter 0. If you can implement a multilayer perceptron in PyTorch and explain why gradient descent converges, you have enough background to begin. If you need to build that foundation first, the [Machine Learning Textbook](https://anvithpothula.github.io/machine-learning-textbook/) covers supervised learning, neural networks, backpropagation, and optimization from scratch. See [Chapter 0: Math and Programming Prerequisites](chapters/00-math-prerequisites/index.md) for a full review.

### Do I need to know graph theory before starting?

No. Graph theory is introduced from the ground up in Chapter 1, with definitions of nodes, edges, adjacency matrices, degree distributions, paths, and connected components. The first two chapters deliberately assume no prior graph knowledge. Chapter 0 reviews the linear algebra that makes graph operations tractable (matrix multiplication, eigendecomposition), so by Chapter 2 you have the full mathematical vocabulary. See [Chapter 1: Introduction to Graphs and Networks](chapters/01-intro-to-graphs/index.md) for the starting point.

### How is the textbook organized?

The textbook spans 27 chapters across 6 parts. **Part 0** (Chapters 0–1) covers prerequisites. **Part 1** (Chapters 2–5) covers classical graph analysis — features, PageRank, node embeddings, label propagation. **Part 2** (Chapters 6–11) covers GNN architectures and theory. **Part 3** (Chapters 12–14) covers Knowledge Graphs. **Part 4** (Chapters 15–19) covers heterogeneous graphs and applications. **Part 5** (Chapters 20–22) covers scalability and generation. **Part 6** (Chapters 23–26) covers frontier topics. Each part is largely self-contained; practitioners can start at Chapter 6 after skimming the prerequisites.

### How long does it take to complete the textbook?

A graduate student reading carefully, completing exercises, and running code should budget approximately 30–40 hours per part, or roughly 150–200 hours for the full textbook. If you already have strong graph theory and ML backgrounds, you can move through Parts 0–1 quickly. A practitioner focused on GNN implementation (Chapters 6–8) without full theoretical depth can reach functional competency in 20–30 hours. The chapter structure — motivation, intuition, math, code, exercises — is designed to support both deep reading and targeted reference lookup.

### What programming framework does the textbook use?

All code examples use **PyTorch** with **PyTorch Geometric (PyG)**. Every snippet is self-contained and runnable. Chapter 0 teaches PyTorch tensors and automatic differentiation from scratch; later chapters progressively introduce PyG's `Data`, `DataLoader`, `MessagePassing`, and pre-built layer classes. The Open Graph Benchmark (OGB) evaluation pipeline is also used throughout for reproducible comparisons. No other graph frameworks (DGL, Spektral, JAX) are used.

### What is a MicroSim, and how do I use it?

A **MicroSim** is an interactive p5.js visualization embedded in each chapter. MicroSims let you manipulate parameters and observe the algorithm's behavior in real time — for example, you can adjust the teleportation probability in a PageRank simulation and watch the stationary distribution change, or change the number of GNN layers and observe how node representations evolve. They appear in the chapter text as embedded interactive widgets; just click or drag to interact. No installation is required; they run in the browser.

### What datasets are used as running examples?

Two datasets appear consistently throughout. The **Karate Club graph** (Zachary 1977) — 34 nodes, 78 edges — is used in Chapters 1–9 for all illustrative examples; it is always available via `networkx.karate_club_graph()`. For performance benchmarking from Chapter 6 onward, the **ogbn-arxiv** dataset from the Open Graph Benchmark (170,000 nodes, 1.1 million edges) is the primary citation-network benchmark. Additional datasets introduced in specific chapters include ogbg-molhiv (molecular graphs), FB15k-237 (knowledge graph completion), MovieLens-1M (recommendation), and RelBench (relational databases).

### What topics are explicitly out of scope?

The textbook intentionally does not cover: reinforcement learning on graphs, the full graph signal processing literature beyond the GCN spectral derivation, streaming and online graph algorithms, hardware-level GPU kernel optimization, GNN libraries other than PyTorch Geometric, continuous-time neural ODEs on graphs (beyond a brief mention), and quantum graph algorithms. If you need RL on graphs or deep GSP theory, companion resources are suggested in the Further Reading sections.

### Should I read every chapter in order?

Not necessarily. The chapters are designed with a dependency graph in mind, and the prerequisites listed at the top of each chapter make the DAG explicit. Readers with strong ML backgrounds and some graph familiarity can start at Chapter 6 after reading Chapter 0 for PyG setup. Knowledge Graph readers can jump to Chapter 12 after Chapter 6. The Chapter 0 prerequisites page is the only one that is strongly recommended for everyone.

### What exercises are included, and how are they structured?

Each chapter contains 12 exercises distributed across all 6 levels of Bloom's Taxonomy: 2 Remember questions (recall definitions and key facts), 2 Understand questions (explain mechanisms in your own words), 2 Apply questions (implement or use the method), 2 Analyze questions (compare architectures or decompose trade-offs), 2 Evaluate questions (assess when a method is appropriate), and 2 Create questions (design novel systems or extensions). This structure ensures you develop both conceptual depth and practical skill.

### Are there companion resources I should use alongside this textbook?

For readers who need to build or refresh their general ML foundation before starting, the [Machine Learning Textbook](https://anvithpothula.github.io/machine-learning-textbook/) by the same author covers supervised and unsupervised learning, neural networks, optimization, and evaluation from the ground up — at the same level of interactivity as this textbook. For graph-specific companions: *Graph Representation Learning* by William Hamilton (free online) is the closest textbook in scope, useful for alternative derivations. *Network Science* by Barabási (free online) provides deep background on complex network properties. The PyTorch Geometric documentation is the primary coding reference. For concise formula lookup, the CS224W Course Notes PDF is useful. These are linked in the course description for convenience.

### What benchmarks does the textbook report results for?

Benchmark results appear in every chapter and use the most recent published numbers available through 2024–2025. Primary benchmarks include ogbn-arxiv, ogbn-products, and ogbg-molhiv from the Open Graph Benchmark; Cora, CiteSeer, and PubMed for classic citation graphs; TUDatasets for graph classification; FB15k-237 and WN18RR for knowledge graph completion; MovieLens-1M for recommendation; ZINC for molecular property prediction; and RelBench for relational deep learning.

---

## Core GNN Concepts

### What is a Graph Neural Network?

A **Graph Neural Network (GNN)** is a parameterized function that takes a graph's structure (adjacency matrix) and node features as input and produces node, edge, or graph-level representations as output. The key distinguishing property is that these representations are produced by a function that respects the graph's topology — a node's representation encodes information from its local neighborhood — and the function generalizes inductively to graphs not seen during training. GNNs subsume shallow embedding methods (which are purely transductive) and outperform them when node features are available or when generalization to new graphs is required. See [Chapter 6: GNN Foundations](chapters/06-gnn-foundations/index.md).

### What is message passing?

**Message passing** is the computational mechanism underlying most GNNs. In each layer, every node sends a "message" to its neighbors, neighbors aggregate those messages, and each node updates its own representation by combining the aggregated message with its current state. Formally, one layer computes: (1) a message function generating a vector per edge, (2) an aggregation function combining incoming messages (typically sum, mean, or max), and (3) an update function merging the aggregated message with the node's previous representation. After $K$ layers, each node's representation encodes its $K$-hop neighborhood. See [Chapter 6: GNN Foundations](chapters/06-gnn-foundations/index.md).

### What is the difference between GCN, GraphSAGE, GAT, and GIN?

All four are instances of the message passing framework but differ in how they aggregate neighbor information. **GCN** (Graph Convolutional Network) uses fixed symmetric normalization: $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H W$, weighting neighbors by degree. **GraphSAGE** separates the node's own representation from aggregated neighbors and concatenates them, supporting multiple aggregators and inductive generalization. **GAT** (Graph Attention Network) learns data-dependent attention weights over neighbors, so each edge's contribution is input-dependent rather than fixed. **GIN** (Graph Isomorphism Network) uses sum aggregation with an $\varepsilon$-scaled self-loop, which is provably as expressive as the WL test. See [Chapter 6](chapters/06-gnn-foundations/index.md), [Chapter 7](chapters/07-gnn-design-space/index.md), and [Chapter 10](chapters/10-powerful-gnns/index.md).

### Why does aggregation function choice matter?

The choice of aggregation function directly determines what structural differences a GNN can distinguish. **Sum aggregation** captures the full multiset of neighbor features — it can tell apart a node with two neighbors both having feature value 1 from a node with one neighbor having feature value 2. **Mean aggregation** normalizes by degree, losing count information. **Max aggregation** is insensitive to the number of neighbors with the same feature. For graph classification and expressiveness-sensitive tasks, sum aggregation is strictly more powerful. For tasks where relative proportions matter more than absolute counts, mean can generalize better. See [Chapter 9: Theory of GNNs](chapters/09-gnn-theory/index.md) for the formal proof.

### What is the Weisfeiler-Lehman (WL) test?

The **Weisfeiler-Lehman (WL) test** is a classical graph isomorphism test that iteratively refines node "color" labels by hashing a node's current label with the multiset of its neighbors' labels. Two graphs are declared non-isomorphic if at any iteration they produce different color histograms. The 1-WL test (standard WL) is provably the expressive power ceiling for any GNN using sum/mean/max aggregation over neighborhoods — a GNN cannot distinguish two graphs that the 1-WL test deems identical. This connection, formalized in 2019 by Xu et al., gives us a precise characterization of what GNNs can and cannot learn. See [Chapter 9: Theory of GNNs](chapters/09-gnn-theory/index.md).

### What does "expressiveness" mean for a GNN?

**Expressiveness** refers to the set of graph structural differences a GNN can distinguish in its output representations. A GNN is at most as expressive as the 1-WL test: if two graphs are indistinguishable by 1-WL (they look the same to repeated neighborhood hashing), no standard message-passing GNN can distinguish them either. For example, a regular graph (every node has the same degree) stumps standard GNNs — all nodes produce identical representations. GIN achieves the upper bound of 1-WL expressiveness; GCN and GraphSAGE are strictly less expressive. Higher-order GNNs based on $k$-WL tests can go beyond 1-WL but at exponential cost. See [Chapter 9](chapters/09-gnn-theory/index.md) and [Chapter 10](chapters/10-powerful-gnns/index.md).

### What is over-smoothing in GNNs?

**Over-smoothing** occurs when a GNN has too many layers, causing node representations to converge to the same indistinguishable vector regardless of the node's identity. After $K$ layers, a node's representation is a function of its $K$-hop neighborhood; on most graphs, $K$-hop neighborhoods quickly expand to cover the entire graph. When that happens, all nodes share the same neighborhood context, and message passing becomes a low-pass filter that averages everything to the same value. This degrades performance on tasks that require distinguishing nodes. Mitigation strategies include residual connections, PairNorm, DropEdge, and the APPNP decoupling approach. See [Chapter 9: Theory of GNNs](chapters/09-gnn-theory/index.md).

### What is over-squashing in GNNs?

**Over-squashing** is a distinct pathology from over-smoothing. It occurs when long-range information from exponentially many nodes must be compressed into a fixed-size representation through a bottleneck node with limited connectivity. The information from distant parts of the graph is effectively "squashed" into too few dimensions, causing gradient vanishing and poor performance on tasks requiring long-range reasoning. Over-squashing is structural — it depends on the graph's topology, particularly on nodes with low **graph curvature** or acting as bridges. Graph rewiring and virtual node augmentation are common remedies. See [Chapter 9: Theory of GNNs](chapters/09-gnn-theory/index.md).

### What is a Graph Transformer?

A **Graph Transformer** augments the standard Transformer's self-attention mechanism with graph structural information so that attention is aware of graph topology rather than treating all nodes as unordered sequences. Key architectures include **Graphormer** (Microsoft, 2021), which adds centrality encoding and spatial bias to attention scores based on shortest-path distances; **SAN** (Spectral Attention Network), which uses Laplacian eigenvectors as positional encodings; and **GPS** (General, Powerful, Scalable), which combines local MPNN layers with global Transformer layers. Graph Transformers partially overcome the long-range limitation of message-passing GNNs. See [Chapter 11: Graph Transformers](chapters/11-graph-transformers/index.md).

### How does a GNN differ from a standard Transformer applied to nodes?

A standard **Transformer** treats its input as a fully-connected set — every token attends to every other token with learned weights and no structural prior. Applied naively to a graph, it ignores edges entirely. A GNN, by contrast, restricts information flow to the edges that exist in the graph, embedding the topology into the computation. Graph Transformers bridge the two: they use attention over all nodes (like a Transformer) but inject graph structure through positional encodings, spatial biases, or masking so that graph connectivity still shapes what information is exchanged. For large sparse graphs, full attention is also computationally prohibitive without approximation. See [Chapter 11: Graph Transformers](chapters/11-graph-transformers/index.md).

### What is the difference between transductive and inductive learning?

In **transductive learning**, the model is trained and tested on the same fixed graph — test nodes are present during training (without labels), so their structural context is used. In **inductive learning**, the model is trained on one set of graphs or subgraphs and deployed on entirely new, unseen graphs. GCN is transductive: it requires the full adjacency matrix at training time. GraphSAGE is inductive: it learns aggregation functions that operate on any local neighborhood and can embed nodes in graphs never seen during training, including newly added nodes. For dynamic or multi-graph settings, inductive methods are essential. See [Chapter 7: GNN Design Space](chapters/07-gnn-design-space/index.md).

### What is the receptive field of a GNN?

The **receptive field** of a $K$-layer GNN is the $K$-hop neighborhood of a node — the set of all nodes reachable within $K$ hops. Each additional layer expands the receptive field by one hop. This is analogous to how the receptive field of a CNN grows with depth, but on graphs, the expansion is irregular and can grow exponentially on high-degree graphs. A 2-layer GCN on ogbn-arxiv, for instance, has a receptive field that can encompass thousands of nodes. This exponential expansion motivates neighbor sampling in scalable training. See [Chapter 6](chapters/06-gnn-foundations/index.md) and [Chapter 20: Scaling GNNs](chapters/20-scaling-gnns/index.md).

### How does GNN layer depth affect performance?

More layers give each node access to a larger neighborhood, which is beneficial when the task requires integrating information from distant nodes. However, performance on most benchmarks peaks at 2–3 layers. The reasons are: over-smoothing (deep stacks wash out local structure), over-squashing (bottleneck nodes lose distant signals), and overfitting (more parameters with fewer training signals per node). Architectures like Jumping Knowledge Networks, residual connections, and APPNP address this by aggregating representations from multiple layers or decoupling propagation depth from transformation depth. See [Chapter 7: GNN Design Space](chapters/07-gnn-design-space/index.md).

### What is neighborhood aggregation versus graph-level readout?

**Neighborhood aggregation** operates at the node level — each node collects and combines information from its neighbors to update its own representation. **Graph-level readout** (pooling) operates across all nodes to produce a single vector representing the entire graph. Common readout functions include global sum, mean, or max pooling; learnable hierarchical pooling methods like **DiffPool** and **MinCutPool** cluster nodes progressively into coarser representations. Readout is needed for graph classification and regression tasks. The choice of readout function affects expressiveness: global sum pooling is more expressive than global mean pooling for the same reasons sum aggregation outperforms mean at the node level. See [Chapter 7](chapters/07-gnn-design-space/index.md).

### What is a skip connection in a GNN, and why does it help?

A **skip connection** (residual connection) adds a node's representation from layer $k-1$ directly to the output of layer $k$: $\mathbf{h}_v^{(k)} = \sigma(W \cdot \text{AGGREGATE}(\ldots)) + \mathbf{h}_v^{(k-1)}$. This gives gradients a direct path back to earlier layers, mitigating vanishing gradients in deep networks. In the context of GNNs, residual connections also partially mitigate over-smoothing by preserving local node identity across layers — the node always retains some of its original representation, even when its neighborhood signals have averaged out. **Jumping Knowledge Networks** generalize this idea by concatenating representations from all layers before the final prediction. See [Chapter 7: GNN Design Space](chapters/07-gnn-design-space/index.md).

### What is the Graph Isomorphism Network (GIN)?

**GIN** (Graph Isomorphism Network, Xu et al. 2019) is the GNN architecture that achieves the maximum expressive power attainable under the 1-WL framework. Its key design is: $\mathbf{h}_v^{(k)} = \text{MLP}^{(k)}\!\left((1+\varepsilon^{(k)}) \cdot \mathbf{h}_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}\right)$. The $\varepsilon$ parameter ensures the node's own features are not overwhelmed by its neighbors, and the MLP replaces simple linear transformation to make the function injective over multisets. GIN matches 1-WL on every pair of non-isomorphic graphs that 1-WL can distinguish. See [Chapter 10: Designing Powerful Encoders](chapters/10-powerful-gnns/index.md).

### How does attention work in a Graph Attention Network (GAT)?

In a **GAT**, the attention coefficient $\alpha_{vu}$ between nodes $v$ and $u$ is computed by a shared learnable scoring function applied to the concatenation of their feature vectors, followed by softmax normalization over all of $v$'s neighbors. The aggregation then becomes a weighted sum: $\mathbf{h}_v^{(k)} = \sigma\!\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} \mathbf{W} \mathbf{h}_u^{(k-1)}\right)$. **Multi-head attention** runs this independently with $K$ separate weight matrices and concatenates (or averages) the results, stabilizing learning. GAT attention is input-dependent — the same graph edge can have different weights for different input feature configurations. See [Chapter 7: GNN Design Space](chapters/07-gnn-design-space/index.md).

### What four tasks can GNNs solve?

GNNs support four main prediction levels. **Node classification** predicts a label for each node (e.g., classifying paper topics in a citation graph). **Link prediction** predicts whether an edge exists or will exist between two nodes (e.g., friend recommendation, knowledge graph completion). **Graph classification** assigns a single label to an entire graph (e.g., predicting molecular toxicity). **Graph regression** predicts a continuous value for a graph (e.g., predicting a molecule's HOMO-LUMO gap). Each task requires different output heads and loss functions but the same underlying GNN encoder.

### What is homophily, and how does it affect GNN performance?

**Homophily** is the tendency for connected nodes to share the same label — birds of a feather flock together. In high-homophily graphs (social networks, citation graphs), message passing is highly effective because neighbors' features are informative about a node's own label. In **heterophilic** graphs (where connected nodes tend to have different labels, as in some dating or interaction networks), standard GNNs that aggregate neighbor features can hurt performance. This is why understanding your graph's homophily before selecting a GNN architecture matters. Some architectures are specifically designed for heterophilic settings. See [Chapter 9: Theory of GNNs](chapters/09-gnn-theory/index.md).

### What is self-supervised learning on graphs?

**Self-supervised learning (SSL)** on graphs trains representations without node labels by defining pretext tasks from the graph structure itself. Two main paradigms: **contrastive methods** (e.g., Deep Graph Infomax, GraphCL) create two augmented views of a graph and train the encoder to produce similar representations for the same node/graph across views while pushing apart different ones; **predictive methods** mask parts of the graph (node features, edges) and train the model to reconstruct them. SSL is critical when labeled data is scarce, enabling pre-training on large unlabeled graphs followed by fine-tuning on small labeled sets. See [Chapter 23: Advanced GNN Topics](chapters/23-llm-gnn/index.md).

### What is GraphSAGE's sampling strategy?

**GraphSAGE** (Hamilton et al. 2017) introduced inductive representation learning with a key practical innovation: during training, instead of aggregating over a node's full neighborhood (which can be enormous), it samples a fixed number of neighbors at each hop. For a two-layer GraphSAGE with 25 first-hop neighbors and 10 second-hop neighbors, each node's mini-batch involves at most 250 neighbor lookups regardless of graph size. This bounded computational cost makes GraphSAGE tractable on graphs with millions of nodes and high-degree hubs. See [Chapter 7: GNN Design Space](chapters/07-gnn-design-space/index.md) and [Chapter 20: Scaling GNNs](chapters/20-scaling-gnns/index.md).

---

## Technical Details

### What is the Graph Laplacian, and why does it matter for GNNs?

The **Graph Laplacian** is $L = D - A$, where $D$ is the diagonal degree matrix and $A$ is the adjacency matrix. The **normalized Laplacian** is $\tilde{L} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$. Its eigendecomposition $L = U \Lambda U^\top$ defines the **graph Fourier transform** — multiplying a signal by $U^\top$ projects it into the spectral domain. GCN's propagation rule is derived by truncating and approximating a spectral filter in this domain, yielding the familiar $\hat{A} H W$ form. Understanding the Laplacian is essential for spectral GNNs and for understanding why GNNs act as low-pass filters on graph signals. See [Chapter 6: GNN Foundations](chapters/06-gnn-foundations/index.md).

### How is GCN's propagation rule derived from spectral graph theory?

Starting from spectral graph convolution $g_\theta \star x = U g_\theta(\Lambda) U^\top x$, which is $O(N^2)$ and non-spatially-local, Kipf & Welling (2017) applied a chain of approximations: (1) truncate spectral filters to 1st-order Chebyshev polynomials ($g_\theta \approx \theta_0 + \theta_1 \tilde{L}$), (2) set $\theta = \theta_0 = -\theta_1$ to reduce parameters, yielding $g_\theta \star x \approx \theta (I + D^{-1/2} A D^{-1/2}) x$, and (3) apply a **renormalization trick** ($\tilde{A} = A + I$, $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$) for numerical stability. The result is the single-layer update $H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$. Each approximation step trades theoretical exactness for computational tractability. See [Chapter 6: GNN Foundations](chapters/06-gnn-foundations/index.md).

### What is the difference between spectral and spatial GNN approaches?

**Spectral GNNs** define graph convolutions in the Fourier domain via the Laplacian's eigenvectors — they operate on the entire graph and require the graph to be fixed (transductive). **Spatial GNNs** define message passing directly in terms of node neighborhoods — they are naturally inductive, transferable across graphs, and computationally local. GCN has both interpretations: it is derived spectrally but can be implemented spatially. In practice, spatial approaches (GraphSAGE, GAT, GIN) dominate modern research because they generalize across graphs. See [Chapter 6: GNN Foundations](chapters/06-gnn-foundations/index.md).

### What are positional encodings in Graph Transformers, and why are they needed?

Standard message-passing GNNs are inherently permutation-equivariant — they treat nodes without positional identity. Transformers require positional information to know *where* each element is. In Graph Transformers, **positional encodings** inject structural position by encoding each node's location in the graph. Common choices: **Laplacian PE** (the $k$ smallest eigenvectors of the normalized Laplacian, capturing global structural position) and **random walk structural encoding** (diagonal entries of the $k$-step random walk matrix, capturing local structure). These are concatenated or added to node features before the Transformer layers. Laplacian eigenvectors are sign-ambiguous, requiring sign-invariant or sign-equivariant processing networks. See [Chapter 11: Graph Transformers](chapters/11-graph-transformers/index.md).

### What is permutation invariance vs. permutation equivariance?

A function $f$ is **permutation invariant** if $f(PX, PAP^\top) = f(X, A)$ for any permutation matrix $P$ — the output does not change when nodes are relabeled. This is required for graph-level tasks (classifying a graph regardless of how its nodes are numbered). A function is **permutation equivariant** if $f(PX, PAP^\top) = P f(X, A)$ — the output transforms consistently with the input relabeling. This is required for node-level tasks (each node's representation should transform with the node). Message passing GNNs are permutation equivariant by construction; global pooling converts equivariant node representations into invariant graph representations. See [Chapter 9: Theory of GNNs](chapters/09-gnn-theory/index.md).

### How does the GCN handle graphs without node features?

When no node features are available, a common strategy is to initialize node features as the identity matrix $I_N$ (one-hot node identifiers) or a degree vector or a constant vector of ones. Using $I_N$ makes GCN equivalent to a spectral embedding method; using constants means GCN relies entirely on structural information. For large graphs, one-hot initialization is memory-prohibitive, and structural features (degree, clustering coefficient, PageRank score) are used instead. This choice has a large effect on downstream performance and should be treated as a hyperparameter. See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

### What is DropEdge and how does it mitigate over-smoothing?

**DropEdge** randomly removes a fraction of edges from the graph during each training iteration, analogous to Dropout for activations. By removing edges, it reduces the neighborhood size each node aggregates from, effectively increasing the model's diversity of gradient signals across training steps. It also acts as a data augmentation method, making the model more robust to missing edges at test time. DropEdge does not eliminate over-smoothing in very deep networks but delays its onset and improves generalization. It is typically applied only during training, not inference. See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

### What is PairNorm?

**PairNorm** is a normalization technique designed specifically to counteract over-smoothing. After aggregation, it rescales node representations so that the average pairwise squared distance between node vectors remains constant across layers. Standard batch normalization can inadvertently accelerate over-smoothing by centering all nodes toward the same mean; PairNorm centers and then spreads representations to preserve inter-node distances. It is architecture-agnostic and can be plugged into any message-passing GNN. See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

### What loss function is used for node classification vs. link prediction?

For **node classification**, the standard loss is **cross-entropy** between predicted class probabilities (softmax output) and true one-hot labels, computed only over labeled training nodes. For **link prediction**, the standard loss is **binary cross-entropy** with negative sampling: for each observed edge $(u,v)$ (positive example), sample several non-edges (negative examples) and train the decoder — typically an inner product or MLP of node embeddings — to score positives higher than negatives. The number of negative samples per positive (typically 1–5) is a hyperparameter affecting training speed and false-negative contamination. See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

### What is the virtual node augmentation technique?

A **virtual node** is a synthetic node added to the graph and connected to every real node. During message passing, it acts as a global communication channel: in one layer, every node sends information to the virtual node; in the next, the virtual node sends a global summary back to every node. This sidesteps the depth limitations of standard message passing for long-range tasks — information can travel from any node to any other node in just two hops via the virtual node. It is widely used in molecular graph tasks and competitive with expensive higher-order methods. See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

### How do train/val/test splits differ for transductive vs. inductive settings?

In the **transductive** setting (e.g., node classification on a single large graph), *all* nodes are present in the graph during training — only their labels are withheld for validation and test. The model sees the full adjacency matrix and all node features, even for test nodes, so test nodes' structural context is used during training. In the **inductive** setting, test graphs or subgraphs are completely held out — the model must generalize to structures it has never seen. This distinction critically affects how data leakage is defined and how models should be evaluated. See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

### What is the Open Graph Benchmark (OGB)?

The **Open Graph Benchmark** is a standardized collection of large-scale graph datasets with fixed train/val/test splits, evaluation metrics, and a leaderboard. It was created to prevent the data leakage and cherry-picking that plagued early GNN papers using Cora and Citeseer. Key datasets: **ogbn-arxiv** (node classification, 170K nodes), **ogbn-products** (node classification, 2.4M nodes), **ogbg-molhiv** (graph classification, 41K molecules), **ogbl-collab** (link prediction, 235K nodes). All benchmark results in this textbook use the OGB evaluation pipeline for reproducibility. See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

### What is the APPNP architecture?

**APPNP** (Approximate Personalized Propagation of Neural Predictions) decouples feature transformation from propagation: it first transforms node features with a standard MLP to get initial predictions $Z^{(0)}$, then propagates those predictions using an approximate **Personalized PageRank** scheme $Z^{(k)} = (1-\alpha) \hat{A} Z^{(k-1)} + \alpha Z^{(0)}$, where $\alpha$ is the teleportation probability. The teleportation term prevents representations from fully diffusing to equilibrium (over-smoothing) because each node always retains a fraction $\alpha$ of its initial local prediction. APPNP shows that long-range propagation can be added cheaply and without over-smoothing as a post-processing step. See [Chapter 9: Theory of GNNs](chapters/09-gnn-theory/index.md).

### What is the Simple Graph Convolution (SGC)?

**SGC** (Simple Graph Convolution, Wu et al. 2019) removes all nonlinear activations from a GCN and shows that the $K$-layer GCN collapses to a single linear layer $Y = \text{softmax}(\hat{A}^K X W)$, where $\hat{A}^K$ is the normalized adjacency matrix raised to the $K$-th power. This is equivalent to pre-computing the $K$-step diffusion of features (a one-time, graph-dependent preprocessing step) and then training a logistic regression. SGC achieves competitive accuracy on many benchmarks at a fraction of the training cost, suggesting that much of GCN's power comes from the graph propagation, not the nonlinear layers. It is a useful strong baseline. See [Chapter 9: Theory of GNNs](chapters/09-gnn-theory/index.md).

### What is the Jumping Knowledge Network?

A **Jumping Knowledge Network** (JK-Net) addresses the problem that different nodes in a graph benefit from different neighborhood sizes. Rather than using only the final layer's representations, JK-Net aggregates representations from all layers for each node before prediction: $\mathbf{h}_v = \text{AGG}(\mathbf{h}_v^{(1)}, \mathbf{h}_v^{(2)}, \ldots, \mathbf{h}_v^{(K)})$, where the aggregator can be concatenation, max, or LSTM. Nodes in dense regions (which suffer from over-smoothing sooner) implicitly use shallower representations; nodes that need long-range context use deeper representations. The name refers to these "jumping" skip connections from every layer directly to the output. See [Chapter 7: GNN Design Space](chapters/07-gnn-design-space/index.md).

### What hyperparameters matter most when training a GNN?

The most impactful hyperparameters are: **number of layers** (2–3 is almost always optimal for standard benchmarks), **hidden dimension** (64–512), **dropout rate** (0.0–0.5, tuned per dataset), **learning rate** (typically 1e-3 with Adam), **aggregation function** (task-dependent), and **normalization scheme** (batch vs. layer norm). Less intuitive but often high-impact: whether to use **skip connections**, the **graph sampling strategy** (full-batch vs. neighbor sampling), and **feature normalization** (normalizing node features before training often matters as much as architecture choice). See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

---

## Knowledge Graphs and Applications

### What is a Knowledge Graph?

A **Knowledge Graph (KG)** is a directed, multi-relational graph where nodes represent entities (people, places, concepts) and typed edges represent relations between them, stored as triples $(head, relation, tail)$. For example, (Einstein, bornIn, Ulm) or (Python, isA, ProgrammingLanguage). KGs encode world knowledge in a structured, queryable form. Large KGs like Freebase, Wikidata, and domain-specific medical KGs (DrugBank, OMIM) are used for question answering, drug discovery, and recommendation. They are inherently incomplete — **KG completion** (predicting missing triples) is the central learning task. See [Chapter 12: Knowledge Graph Embeddings](chapters/12-knowledge-graph-embeddings/index.md).

### What are knowledge graph embedding methods?

**KG embedding methods** assign entities and relations to continuous vectors (or geometric objects) in a low-dimensional space such that a scoring function $f(h, r, t)$ assigns high scores to true triples and low scores to false ones. The most influential families: **TransE** models relations as translations ($\mathbf{h} + \mathbf{r} \approx \mathbf{t}$, good for hierarchical/compositional relations); **DistMult** uses bilinear scoring ($\mathbf{h}^\top \text{diag}(\mathbf{r}) \mathbf{t}$, handles symmetric relations); **ComplEx** extends to complex-valued embeddings to handle antisymmetry; **RotatE** models relations as rotations in complex space, capturing symmetry, antisymmetry, inversion, and composition patterns. See [Chapter 12: Knowledge Graph Embeddings](chapters/12-knowledge-graph-embeddings/index.md).

### What relation patterns must a KG embedding model handle?

Four key patterns determine which KG embedding geometry is needed. **Symmetry:** $r(x,y) \Rightarrow r(y,x)$ (e.g., "isSiblingOf"). **Antisymmetry:** $r(x,y) \Rightarrow \neg r(y,x)$ (e.g., "isParentOf"). **Inversion:** $r_1(x,y) \Rightarrow r_2(y,x)$ (e.g., "hasEmployee" and "worksFor"). **Composition:** $r_1(x,y) \land r_2(y,z) \Rightarrow r_3(x,z)$ (e.g., motherOf + fatherOf → grandmotherOf). TransE handles antisymmetry and composition but not symmetry. DistMult handles symmetry but not antisymmetry. RotatE handles all four, making it the most expressive single-space model among the classics. Choosing the wrong model for your KG's dominant relation patterns leads to poor link prediction performance. See [Chapter 12](chapters/12-knowledge-graph-embeddings/index.md).

### What is multi-hop reasoning over knowledge graphs?

**Multi-hop reasoning** answers complex queries that require traversing multiple edges — for example, "Who are the advisors of researchers who published papers on GNNs?" cannot be answered with a single triple lookup. Query2Box (Ren et al. 2020) addresses this by embedding conjunctive queries as **axis-aligned hyperboxes** in entity space and training the model to map logical operators (projection, intersection) to geometric operations (box translation, box intersection). Entities inside the answer box at query time are the predicted answers. The approach handles both *existential* and *intersection* operators and generalizes to entities not seen in training paths. See [Chapter 13: Reasoning over Knowledge Graphs](chapters/13-kg-reasoning/index.md).

### What are GNNs for recommender systems?

Graph-based recommenders model the recommendation problem as **link prediction on a user-item bipartite graph**, where an edge between user $u$ and item $i$ means $u$ interacted with $i$. **LightGCN** simplifies GCN for this setting by removing feature transformation and nonlinear activation — it performs pure graph propagation over the user-item graph and takes the weighted average of representations across layers as the final embedding. **PinSage** (Pinterest, 2018) scales this to billions of nodes using random-walk-based neighborhood sampling. GNN recommenders consistently outperform matrix factorization baselines by capturing higher-order collaborative filtering signals (e.g., user-item-user-item patterns). See [Chapter 16: GNNs for Recommender Systems](chapters/16-recommender-systems/index.md).

### How are GNNs used in drug discovery?

GNNs are applied at multiple stages of the drug discovery pipeline. **Molecular property prediction:** a drug molecule is a graph (atoms=nodes, bonds=edges with type features) and GNNs predict properties like toxicity, solubility, and binding affinity (benchmarked on ogbg-molhiv, ZINC). **Drug-drug interaction prediction:** a bipartite graph of drugs and proteins, or a KG of known interactions, is used to predict adverse combinations. **Molecule generation:** generative GNN models (GCPN, DiGress) propose new molecules with desired properties. GNNs for molecules outperform SMILES-based sequence models on many tasks because they natively capture bond topology. See [Chapter 17: Relational Deep Learning](chapters/17-relational-deep-learning/index.md) and [Chapter 21: Generative Models](chapters/21-generative-models/index.md).

### What is Relational Deep Learning (RDL)?

**Relational Deep Learning** (Fey et al. 2023) extends GNN methodology to relational databases by converting database tables and foreign key relationships into a heterogeneous graph and training GNNs directly on the resulting structure. **RelBench** provides standardized benchmark tasks over real database snapshots. The key insight is that most enterprise data is inherently relational and already encodes a graph structure — RDL automates the conversion of that structure into trainable GNN inputs without requiring manual feature engineering. It enables predictive modeling on data that previously required complex SQL-based feature pipelines. See [Chapter 17: Relational Deep Learning](chapters/17-relational-deep-learning/index.md).

### What is a Knowledge Graph Foundation Model?

A **KG Foundation Model** is a single GNN-based model that can perform link prediction (triple completion) on *any* knowledge graph, including KGs with entity and relation vocabularies never seen during training. Classical KG embedding methods (TransE, RotatE) require retraining from scratch for each new KG. Models like **ULTRA** (2023) overcome this by learning on multiple KGs simultaneously, using a relation graph to encode relational structure rather than fixed relation embeddings. Given a new KG at inference time, ULTRA generalizes zero-shot. This mirrors the foundation model paradigm in NLP — one pre-trained model, many downstream graphs. See [Chapter 14: Knowledge Graph Foundation Models](chapters/14-kg-foundation-models/index.md).

### What is a heterogeneous graph?

A **heterogeneous graph** is a graph with multiple node types and/or edge types — for example, a citation network with paper nodes, author nodes, and venue nodes, connected by "writes," "cites," and "publishedAt" edges. Standard GNNs assume a single node and edge type; they cannot leverage type distinctions. **R-GCN** (Schlichtkrull et al. 2018) handles this by using relation-specific weight matrices with basis decomposition to manage parameter count. **HAN** uses meta-path-based attention. **HGT** (Heterogeneous Graph Transformer) learns type-specific projections and attention directly. Heterogeneous graphs are ubiquitous in practice — most real databases and KGs are heterogeneous. See [Chapter 15: Heterogeneous Graphs](chapters/15-heterogeneous-graphs/index.md).

### How are GNNs used in fraud detection?

Fraud detection is inherently relational: fraudulent accounts tend to share phone numbers, devices, or IP addresses (forming clusters), and fraud rings leave distinctive subgraph patterns that isolated feature analysis misses. GNNs can detect these patterns by propagating risk signals through the transaction or account-sharing graph. A node's GNN representation encodes whether its neighbors exhibit suspicious connectivity patterns, enabling detection of fraud rings that disguise individual account features. **GraphSAGE** and **GAT** have both been applied in production fraud detection at major financial institutions. The key challenge is class imbalance and adversarial graph modification by fraudsters. See [Chapter 17: Relational Deep Learning](chapters/17-relational-deep-learning/index.md).

### What are the standard evaluation metrics for knowledge graph completion?

KG completion is evaluated by ranking all entities as candidates for the missing entity in a triple and measuring where the correct answer appears. **MRR (Mean Reciprocal Rank)** averages $1/\text{rank}$ over all test triples — high MRR means correct answers appear near the top. **Hits@K** (K=1,3,10) measures the fraction of test triples where the correct entity ranks in the top K. Both are computed under the "filtered" protocol, which removes other true triples from the ranking to avoid penalizing correct predictions. Higher is better for both metrics. See [Chapter 12: Knowledge Graph Embeddings](chapters/12-knowledge-graph-embeddings/index.md).

---

## Common Challenges and Debugging

### My GNN's validation performance degrades after 3–4 layers. What is happening?

Almost certainly **over-smoothing**. After 3–4 layers on most graphs, a node's receptive field covers a large fraction of the graph, and mean/sum aggregation washes out local structure. Remedies to try in order: (1) add residual connections; (2) apply PairNorm or batch normalization; (3) use DropEdge during training; (4) switch to APPNP or JK-Net, which decouple propagation depth from representation collapse; (5) try SGC to see if linear propagation already solves the problem. If performance is already best at 2 layers, do not force deeper architectures. See [Chapter 9: Theory of GNNs](chapters/09-gnn-theory/index.md).

### My GNN performs well on the training graph but poorly on new graphs. What should I do?

This points to overfitting or transductive brittleness. First, check whether your architecture is truly **inductive** — GCN requires the full adjacency matrix at training time and cannot generalize to new graphs by default. Switch to GraphSAGE, GAT, or GIN for inductive settings. Second, apply regularization: dropout (0.3–0.5), weight decay (1e-4 to 1e-5), and early stopping. Third, ensure your train/val/test split does not leak graph structural information — in inductive settings, test graphs must be completely held out. Finally, consider whether node features adequately describe nodes for the out-of-distribution graphs. See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

### My GNN struggles with tasks requiring long-range information. How should I address this?

Long-range information flow is limited by the **receptive field** and worsened by **over-squashing** at bottleneck nodes. Options: (1) Add a **virtual node** connected to all nodes — provides a 2-hop shortcut for any node pair. (2) Use **APPNP** to propagate features over many hops without adding trainable layers. (3) Switch to a **Graph Transformer** (GPS, Graphormer), which gives every node access to every other node via global attention. (4) Apply **graph rewiring** techniques that add shortcut edges between distant nodes with high curvature. (5) Use **positional encodings** that encode global structure. See [Chapter 9](chapters/09-gnn-theory/index.md) and [Chapter 11: Graph Transformers](chapters/11-graph-transformers/index.md).

### How do I choose between GCN, GraphSAGE, GAT, and GIN for my task?

A practical decision tree: Use **GCN** as a fast baseline for homophilic graphs where you have the full graph at training time and need a simple, well-understood model. Use **GraphSAGE** for inductive settings, dynamic graphs, or when you need to embed new nodes at inference time. Use **GAT** when you suspect that different neighbors have heterogeneous relevance and the task benefits from dynamic, data-driven weighting. Use **GIN** when expressiveness matters — graph classification, molecular tasks, or any setting where distinguishing non-isomorphic structures is important. Always benchmark GCN as a baseline: it is frequently competitive on node classification with simple homophilic graphs. See [Chapter 7: GNN Design Space](chapters/07-gnn-design-space/index.md).

### How do I scale a GNN to a graph with millions of nodes?

Full-batch training on large graphs exhausts GPU memory. Three main approaches: (1) **Neighbor sampling** (GraphSAGE-style) samples a fixed number of neighbors per node per hop, bounding the mini-batch size regardless of graph size. (2) **Cluster-GCN** partitions the graph into dense clusters and trains on one cluster per batch, reducing cross-cluster edge cuts. (3) **GraphSAINT** samples subgraphs (random walks, random edges, random nodes) with bias correction to ensure unbiased gradient estimates. For inference on very large graphs, **historical embeddings** (SIGN) precompute diffusion matrices offline and train only a final MLP online. See [Chapter 20: Scaling GNNs to Billion-Node Graphs](chapters/20-scaling-gnns/index.md).

### My GNN's loss diverges or produces NaN during training. What should I check?

Common causes: (1) **Missing normalization** — adding a self-loop ($\tilde{A} = A + I$) and symmetric normalization prevents degree-1 nodes from causing exploding activations. (2) **Learning rate too high** — try 1e-3 with Adam; if unstable, reduce to 1e-4. (3) **Unnormalized node features** — features with very different scales cause unstable gradients; normalize each feature dimension to zero mean, unit variance. (4) **Dead ReLU neurons** — if using ReLU, check for all-zero activation layers; switch to ELU or add layer normalization. (5) **Numerical issues in KG embeddings** — check that entity vectors are $\ell_2$-normalized if required by the model (e.g., RotatE uses modulus constraints). See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

### How do I know if graph structure is actually useful for my task?

Always run a **feature-only baseline** (an MLP trained on node features only, ignoring graph edges) and compare against your GNN. If the MLP matches or beats the GNN, graph structure is not adding value for this task — which could mean: (a) the task is dominated by node features, (b) homophily is low, (c) the graph is too noisy. Also run a **structure-only baseline** (GNN with constant node features or degree features). The gap between these baselines and your full GNN tells you how much each modality contributes. Skipping these baselines is a common pitfall in GNN papers. See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

### What is the neighbor explosion problem, and how is it handled?

In a $K$-layer GNN with average degree $d$, full neighborhood aggregation requires $O(d^K)$ node lookups per training node. For a graph with $d=50$ (typical for social networks) and $K=3$, that is 125,000 nodes per training example. This makes mini-batch training infeasible without sampling. Solutions: **neighbor sampling** caps this at a fixed budget per hop; **layer-wise sampling** (FastGCN) samples independently per layer; **cluster-based sampling** avoids cross-partition edges. The trade-off is bias in gradient estimation versus computational tractability. See [Chapter 20: Scaling GNNs to Billion-Node Graphs](chapters/20-scaling-gnns/index.md).

### How should I handle graphs with very high-degree nodes (hubs)?

High-degree nodes (hubs) cause two problems: (1) they dominate aggregation — their features are aggregated into many neighbors' representations, creating outsized influence; (2) neighbor sampling at hubs is expensive. Common remedies: (a) **Degree normalization** (GCN's $D^{-1/2} A D^{-1/2}$) down-weights high-degree neighbors; (b) **Top-$k$ sampling** at training time samples at most $k$ neighbors even for high-degree nodes; (c) **GAT's attention weights** can learn to down-weight hub neighbors that are not relevant; (d) for scalability, consider **PinSage's random-walk-based neighbor selection**, which implicitly weights neighbors by visit frequency rather than edge existence. See [Chapter 20: Scaling GNNs](chapters/20-scaling-gnns/index.md).

### What is data leakage in graph learning, and how do I avoid it?

**Data leakage** occurs when information about test examples is used during training. In graphs, this is subtler than in i.i.d. settings. For **link prediction**, if you split edges into train/val/test, you must also ensure test-edge endpoints' structural features (degrees, clustering coefficients) are not computed using test edges — otherwise the model sees test-time structure during training. For **node classification** in transductive settings, test nodes' features are visible but their labels must not be. The OGB benchmark's fixed splits and the explicit inductive/transductive protocol distinction were created specifically to prevent these leakages. See [Chapter 8: GNN Training](chapters/08-gnn-training/index.md).

### When should I use a Graph Transformer instead of a message-passing GNN?

Graph Transformers are worth the extra computational cost when: (1) the task requires long-range structural reasoning (e.g., predicting properties that depend on global molecular topology); (2) the graph has high diameter with important but distant node pairs; (3) you have sufficient data to train the larger model; (4) you can precompute positional encodings (Laplacian PE, RWSE) offline. For tasks where 2–3 hop neighborhoods are sufficient (most social network node classification tasks), GNNs are faster, simpler, and often competitive. Graph Transformers shine on molecular benchmarks (ZINC, PCQM4Mv2) where global topology matters. See [Chapter 11: Graph Transformers](chapters/11-graph-transformers/index.md).

---

## Advanced and Research Topics

### What is a Graph Foundation Model?

A **Graph Foundation Model** is a single pre-trained GNN that can generalize across multiple graphs, tasks, and domains — analogous to GPT-3 for text or CLIP for vision. The ambition is one model that, given any graph with any node features, produces useful representations for node classification, link prediction, or graph classification without retraining from scratch. Current approaches include **ULTRA** for KG reasoning (generalizes to unseen KGs zero-shot), **OFA** (One-For-All), which uses LLM-generated text descriptions to bridge diverse node feature spaces, and **graph in-context learning**, where examples are encoded directly in the graph structure. The field is nascent — true general-purpose graph foundation models remain an open problem. See [Chapter 14: Knowledge Graph Foundation Models](chapters/14-kg-foundation-models/index.md) and [Chapter 23: Advanced GNN Topics](chapters/23-llm-gnn/index.md).

### How do LLMs and GNNs complement each other?

**LLMs** are powerful at processing unstructured text (node descriptions, paper abstracts, entity names) and performing semantic reasoning, but they treat input as an unstructured sequence and cannot natively operate on relational structure. **GNNs** excel at capturing graph topology and propagating structural signals, but they struggle when node features are heterogeneous text. Combined approaches: (1) **LLM as feature encoder** — use an LLM to produce node embeddings from text, then feed these as features to a GNN (e.g., TAPE for citation graphs); (2) **GNN-augmented LLM** — provide GNN-derived structural context in the LLM's prompt; (3) **joint training** on text-attributed graphs. These combinations achieve state-of-the-art on most text-graph benchmarks. See [Chapter 24: LLMs + GNNs](chapters/24-advanced-gnn-topics/index.md).

### What are temporal and dynamic graphs?

A **temporal graph** is a graph that changes over time — nodes and edges are added, removed, or modified. Two main models: **snapshot models** discretize time into graph snapshots $G_1, G_2, \ldots$ and process each with a GNN, sharing parameters and using sequence models (LSTM, Transformer) across snapshots; **continuous-time models** represent each interaction as a timestamped event and update representations on the fly using neural temporal processes. Key architectures include **TGAT** (Temporal Graph Attention) and **TGN** (Temporal Graph Networks). Temporal graphs arise naturally in financial transactions, social media, and biological signaling. See [Chapter 22: Temporal and Dynamic Graphs](chapters/22-temporal-graphs/index.md).

### What is graph contrastive learning?

**Graph contrastive learning** is a form of self-supervised learning that trains node or graph encoders without labels by contrasting augmented views. Two views of the same graph (produced by randomly dropping edges, masking features, or diffusing the graph) should have similar representations; views of different graphs should be dissimilar. **Deep Graph Infomax (DGI)** maximizes mutual information between node-level and graph-level representations. **GraphCL** (You et al. 2020) applies four augmentation strategies on graph-level tasks. Contrastive pre-training followed by fine-tuning on small labeled sets achieves strong performance in low-label regimes, particularly for molecular and biological graphs. See [Chapter 23: Advanced GNN Topics](chapters/23-llm-gnn/index.md).

### What does "in-context learning" mean for graphs?

In NLP, **in-context learning** allows a model to solve new tasks from a few examples presented in the input prompt, without gradient updates. For graphs, analogous approaches encode labeled example nodes or graphs directly into the input structure and train the model to produce correct predictions for query nodes given those examples. One technique constructs a meta-graph connecting query nodes to support examples and runs message passing to transfer labels. This is an active research area motivated by the success of in-context learning in LLMs and the desire for label-efficient graph learning. See [Chapter 23: Advanced GNN Topics](chapters/23-llm-gnn/index.md).

### What are the main open problems in GNN research?

Several important problems remain unsolved as of 2025. (1) **Generalization theory:** tight generalization bounds for GNNs on real heterophilic graphs are lacking. (2) **Scalable expressiveness:** going beyond 1-WL without exponential cost. (3) **Long-range dependencies:** no single architecture cleanly solves over-squashing while remaining computationally efficient. (4) **True graph foundation models:** cross-domain generalization across heterogeneous graph types remains unsolved. (5) **Temporal graphs at scale:** efficient continuous-time models for billion-edge streams. (6) **Uncertainty quantification:** reliable confidence estimation for GNN predictions in safety-critical domains. See [Chapter 26: Conclusion](chapters/26-conclusion/index.md).

### What are subgraph GNNs, and why do they matter?

**Subgraph GNNs** extend the expressiveness of standard message-passing GNNs beyond the 1-WL limit by running the GNN not just on the original graph but on multiple subgraphs — typically node-deleted subgraphs or ego-networks. The collection of these subgraph representations is then aggregated into the final node or graph representation. This strictly increases expressiveness: some pairs of non-isomorphic graphs that are 1-WL-equivalent become distinguishable when viewed through their subgraph collections. Architectures like NGNN, OSAN, and SUN fall into this family. The computational overhead is $O(N)$ times larger than standard GNNs, making scalability an active research challenge. See [Chapter 10: Designing Powerful Encoders](chapters/10-powerful-gnns/index.md).

### What are agents on graphs?

**Agents on graphs** refers to AI agents that use graph structures to organize memory, tool use, and reasoning. Two main threads: (1) **Agent memory as graphs** — rather than flat text memories, an agent stores past observations, entities, and their relationships as a graph, enabling structured retrieval and relational reasoning over remembered context; (2) **Agents operating on graphs** — agents that plan routes in road networks, reason over knowledge graphs, or traverse web link graphs as part of task execution. GNNs are used both to process the graph-structured memory and to encode the state of the environment the agent navigates. See [Chapter 25: Agents and Graphs](chapters/25-agents-and-graphs/index.md).

### How is conformalized prediction used with GNNs?

**Conformal prediction** is a distribution-free framework for constructing prediction sets with guaranteed coverage — the true label is inside the predicted set with at least $1-\alpha$ probability, regardless of the model's architecture. For GNNs, **conformal GNNs** (Huang et al. 2023) adapt the calibration procedure to account for the non-i.i.d. structure of graph data, where nodes in the calibration set share edges with test nodes. This is important for safety-critical applications (drug property prediction, financial fraud) where knowing the model's uncertainty is as important as its point predictions. See [Chapter 23: Advanced GNN Topics](chapters/23-llm-gnn/index.md).
