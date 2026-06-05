# Glossary of Terms

#### Adam Optimizer

A first-order optimization algorithm that adapts per-parameter learning rates using running estimates of the first and second moments of the gradients, combining momentum with adaptive scaling. It often converges faster than plain gradient descent.

It is the default optimizer in most GNN training pipelines.

**Example:** A GraphSAGE model trained with Adam at learning rate \( 0.01 \) typically converges within a few hundred epochs.

See also: Gradient Descent.

#### Adjacency Matrix

A square matrix \( A \) of size \( N \times N \) for a graph with \( N \) nodes, where entry \( A_{ij} \) is nonzero (often 1) if an edge connects node \( i \) to node \( j \) and zero otherwise. It encodes graph structure algebraically.

Multiplying it by a feature matrix realizes one step of neighborhood aggregation, linking graph structure to linear algebra.

**Example:** For the Karate Club graph, \( A_{ij} = 1 \) whenever members \( i \) and \( j \) interact outside the club.

See also: Graph Laplacian, Matrix Multiplication.

#### Agent Memory as Graph

A design in which an autonomous agent stores its experiences, facts, and their relationships as a graph that it queries and updates over time, providing structured, navigable long-term memory. It organizes memory relationally rather than as flat text.

It lets agents retrieve connected context and reason over relationships among remembered items.

**Example:** A conversational agent records people and their relationships as a graph so it can later answer queries by traversing remembered connections.

See also: Tool-Use Graph, Knowledge Graph.

#### Aggregation Function

The permutation-invariant operation in a GNN that combines the multiset of messages from a node's neighbors into a single vector, using operators such as sum, mean, or max. Its invariance ensures output does not depend on neighbor ordering.

The choice of aggregator critically affects a GNN's expressive power and behavior on different graphs.

**Example:** Sum aggregation distinguishes neighborhoods with different neighbor counts, whereas mean aggregation does not.

See also: Sum Aggregation, Mean Aggregation, Max Aggregation.

#### Antisymmetry (Relation)

A relational pattern in which a relation cannot hold in both directions, so \( r(x, y) \) implies that \( r(y, x) \) is false, as for "parentOf". Expressing it requires asymmetric scoring.

A model's ability to represent antisymmetry distinguishes expressive embeddings from purely symmetric ones.

**Example:** ComplEx and RotatE can represent the antisymmetric relation "parentOf", whereas DistMult cannot.

Contrast with: Symmetry (Relation Pattern).

#### APPNP

A GNN that decouples feature transformation from propagation by first transforming node features with a neural network and then propagating the predictions using a personalized-PageRank-based scheme. This separation enables propagation over many hops without over-smoothing.

It shows that long-range propagation can be added cheaply without stacking many learnable layers.

**Example:** APPNP computes initial node predictions with a multilayer perceptron, then smooths them via approximate personalized PageRank propagation.

See also: Personalized PageRank, Simple Graph Conv (SGC), Over-Smoothing.

#### Attention Mechanism (Graph)

A mechanism that computes data-dependent weights for a node's neighbors, typically via a learnable scoring function followed by softmax normalization, so aggregation is a weighted combination reflecting learned importance. It replaces fixed aggregation weights.

It enables GNNs to focus selectively on relevant parts of a neighborhood.

**Example:** The graph attention mechanism assigns higher weight to neighbors whose features best explain the target node's label.

See also: Graph Attention Network (GAT), Softmax Function.

#### Attribute Graph

A graph whose nodes and/or edges are annotated with feature vectors describing their properties, in addition to the connectivity structure. It pairs relational structure with content information.

Attribute graphs supply the input features that GNNs combine with topology to make predictions.

**Example:** A citation graph where each paper node carries a bag-of-words feature vector of its abstract is an attribute graph.

See also: Node (Vertex), Text-Attributed Graph.

#### Automatic Differentiation

A technique for computing exact derivatives of programs by tracking elementary operations and applying the chain rule mechanically, distinct from numerical or symbolic differentiation. Reverse-mode automatic differentiation implements backpropagation.

It frees practitioners from deriving gradients by hand when building novel GNN architectures.

**Example:** PyTorch records each operation in a GNN forward pass on a tape, then replays it in reverse to obtain gradients automatically.

See also: PyTorch Autograd, Backpropagation.

#### Backpropagation

An algorithm that computes gradients of a scalar loss with respect to all network parameters by applying the chain rule backward through the computation, reusing intermediate results for efficiency. It enables gradient-based training of deep models.

In GNNs, gradients must also flow backward through the message-passing operations across the graph.

**Example:** Backpropagation through a two-layer GNN sends error signals from output predictions back through neighbor aggregations to the input feature weights.

See also: Chain Rule (Calculus), Automatic Differentiation.

#### Barabási–Albert Model

A growth model that builds a network by adding nodes one at a time, each connecting to existing nodes with probability proportional to their current degree. It produces a scale-free, power-law degree distribution.

It explains how preferential attachment alone can generate the hub structure seen in real networks.

**Example:** Simulating the Barabási–Albert model yields a few dominant hubs accumulating most connections, mimicking citation networks.

See also: Preferential Attachment, Scale-Free Network.

#### Basis Decomposition (R-GCN)

A parameter-sharing technique in R-GCN that expresses each relation's weight matrix as a learned linear combination of a small shared set of basis matrices, drastically reducing parameters. It curbs overfitting on graphs with many relations.

It makes relational GNNs tractable when the number of relation types is large.

**Example:** With basis decomposition, an R-GCN on a knowledge graph with hundreds of relations shares a handful of basis matrices among them.

See also: Relation-Specific Weights, R-GCN (Relational GCN).

#### Basis-Invariant Network

A neural module whose output is invariant to changes of orthonormal basis within an eigenspace, handling the ambiguity that eigenvectors of repeated eigenvalues are defined only up to rotation. It strengthens sign-invariance to full basis-invariance.

It is needed for robust spectral positional encodings when eigenvalues are degenerate.

**Example:** A basis-invariant network gives the same encoding regardless of which orthonormal basis is chosen for a repeated-eigenvalue subspace.

See also: Sign-Invariant Network, Laplacian Positional Encoding.

#### Batch Normalization (GNN)

A normalization technique that standardizes the activations across a batch using their running mean and variance, stabilizing and accelerating training, applied to node representations in GNNs. It reduces internal covariate shift.

Its reliance on batch statistics can interact awkwardly with variable graph and neighborhood sizes.

**Example:** Batch normalization rescales the node embeddings in a mini-batch to zero mean and unit variance before the next GNN layer.

See also: Layer Normalization (GNN), PairNorm.

#### Belief Propagation

A message-passing inference algorithm on graphical models that computes marginal distributions by exchanging probabilistic "belief" messages between nodes along edges. It is exact on trees and approximate on graphs with cycles.

It is a probabilistic ancestor of neural message passing and underlies inference in some graph models.

**Example:** Belief propagation infers the most likely category of each node in a Markov random field defined over a graph.

See also: Message Passing Neural Net, Label Propagation.

#### Betweenness Centrality

A node centrality measure equal to the fraction of all-pairs shortest paths that pass through a given node, capturing its role as a bridge in the flow of information. High values mark gatekeeper nodes.

It identifies brokers whose removal would most disrupt communication across the network.

**Example:** A person connecting two otherwise separate friend groups has high betweenness centrality.

See also: Closeness Centrality, Shortest Path.

#### BFS Strategy (node2vec)

A configuration of node2vec's biased walk that keeps the walk near its origin, repeatedly sampling immediate neighbors in a breadth-first manner. It emphasizes a node's local microscopic neighborhood.

This setting yields embeddings that reflect structural equivalence, grouping nodes by local role.

**Example:** With a small in-out parameter, node2vec walks stay local, giving two hub nodes similar embeddings even if distant.

Contrast with: DFS Strategy (node2vec).

#### Biased Random Walk

A random walk whose transition probabilities are skewed by tunable parameters rather than uniform over neighbors, steering exploration toward particular structural patterns. It generalizes the uniform walk used in DeepWalk.

Biasing the walk controls whether the resulting embeddings capture proximity or structural similarity.

**Example:** node2vec uses a biased random walk with return and in-out parameters to favor either local or far-reaching exploration.

See also: node2vec, Random Walk.

#### BigCLAM Model

A generative model for overlapping community detection that assigns each node nonnegative affiliation strengths to latent communities and predicts edge probability from shared affiliations. It is fit by maximizing the likelihood of the observed graph.

It scales overlapping community detection to large networks via a matrix-factorization-like formulation.

**Example:** BigCLAM infers that two users connect partly because they share membership in the same hobby community.

See also: Overlapping Community, Matrix Factorization (Graph).

#### Bilinear KG Model

A family of knowledge-graph embedding models that score a triple with a bilinear form \( h^\top M_r t \), where the relation supplies a matrix \( M_r \). The structure imposed on \( M_r \) defines each specific model.

Bilinear scoring gives a flexible template spanning models from DistMult to ComplEx.

**Example:** DistMult is the bilinear model in which the relation matrix \( M_r \) is constrained to be diagonal.

See also: DistMult, ComplEx.

#### Binary Cross-Entropy (LP)

The binary cross-entropy loss applied to link prediction, treating each candidate edge as a binary "exists or not" decision and penalizing the model's predicted edge probability against the true label. It trains edge scorers.

It is typically paired with negative sampling, since most node pairs are non-edges.

**Example:** A link predictor minimizes binary cross-entropy, pushing scores of observed edges toward 1 and sampled non-edges toward 0.

See also: Link Prediction, Negative Sampling (LP), Cross-Entropy Loss.

#### Bipartite Graph

A graph whose nodes split into two disjoint sets such that every edge connects a node in one set to a node in the other, with no edges within a set. Its structure encodes two interacting entity types.

Bipartite structure naturally models user-item, author-paper, and similar dual-domain relationships in recommendation.

**Example:** A user-movie rating graph is bipartite, with edges only between user nodes and movie nodes.

See also: Recommender System (Graph), Heterogeneous Graph.

#### Box Embedding (Query2Box)

A query-embedding method that represents a query's answer set as an axis-aligned box (hyperrectangle) in embedding space, with entities answering the query if they fall inside the box. Boxes naturally model sets and their intersection.

Representing answers as regions rather than points lets the model handle queries with many answers and logical conjunction.

**Example:** In Query2Box, intersecting two query boxes yields a smaller box representing entities satisfying both conditions.

See also: Query Embedding, Conjunctive Query.

#### Breadth-First Search

A graph traversal algorithm that explores nodes in order of increasing distance from a source, visiting all neighbors at one depth before proceeding to the next. It finds shortest paths in unweighted graphs.

It defines local, neighborhood-oriented exploration and informs sampling strategies in graph embeddings.

**Example:** Breadth-first search from a node enumerates its one-hop neighbors, then its two-hop neighbors, and so on.

See also: Depth-First Search, BFS Strategy (node2vec).

#### Centrality Encoding

A node feature, used in graph transformers, that encodes a node's degree or centrality so that highly connected nodes are represented distinctly. It supplies importance information that pure attention lacks.

It lets a transformer account for the influence of high-degree hubs during global attention.

**Example:** Centrality encoding adds learnable vectors indexed by a node's in-degree and out-degree to its input representation in Graphormer.

See also: Graphormer, Node Degree.

#### Chain Rule (Calculus)

The differentiation rule stating that the derivative of a composition \( f(g(x)) \) equals \( f'(g(x)) \cdot g'(x) \), extended to vectors via Jacobian products. It decomposes derivatives of nested functions into factors.

It is the mathematical foundation that makes backpropagation through layered GNNs possible.

**Example:** Differentiating a loss through stacked GNN layers multiplies the per-layer Jacobians via the chain rule.

#### Chebyshev Polynomial Conv

A spectral graph convolution that approximates the spectral filter by a truncated Chebyshev polynomial of the Laplacian, making the operation localized to a fixed number of hops and avoiding explicit eigendecomposition. It trades exact spectral filtering for efficiency.

It bridges costly spectral convolution and the efficient first-order graph convolutional network.

**Example:** A Chebyshev convolution of order \( K \) computes a filter spanning each node's \( K \)-hop neighborhood without eigendecomposing the Laplacian.

See also: Spectral Graph Convolution, Graph Convolutional Network.

#### Clique

A subset of nodes in which every pair is directly connected by an edge, forming a fully connected group. Larger cliques indicate denser local cohesion.

Cliques signal tightly knit communities and appear as elementary dense motifs in network analysis.

**Example:** Three mutual friends who all know one another form a triangle, the smallest nontrivial clique.

See also: Local Clustering Coefficient, Community Detection.

#### Closeness Centrality

A node centrality measure defined as the reciprocal of the sum (or average) of shortest-path distances from a node to all others, rewarding nodes that can reach the rest of the network quickly. Higher values indicate more central positioning.

It ranks nodes by how efficiently they can disseminate information.

**Example:** A node at the structural center of a network, close to all others, has high closeness centrality.

See also: Betweenness Centrality, Shortest Path.

#### Cluster-GCN

A scalable GNN training method that first partitions the graph into densely connected clusters and then forms mini-batches from one or a few clusters, keeping most edges within each batch. It limits cross-batch neighbor lookups.

Clustering before batching reduces neighborhood explosion while preserving local structure.

**Example:** Cluster-GCN partitions a large citation graph with a graph-clustering algorithm and trains on subgraphs formed from those clusters.

See also: Graph Partitioning, GraphSAINT, Mini-Batch Training (GNN).

#### Collaborative Filtering

A recommendation technique that predicts a user's preferences from patterns in the preferences of many users, assuming users with similar past behavior will share future tastes. It relies on interaction data rather than item content.

It is the foundational paradigm underlying graph-based recommendation.

**Example:** Collaborative filtering recommends a product to a user because users with similar purchase histories also bought it.

See also: Matrix Factorization (Rec), Neural Collaborative Filter.

#### Community Detection

The task of partitioning a graph's nodes into groups that are densely connected internally and sparsely connected to other groups. It uncovers mesoscale structure between individual nodes and the whole graph.

Detected communities reveal functional modules, social groups, or topical clusters.

**Example:** Community detection on a citation network groups papers into research subfields.

See also: Modularity (Network), Louvain Algorithm.

#### ComplEx

A knowledge-graph embedding model that places entity and relation embeddings in complex-valued space and scores triples with a Hermitian bilinear form, allowing asymmetric relations. The imaginary components break the symmetry of DistMult.

It models symmetric, antisymmetric, and inverse relations within one efficient bilinear framework.

**Example:** Using complex embeddings, ComplEx can score (A, parentOf, B) high while scoring its reverse low, capturing antisymmetry.

See also: DistMult, Antisymmetry (Relation), Bilinear KG Model.

#### Composition (Relation)

A relational pattern in which one relation equals the chaining of two others, so \( r_1(x, y) \) and \( r_2(y, z) \) imply \( r_3(x, z) \), as in "fatherOf" composed with "fatherOf" giving "grandfatherOf". Expressing it requires composable relation operations.

It is central to multi-hop reasoning, where chained relations imply new facts.

**Example:** With translational embeddings, composing the "fatherOf" relation twice yields a vector approximating "grandfatherOf".

See also: Multi-Hop Query, RotatE.

#### Conformalized GNN

A GNN equipped with conformal prediction, producing prediction sets or intervals that contain the true label with a guaranteed probability under exchangeability assumptions. It quantifies uncertainty with statistical coverage guarantees.

It provides calibrated reliability, important when GNN predictions inform high-stakes decisions.

**Example:** A conformalized GNN outputs, for each node, a set of candidate classes guaranteed to include the true class 90% of the time.

See also: Node Classification.

#### Conjunctive Query

A logical knowledge-graph query that combines multiple conditions with logical AND, requiring answers to satisfy all of them simultaneously. It corresponds to intersecting several answer sets.

Handling conjunction is a core capability of structured query-embedding methods.

**Example:** "People who are both physicists and Nobel laureates" is a conjunctive query intersecting two entity sets.

See also: Multi-Hop Query, Box Embedding (Query2Box).

#### Connected Component

A maximal subset of nodes in which every pair is joined by some path, such that no additional node can be added while preserving connectivity. A graph partitions uniquely into its connected components.

Identifying components separates isolated subnetworks before analysis or learning.

**Example:** A friendship graph may split into several connected components, each a distinct social circle with no cross-links.

See also: Giant Component, Strongly Connected Component.

#### Contrastive Loss

A loss function that trains representations by rewarding high similarity between designated positive pairs and low similarity between negative pairs, often via a temperature-scaled softmax over similarities. It shapes the embedding geometry directly.

It is the optimization objective underlying graph contrastive learning.

**Example:** A contrastive loss maximizes the similarity of two augmented views of the same node relative to views of all other nodes in the batch.

See also: Graph Contrastive Learning, Negative Sampling.

#### Cosine Similarity

A measure of the angle between two vectors, computed as their dot product divided by the product of their magnitudes, yielding a value in \( [-1, 1] \). It compares direction independently of magnitude.

It is widely used to compare node or graph embeddings where vector length is not informative.

**Example:** Two papers with cosine similarity near 1 between their node embeddings are likely on the same topic regardless of embedding norm.

Contrast with: Dot Product.

#### Cross-Entropy (Node Class)

The cross-entropy loss applied to multi-class node classification, summing over labeled nodes the negative log-probability assigned to each node's true class. It trains GNNs to output correct categorical labels per node.

It is the standard training objective for transductive and inductive node classifiers.

**Example:** A node classifier minimizes cross-entropy over the labeled nodes of a citation graph to predict each paper's subject.

See also: Cross-Entropy Loss, Node Classification.

#### Cross-Entropy Loss

A loss function measuring the discrepancy between a predicted probability distribution and a one-hot target, \( -\sum_c y_c \log \hat{y}_c \), penalizing confident wrong predictions heavily. It is the standard objective for classification.

It trains GNN node and graph classifiers against ground-truth category labels.

**Example:** A Karate Club node classifier minimizes cross-entropy between predicted and true faction labels.

See also: Cross-Entropy (Node Class), Softmax Function.

#### Curriculum Learning (Graph)

A training strategy that presents graph training examples in a deliberately ordered sequence, typically from easier to harder, to improve convergence and final performance. Difficulty may be gauged by structural or label-noise criteria.

It can stabilize learning on graphs with heterogeneous example difficulty.

**Example:** Curriculum learning trains a graph classifier first on small, simple molecules before introducing large, complex ones.

#### Cycle

A path that begins and ends at the same node without repeating intermediate nodes, forming a closed loop. The presence or absence of cycles distinguishes general graphs from trees.

Cycles affect message-passing dynamics and the structural patterns GNNs can or cannot detect.

**Example:** A triangle of three mutually connected nodes is a cycle of length three.

Contrast with: Tree (Graph Theory).

#### Data Augmentation (Graph)

The creation of modified versions of graphs or their features during training, through perturbations such as edge dropping, feature masking, or subgraph sampling, to improve generalization and robustness. It expands the effective training distribution.

It is especially central to self-supervised graph contrastive learning, which contrasts augmented views.

**Example:** Graph data augmentation randomly removes a fraction of edges to produce a perturbed view of the graph for contrastive training.

See also: DropEdge, Graph Augmentation (SSL), Graph Contrastive Learning.

#### Deep Graph Infomax (DGI)

A self-supervised graph representation method that trains a GNN to maximize the mutual information between local node representations and a global summary of the graph, contrasting real nodes against corrupted ones. It learns without labels by an informativeness objective.

It was an influential demonstration of unsupervised representation learning on graphs.

**Example:** Deep Graph Infomax trains node embeddings to be predictive of a graph-level summary while a corrupted graph's nodes are made unpredictable of it.

See also: Self-Supervised Learning, Graph Contrastive Learning.

#### DeepSNAP

A library that bridges NetworkX graph objects and PyTorch Geometric tensors, easing dataset preparation, transformation, and splitting for GNN experiments. It connects flexible graph manipulation with efficient training.

It streamlines moving between analysis-oriented and learning-oriented graph representations.

**Example:** DeepSNAP converts a NetworkX heterogeneous graph into a PyTorch Geometric dataset ready for GNN training.

See also: NetworkX, PyTorch Geometric (PyG).

#### DeepWalk

A node-embedding method that generates uniform random walks from each node, treats the walks as sentences, and applies the Skip-Gram word-embedding model to learn vectors capturing co-occurrence. It introduced the walk-based embedding paradigm.

It showed that language-modeling techniques transfer naturally to graphs.

**Example:** DeepWalk embeds a social network so that users frequently appearing together in random walks receive similar vectors.

See also: node2vec, Skip-Gram Model, Random Walk.

#### Degree Distribution

The probability distribution \( P(k) \) describing the fraction of nodes in a graph that have degree \( k \). It summarizes a network's connectivity at the population level.

Its shape distinguishes random graphs with light-tailed distributions from real networks with heavy tails.

**Example:** Plotting \( P(k) \) for a social network often reveals many low-degree users and a few high-degree influencers.

See also: Power-Law Degree Distribution, Scale-Free Network.

#### Depth-First Search

A graph traversal algorithm that explores as far as possible along each branch before backtracking, using a stack or recursion. It naturally reaches distant nodes quickly.

It contrasts with breadth-first exploration and informs walk strategies that emphasize structural reach.

**Example:** Depth-first search from a node dives deep along one path before returning to explore alternatives.

See also: Breadth-First Search, DFS Strategy (node2vec).

#### DFS Strategy (node2vec)

A configuration of node2vec's biased walk that encourages the walk to move outward to ever more distant nodes in a depth-first manner. It emphasizes macroscopic community structure.

This setting yields embeddings reflecting homophily, grouping nodes that lie in the same region of the graph.

**Example:** With a large in-out parameter, node2vec walks roam far, embedding nodes of the same community close together.

Contrast with: BFS Strategy (node2vec).

#### DGL

The Deep Graph Library, a framework-agnostic library for building graph neural networks with efficient, scalable message-passing primitives that run on PyTorch, TensorFlow, or MXNet. It emphasizes performance on large graphs.

It is a major alternative to PyTorch Geometric for implementing GNNs.

**Example:** A team uses DGL's sampling and distributed-training utilities to scale a GNN to a billion-edge graph.

See also: PyTorch Geometric (PyG).

#### DiffPool

A differentiable hierarchical graph-pooling method that learns soft assignments of nodes to a smaller set of clusters at each layer, coarsening the graph progressively. It builds a learned multilevel graph hierarchy for graph classification.

It lets GNNs capture hierarchical graph structure rather than treating readout as a single flat step.

**Example:** DiffPool coarsens a molecular graph into successively smaller learned super-node graphs before final classification.

See also: MinCutPool, Graph-Level Readout.

#### DiGress

A discrete denoising-diffusion graph generative model that gradually corrupts graphs by randomly editing node and edge categories, then trains a GNN to reverse the corruption, generating graphs by iterative denoising. It adapts diffusion models to discrete graph structure.

It achieves strong results on molecular generation by respecting the discrete nature of graphs.

**Example:** DiGress generates a molecule by starting from random node and edge labels and iteratively denoising them into a valid molecular graph.

See also: Graph Generative Model, Molecule Generation.

#### DistMult

A bilinear knowledge-graph embedding model that scores a triple by a weighted dot product \( h^\top \text{diag}(r) t \) using a diagonal relation matrix. Its simplicity makes it efficient but inherently symmetric.

It models symmetric relations well but cannot distinguish a relation from its inverse.

**Example:** DistMult scores (A, sibling, B) and (B, sibling, A) identically, which fits the symmetric "sibling" relation but fails for "parentOf".

See also: ComplEx, Bilinear KG Model, Symmetry (Relation Pattern).

#### Dot Product

A scalar formed from two equal-length vectors \( u \) and \( v \) as \( \sum_i u_i v_i \), measuring their alignment. It is zero when the vectors are orthogonal.

Dot products of node embeddings serve as a similarity or scoring function for predicting links.

**Example:** A link predictor scores a candidate edge \( (u,v) \) by the dot product \( z_u^\top z_v \) of the two node embeddings.

See also: Cosine Similarity, Link Prediction.

#### DropEdge

A graph regularization technique that randomly removes a fraction of edges from the graph at each training iteration, reducing message passing and mitigating over-smoothing in deep GNNs. It acts as structural dropout on the adjacency.

It both regularizes training and slows the convergence of representations toward uniformity.

**Example:** DropEdge randomly deletes 20% of edges each epoch, so a deep GNN aggregates over varying sparser neighborhoods.

See also: Dropout (GNN), Over-Smoothing, Data Augmentation (Graph).

#### Dropout (GNN)

A regularization technique that randomly zeroes a fraction of activations or input features during training, preventing co-adaptation of units and reducing overfitting, applied within GNN layers. It is rescaled at inference to preserve expected magnitudes.

It is a standard regularizer in GNN training, complemented by graph-specific variants like DropEdge.

**Example:** Applying dropout to node feature vectors before a GNN layer randomly masks features to regularize training.

See also: DropEdge, Batch Normalization (GNN).

#### Drug Discovery with GNNs

The application of graph neural networks to predict molecular properties, screen candidate compounds, and guide the design of new molecules from their molecular graphs. It accelerates the search for viable drug candidates.

It combines graph-level prediction and graph generation to support the discovery pipeline.

**Example:** A GNN screens millions of molecular graphs to prioritize compounds likely to bind a target protein.

See also: Molecular Graph, Molecule Generation, Graph Regression.

#### Drug-Drug Interaction

A graph machine-learning application that predicts whether two drugs interact, often modeled as link prediction in a graph whose nodes are drugs and edges are known interactions. It supports pharmacovigilance and safe prescribing.

It illustrates link prediction applied to biomedical safety.

**Example:** A GNN predicts a potential adverse drug-drug interaction by scoring a candidate edge between two drug nodes.

See also: Link Prediction, Protein-Protein Interaction.

#### Dynamic Graph

A graph that evolves through additions and deletions of nodes and edges over time, encompassing both discrete-time snapshot sequences and continuous-time event streams. It generalizes static graph learning to change.

Dynamic-graph methods must update representations as the graph changes rather than recomputing from scratch.

**Example:** A financial transaction network is a dynamic graph as new accounts and transfers continually appear.

See also: Temporal Graph, Temporal GNN (TGN).

#### Early Stopping (GNN)

A regularization strategy that halts training when performance on a held-out validation set stops improving, preventing overfitting and saving computation. It selects the model state with best validation performance.

It is a standard, simple safeguard against overfitting in GNN training.

**Example:** Early stopping ends GNN training after validation accuracy fails to improve for a set number of consecutive epochs.

See also: Train/Val/Test Split (Trans).

#### Edge (Link)

A connection between two nodes in a graph representing a relationship or interaction, optionally carrying a direction, weight, type, or feature vector. Edges define the structure over which information flows.

The set of edges determines a node's neighborhood and thus the messages it receives in a GNN.

**Example:** In a knowledge graph, an edge labeled "born_in" links a person node to a city node.

See also: Node (Vertex), Weighted Graph.

#### Edge Encoding (Transformer)

A mechanism in graph transformers that incorporates edge features, often by aggregating features of edges along the shortest path between two nodes, into the attention computation between those nodes. It lets edge information shape global attention.

It allows transformers to use bond types or relation labels that local message passing would otherwise carry.

**Example:** Edge encoding in Graphormer averages bond-type features along the path between two atoms and adds them to their attention score.

See also: Graphormer, Spatial Bias (Graph Attention).

#### Edge-Level Task

A graph learning task whose prediction target concerns pairs of nodes, such as whether an edge exists or what type it is. It operates on node pairs rather than single nodes or whole graphs.

It is one of the three canonical task granularities and central to recommendation and knowledge-graph completion.

**Example:** Predicting whether two users will become friends is an edge-level task.

See also: Link Prediction, Node-Level Task, Graph-Level Task.

#### Ego Network

The subgraph consisting of a focal node (the ego), all of its direct neighbors, and the edges among them. It captures a node's immediate local social structure.

Ego networks are a natural unit for studying local connectivity and for sampling-based GNN training.

**Example:** A user's ego network on a social platform contains that user, their friends, and the friendships between those friends.

See also: Subgraph, K-Hop Neighborhood.

#### Eigenvalue Decomposition

A factorization of a square matrix \( A \) into \( A = Q \Lambda Q^{-1} \), where columns of \( Q \) are eigenvectors and \( \Lambda \) is a diagonal matrix of eigenvalues. For symmetric matrices, \( Q \) can be chosen orthogonal.

This decomposition underlies spectral graph theory, since the eigenstructure of the graph Laplacian defines graph frequencies and spectral filters.

**Example:** Decomposing the normalized graph Laplacian yields eigenvalues used to define low-pass filters in spectral graph convolutions.

#### Eigenvector

A nonzero vector \( v \) that a square matrix \( A \) scales without changing direction, satisfying \( A v = \lambda v \) for scalar eigenvalue \( \lambda \). Eigenvectors of graph matrices encode structural patterns.

**Example:** The leading eigenvector of an adjacency matrix gives eigenvector centrality, ranking nodes by the importance of their neighbors.

See also: Eigenvalue Decomposition, Eigenvector Centrality.

#### Eigenvector Centrality

A node centrality measure assigning each node a score proportional to the sum of its neighbors' scores, given by the leading eigenvector of the adjacency matrix. A node is important if its neighbors are important.

It captures influence that accrues from being connected to other influential nodes, foreshadowing PageRank.

**Example:** In a social network, a user connected to a few highly influential users can have high eigenvector centrality despite a modest degree.

See also: PageRank, Eigenvector.

#### Embedding Space

The continuous vector space into which graph nodes, edges, or whole graphs are mapped, where geometric relationships such as distance and angle encode learned structural relationships. Its dimensionality trades capacity against efficiency.

The geometry of this space determines how relationships are read off via dot products or distances.

**Example:** In a well-trained embedding space, clusters of points correspond to communities in the original graph.

See also: Node Embedding, KG Embedding Geometry.

#### Encoder-Decoder Framework

A unifying view of node-embedding methods in which an encoder maps nodes to vectors and a decoder reconstructs a target graph quantity, such as edge existence or proximity, from those vectors. Training aligns decoded outputs with the true graph.

It clarifies that many embedding methods differ mainly in their encoder, decoder, and similarity choices.

**Example:** In this framework, DeepWalk uses a lookup-table encoder and a decoder that predicts random-walk co-occurrence.

See also: Shallow Embedding, Node Embedding.

#### Erdős–Rényi Random Graph

A random graph model in which each possible edge among \( N \) nodes is included independently with fixed probability \( p \). It serves as a baseline null model of unstructured connectivity.

Its predictable properties contrast with real networks, highlighting features like heavy tails and clustering that it fails to reproduce.

**Example:** An Erdős–Rényi graph with \( N = 100 \) and \( p = 0.05 \) has roughly 250 randomly placed edges and a near-Poisson degree distribution.

See also: Giant Component, Barabási–Albert Model.

#### Fine-Tuning (GNN)

The adaptation of a pretrained GNN to a specific downstream task by continuing training, usually with a smaller learning rate and task-specific labels, starting from the pretrained parameters. It specializes general representations.

It is the second stage of the pretrain-then-adapt transfer-learning workflow.

**Example:** After pretraining, a GNN is fine-tuned on a few thousand labeled molecules to predict a particular biological activity.

See also: Pre-Training (GNN), Transfer Learning (Graphs).

#### Fraud Detection (Graph)

A graph machine-learning application that identifies fraudulent entities or transactions by exploiting relational patterns, often as node classification on graphs where fraudsters cluster or hide among normal nodes. Relations expose coordinated abuse.

Graph structure reveals collusion that per-record models miss, though fraud graphs are often heterophilous.

**Example:** A GNN flags fraudulent accounts by detecting suspicious connectivity patterns linking them to known bad actors.

See also: Node Classification, Homophily.

#### Frequent Subgraph Mining

The task of discovering subgraph patterns that occur at least a specified number of times across a graph or a collection of graphs. It uncovers recurring structural building blocks.

Frequent patterns reveal characteristic substructures, though counting them is computationally demanding.

**Example:** Frequent subgraph mining on a chemical dataset finds molecular substructures common to many active compounds.

See also: Subgraph Isomorphism, SPMiner, Network Motif.

#### GCPN

A goal-directed graph generative model that builds molecular graphs step by step using reinforcement learning, with rewards for desired chemical properties and adversarial signals for realism. It optimizes generation toward target objectives.

It combines generation with property optimization for goal-directed molecular design.

**Example:** GCPN grows a molecular graph atom by atom, receiving reward for high drug-likeness while a discriminator enforces chemical validity.

See also: GraphRNN, Molecule Generation, Drug Discovery with GNNs.

#### Giant Component

A connected component that contains a constant, nonvanishing fraction of all nodes in a large graph, dominating the network's connectivity. Its emergence marks a percolation threshold.

The sudden appearance of a giant component as edge density rises explains abrupt large-scale connectivity in networks.

**Example:** In an Erdős–Rényi graph, once average degree exceeds one, a giant component containing most nodes abruptly forms.

See also: Connected Component, Erdős–Rényi Random Graph.

#### Girvan-Newman Algorithm

A divisive community-detection method that repeatedly removes the edge with highest betweenness centrality, progressively splitting the graph into communities. It produces a hierarchy of partitions.

It illustrates how bridging edges, once removed, reveal underlying community structure, though it scales poorly.

**Example:** The Girvan-Newman algorithm separates the two factions of the Karate Club by removing the edges bridging them.

See also: Betweenness Centrality, Community Detection.

#### Global Mean Pooling

A graph readout operation that averages the representations of all nodes in a graph into one vector. It produces a size-normalized summary insensitive to the number of nodes.

It is a simple, robust default for converting node embeddings into a graph embedding.

**Example:** Global mean pooling over a molecule's atom embeddings yields a single vector summarizing the molecule.

Contrast with: Global Sum Pooling.

#### Global Sum Pooling

A graph readout operation that adds the representations of all nodes in a graph into one vector, preserving information about graph size. It is more expressive than averaging for distinguishing graphs of different sizes.

It retains the cumulative magnitude of node features, aiding tasks where graph size matters.

**Example:** Global sum pooling distinguishes a small molecule from a larger one with the same average atom features.

Contrast with: Global Mean Pooling.

#### GNN Bottleneck

A structural choke point in a graph where many long-range messages must funnel through few edges or nodes, forcing them to be compressed and causing over-squashing. It identifies where information flow is constrained.

Recognizing bottlenecks guides graph-rewiring strategies that add shortcut connections.

**Example:** A single edge bridging two large communities forms a bottleneck through which all cross-community information must squeeze.

See also: Over-Squashing, Virtual Edge Augmentation.

#### GNN Distinguishing Power

The set of structural differences a GNN can detect, measured by which pairs of non-isomorphic graphs or nodes it maps to different representations. It is the operational form of expressiveness.

Architectures are often compared by where their distinguishing power sits in the WL hierarchy.

**Example:** Adding sum aggregation and injective updates raises a GNN's distinguishing power to match the 1-WL test.

See also: GNN Expressiveness, k-WL Test.

#### GNN Expressiveness

The capacity of a GNN to produce distinct representations for non-isomorphic graphs or structurally different nodes, formalized by comparison to the Weisfeiler-Lehman hierarchy. It bounds which structures a model can possibly distinguish.

Understanding expressiveness guides the design of architectures that overcome the limits of basic message passing.

**Example:** A GNN whose expressiveness equals the 1-WL test cannot tell apart two non-isomorphic graphs that 1-WL conflates.

See also: GNN Distinguishing Power, 1-WL Test, Graph Isomorphism Network.

#### GPS (General Powerful GNN)

A modular graph transformer framework that combines a local message-passing GNN module with a global self-attention module in each layer, plus configurable positional and structural encodings. It blends local structure with global reach.

It offers a flexible recipe that often outperforms either pure message passing or pure attention alone.

**Example:** A GPS layer aggregates local neighbors with a GNN and simultaneously lets every node attend globally via a transformer.

See also: Graph Transformer, Random Walk Struct Encoding.

#### Gradient Descent

An iterative optimization method that updates parameters in the direction opposite the gradient of a loss function, \( \theta \leftarrow \theta - \eta \nabla_\theta L \), to minimize that loss. The step size \( \eta \) is the learning rate.

It is the standard procedure for training graph neural networks by minimizing prediction error over labeled nodes, edges, or graphs.

**Example:** Training a GCN node classifier repeatedly nudges weight matrices down the gradient of the cross-entropy loss.

See also: Adam Optimizer, Backpropagation.

#### Graph (Directed)

A graph in which each edge is an ordered pair \( (u, v) \), encoding an asymmetric relationship from a source node to a target node. Its adjacency matrix need not be symmetric.

Directed graphs model relationships like citations or web links where direction carries meaning.

**Example:** A citation network is directed because paper A citing paper B does not imply B cites A.

See also: In-Degree, Out-Degree.

#### Graph (Undirected)

A mathematical structure \( G = (V, E) \) consisting of a set of nodes and a set of unordered node pairs called edges, where an edge \( \{u, v\} \) represents a symmetric relationship. Its adjacency matrix is symmetric.

Many social and molecular networks are naturally undirected, making this the default graph type in introductory GNN material.

**Example:** A friendship network is undirected because if A is friends with B, then B is friends with A.

Contrast with: Graph (Directed).

#### Graph Attention Network (GAT)

A GNN that aggregates neighbor representations using learned attention coefficients, weighting each neighbor by a normalized relevance score computed from the pair of node features. Important neighbors contribute more.

Attention lets the model adaptively emphasize the most informative neighbors rather than weighting them uniformly.

**Example:** A graph attention network may learn to weight a paper's most topically relevant citations more heavily during aggregation.

See also: Attention Mechanism (Graph), Multi-Head Attention (Graph).

#### Graph Augmentation (SSL)

The set of stochastic graph transformations, such as node dropping, edge perturbation, attribute masking, or subgraph sampling, used to generate the differing views required by self-supervised contrastive methods. The augmentations define what invariances are learned.

The choice of augmentations strongly influences the quality and usefulness of the learned representations.

**Example:** Graph augmentation for self-supervised learning produces two views of a graph, one with masked node features and one with dropped edges.

See also: Data Augmentation (Graph), Graph Contrastive Learning.

#### Graph Classification

The task of assigning a categorical label to an entire graph, requiring a graph-level representation produced by readout over node embeddings. It treats each graph as a single labeled example.

It is standard in chemistry and bioinformatics, where each graph is a molecule or structure.

**Example:** Classifying molecular graphs as active or inactive against a target is graph classification.

See also: Graph-Level Task, Graph-Level Readout, Graph Kernel.

#### Graph Contrastive Learning

A self-supervised approach that creates multiple augmented views of a graph and trains representations so that views of the same node or graph are pulled together while views of different ones are pushed apart. It learns invariances to the chosen augmentations.

It is a leading method for label-free graph representation pretraining.

**Example:** Graph contrastive learning makes a node's embedding consistent across two edge-dropped views of its graph while differing from other nodes' embeddings.

See also: Contrastive Loss, Graph Augmentation (SSL), Data Augmentation (Graph).

#### Graph Convolutional Network

A GNN architecture whose layer updates each node by a symmetrically normalized sum of its neighbors' (and its own) transformed features, derived as a first-order approximation of spectral graph convolution. It is a foundational message-passing model.

Its simplicity and effectiveness make it the canonical baseline GNN for node classification.

**Example:** A two-layer graph convolutional network reaches strong accuracy classifying papers on the Cora citation benchmark.

See also: Spectral Graph Convolution, Message Passing Neural Net, Chebyshev Polynomial Conv.

#### Graph Diameter

The greatest shortest-path distance between any pair of nodes in a connected graph, quantifying the network's worst-case span. It bounds how many hops information needs to reach the farthest node.

Small diameters in real networks motivate the surprising effectiveness of shallow GNNs.

**Example:** Many social networks have a diameter of about six, reflecting the "six degrees of separation" phenomenon.

See also: Small-World Network, Shortest Path.

#### Graph Equivariance

The property of a function on graphs whereby permuting the input nodes permutes the output correspondingly, so node-level outputs transform consistently with node relabeling. It is a symmetry requirement for well-posed node-level models.

Respecting equivariance ensures predictions do not arbitrarily depend on node ordering.

**Example:** A node-classification GNN is permutation-equivariant: relabeling the nodes simply relabels the predictions identically.

See also: Permutation Equivariance, Graph Invariance.

#### Graph Foundation Model

A large model pretrained on broad graph data to provide transferable representations or task-solving abilities across many graphs and tasks, adaptable with little or no task-specific training. It aims to bring the foundation-model paradigm to graphs.

It promises reusable graph intelligence rather than bespoke models trained per dataset.

**Example:** A graph foundation model pretrained across diverse networks performs node classification on a new graph with minimal adaptation.

See also: In-Context Learning (Graphs), One-For-All (OFA) Model, ULTRA.

#### Graph Generation Metrics

Quantitative measures that assess how well generated graphs match a target distribution, comparing statistics such as degree distributions, clustering coefficients, and motif counts, alongside validity, uniqueness, and novelty. They evaluate generative quality.

They are needed because generated graphs cannot be compared to a single ground-truth target.

**Example:** Graph generation metrics report the fraction of valid, unique, and novel molecules a model produces and how closely their degree distributions match the training set.

See also: Graph Generative Model, Molecule Generation.

#### Graph Generative Model

A model that learns the distribution of graphs from a dataset and can sample new graphs resembling the training data. It addresses the challenge of generating discrete, permutation-invariant relational structures.

Such models support tasks like designing novel molecules with desired properties.

**Example:** A graph generative model trained on drug-like molecules samples new molecular graphs with similar chemical characteristics.

See also: GraphRNN, Variational Autoencoder (VGAE), DiGress.

#### Graph Homomorphism

A mapping between two graphs that preserves edges, sending adjacent nodes to adjacent nodes, but without requiring injectivity or surjectivity. It is a structure-preserving map weaker than isomorphism.

Homomorphism counts characterize structural patterns and relate to the expressive power of certain GNNs.

**Example:** Counting homomorphisms from a triangle into a target graph counts its closed neighbor relationships.

Contrast with: Graph Isomorphism.

#### Graph Instruction Tuning

The fine-tuning of a large language model on instruction-formatted graph tasks so that it can follow natural-language prompts to reason over graph-structured inputs. It adapts instruction tuning to the graph domain.

It aims to make language models flexibly solve graph tasks specified in plain language.

**Example:** Graph instruction tuning trains a language model on prompts like "Given this node's neighbors, predict its category," with graph context serialized as text.

See also: LLM + GNN Integration, One-For-All (OFA) Model.

#### Graph Invariance

The property of a function on graphs whose output is unchanged when the input nodes are permuted, appropriate for whole-graph outputs that should not depend on node ordering. It is a symmetry requirement for graph-level models.

Invariant readouts guarantee that two isomorphic graphs receive the same prediction.

**Example:** A graph classifier with a sum-pooling readout is permutation-invariant, giving the same label regardless of node order.

See also: Permutation Invariance, Graph Equivariance.

#### Graph Isomorphism

A relationship between two graphs in which a bijection between their node sets preserves adjacency, so the graphs are structurally identical up to relabeling. It formalizes "same shape."

Distinguishing non-isomorphic graphs is the benchmark against which GNN expressiveness is measured.

**Example:** Two molecular graphs are isomorphic if their atoms can be relabeled to make their bond structures coincide exactly.

See also: Graph Isomorphism Problem, Weisfeiler-Lehman (WL) Test.

#### Graph Isomorphism Network

A GNN designed to be maximally expressive among message-passing models by using sum aggregation followed by a multilayer perceptron with an injective update, matching the discriminative power of the 1-WL test. It distinguishes graph structures that weaker aggregators conflate.

Its design is motivated directly by analysis of GNN expressiveness.

**Example:** A graph isomorphism network can distinguish two molecules that mean-aggregation GNNs map to identical representations.

See also: Sum Aggregation, 1-WL Test, GNN Expressiveness.

#### Graph Isomorphism Problem

The computational problem of deciding whether two given graphs are isomorphic, whose general complexity is not known to be polynomial or NP-complete. It frames the difficulty of distinguishing graph structures.

GNN expressiveness is studied through its relationship to this problem and its WL-test approximations.

**Example:** Because no efficient general solution to the graph isomorphism problem is known, GNNs rely on the tractable WL approximation.

See also: Graph Isomorphism, Weisfeiler-Lehman (WL) Test.

#### Graph Kernel

A function that measures similarity between two whole graphs, typically by comparing counts of shared substructures such as walks, paths, or subtrees, while remaining a valid positive semi-definite kernel. It enables kernel methods on graphs.

Graph kernels were the dominant pre-GNN approach to graph classification.

**Example:** A graph kernel can compare two molecules by counting their shared labeled subtree patterns and feeding the similarity to a support vector machine.

See also: Weisfeiler-Lehman Kernel, Graph Classification.

#### Graph Laplacian

The matrix \( L = D - A \), where \( D \) is the diagonal degree matrix and \( A \) the adjacency matrix, encoding a graph's connectivity in a form whose eigenstructure defines graph frequencies. It is symmetric and positive semi-definite for undirected graphs.

It is the central operator of spectral graph theory and the basis of spectral GNNs and clustering.

**Example:** The quadratic form \( x^\top L x \) measures how much a node signal \( x \) varies across edges, quantifying its smoothness.

See also: Normalized Graph Laplacian, Spectral Graph Convolution.

#### Graph Neural Network (GNN)

A neural network that operates directly on graph-structured data by iteratively updating each node's representation from its own features and aggregated information from its neighbors. It learns to combine structure and features.

GNNs unify and generalize earlier graph-learning methods into an end-to-end trainable framework.

**Example:** A GNN classifies papers in a citation network by repeatedly blending each paper's features with those of the papers it cites.

See also: Message Passing Neural Net, Graph Convolutional Network.

#### Graph Partitioning

The task of dividing a graph's nodes into balanced groups while minimizing the number of edges crossing between groups. It supports distributed storage and cluster-based GNN training.

Good partitions keep most message-passing computation local within each partition.

**Example:** Graph partitioning splits a billion-edge graph into balanced blocks so each machine processes one block with few cross-machine edges.

See also: Cluster-GCN, Normalized Cut.

#### Graph Path

A sequence of nodes connected by consecutive edges, typically without repeated nodes, linking a start node to an end node. Its length is the number of edges traversed.

Paths formalize reachability and underpin distance-based graph measures.

**Example:** A path of length two connecting two researchers through a shared co-author indicates an indirect collaboration.

See also: Shortest Path, Graph Diameter.

#### Graph Regression

The task of predicting a continuous numerical value for an entire graph from its structure and features. Like graph classification it requires a graph-level representation, but with a real-valued target.

It applies wherever a scalar property of a whole graph must be estimated.

**Example:** Predicting a molecule's solubility from its molecular graph is graph regression.

See also: Graph-Level Task, Graph Classification.

#### Graph Sampling Strategy

The overall policy governing which nodes, edges, or subgraphs are selected to form each training batch in scalable GNN training, trading off computational cost, memory, and the variance of gradient estimates. It determines how the full graph is approximated per step.

The chosen strategy critically affects both efficiency and final model quality at scale.

**Example:** Choosing a layer-wise graph sampling strategy over naive neighbor expansion reduces redundant computation on a large graph.

See also: Neighbor Sampling, Layer-Wise Sampling, GraphSAINT.

#### Graph Transformer

An architecture that applies the transformer's self-attention over graph nodes, allowing every node to attend to every other node, with graph structure injected through positional or structural encodings and attention biases. It enables direct global interaction.

It sidesteps over-squashing and limited receptive fields by replacing local message passing with global attention.

**Example:** A graph transformer lets two distant atoms in a molecule attend to each other directly rather than through many hops.

See also: Graphormer, GPS (General Powerful GNN), Laplacian Positional Encoding.

#### Graph-Level Readout

An operation that aggregates all node representations in a graph into a single fixed-size vector representing the entire graph, enabling graph-level prediction. It must be invariant to node ordering.

Readout is the bridge from node-level GNN computation to whole-graph tasks.

**Example:** Summing every node embedding produces a graph-level vector that a classifier maps to a molecule's toxicity label.

See also: Global Sum Pooling, Global Mean Pooling, Graph Classification.

#### Graph-Level Task

A graph learning task whose prediction target is a property of an entire graph, requiring a single output per graph. It typically follows a readout that summarizes all node representations.

It is one of the three canonical task granularities, common in molecular and structural prediction.

**Example:** Predicting whether a molecule is toxic from its molecular graph is a graph-level task.

See also: Graph Classification, Graph-Level Readout, Node-Level Task.

#### Graphlet

A small, connected, induced subgraph pattern, considered up to isomorphism, used to characterize a network's local topology. Unlike motifs, graphlets are counted regardless of statistical over-representation.

Graphlet counts provide fine-grained structural fingerprints for nodes and graphs.

**Example:** Counting how often each four-node graphlet appears around a node summarizes its local connectivity pattern.

See also: Graphlet Degree Vector, Network Motif.

#### Graphlet Degree Vector

A feature vector for a node that counts how many times the node participates in each automorphism orbit of small graphlets. It encodes the node's local structural role in detail.

It supplies hand-crafted topological features for nodes when learned embeddings are unavailable or undesired.

**Example:** Two nodes with similar graphlet degree vectors play structurally similar roles even if they are far apart in the graph.

See also: Graphlet, Structural Equivalence.

#### Graphormer

A graph transformer that encodes graph structure into self-attention through centrality encoding of node degrees, spatial encoding of shortest-path distances as attention bias, and edge encoding along paths. It achieved strong results on molecular property benchmarks.

It demonstrated that carefully designed structural encodings let pure transformers excel on graphs.

**Example:** Graphormer biases attention between two atoms by their shortest-path distance, encoding molecular geometry.

See also: Graph Transformer, Spatial Bias (Graph Attention), Centrality Encoding.

#### GraphRNN

A graph generative model that produces a graph autoregressively, adding one node at a time and using recurrent networks to decide its edges to previously added nodes under a fixed node ordering. It frames graph generation as sequence generation.

It demonstrated that autoregressive sequence models can capture complex structural distributions of graphs.

**Example:** GraphRNN generates a graph by sequentially emitting nodes and, for each, predicting its connections to all earlier nodes.

See also: Graph Generative Model, GCPN.

#### GraphSAGE

An inductive GNN that updates each node by sampling a fixed-size subset of its neighbors, aggregating their representations with a learnable function, and combining the result with the node's own representation. It generalizes to unseen nodes.

Neighbor sampling lets it scale to large graphs and embed nodes absent during training.

**Example:** GraphSAGE trained on part of a large citation network can embed newly added papers without retraining.

See also: Inductive Learning, Neighbor Sampling, Mean Aggregation.

#### GraphSAINT

A scalable GNN training method that samples connected subgraphs and runs a full GNN on each, applying normalization to correct for sampling bias in the estimates. It samples subgraphs rather than per-node neighborhoods.

Subgraph sampling reduces the neighbor-explosion problem while keeping unbiased gradient estimates.

**Example:** GraphSAINT trains by repeatedly sampling a subgraph via a random-walk sampler and computing a normalized loss on it.

See also: Cluster-GCN, Mini-Batch Training (GNN), Graph Partitioning.

#### HAN (Hetero Attention Net)

A heterogeneous GNN that applies attention at two levels: node-level attention within each meta-path to weight neighbors, and semantic-level attention to weight the different meta-paths. It fuses multiple meta-path views of a node.

It learns which meta-path-based relationships matter most for a prediction.

**Example:** HAN classifies authors by attending over both "author-paper-author" and "author-venue-author" meta-paths and weighting their contributions.

See also: Meta-Path, Heterogeneous GNN, Attention Mechanism (Graph).

#### Heterogeneous GNN

A GNN designed for graphs with multiple node and edge types, maintaining type-specific transformations so that messages respect the semantics of each relation and node category. It generalizes message passing to typed graphs.

It is required wherever a single uniform aggregation would conflate semantically distinct relations.

**Example:** A heterogeneous GNN on an academic graph uses separate weight matrices for "author-writes-paper" and "paper-cites-paper" edges.

See also: R-GCN (Relational GCN), HAN (Hetero Attention Net), Heterogeneous Graph.

#### Heterogeneous Graph

A graph containing multiple node types and/or multiple edge types, where the semantics of each type differ. It models richly typed relational data beyond a single uniform relation.

Heterogeneous structure demands specialized GNNs that maintain type-specific parameters.

**Example:** An academic graph with author, paper, and venue nodes connected by "writes" and "published_in" edges is heterogeneous.

See also: Heterogeneous GNN, Meta-Path, R-GCN (Relational GCN).

#### Heterogeneous Graph Trans

A graph transformer for heterogeneous graphs that uses type-dependent attention and message parameters, mutually computing attention between nodes according to their and the connecting edge's types. It applies attention across typed relations.

It adapts transformer-style attention to graphs with diverse node and edge semantics.

**Example:** A heterogeneous graph transformer computes distinct attention parameters for "author→paper" versus "paper→venue" connections.

See also: Heterogeneous GNN, HAN (Hetero Attention Net), Meta-Path.

#### Higher-Order GNN

A GNN that operates on tuples of nodes or higher-order structures rather than single nodes, raising expressiveness toward the \( k \)-WL hierarchy at increased computational cost. It captures relational patterns invisible to ordinary message passing.

It trades scalability for the ability to distinguish more graph structures.

**Example:** A higher-order GNN modeled on the 3-WL test can detect substructures that standard node-wise GNNs miss.

See also: k-WL Test, Subgraph GNN.

#### Historical Embeddings (SIGN)

A scalability technique that caches previously computed node representations and reuses them as stand-ins for neighbors not in the current batch, avoiding recursive multi-hop neighbor expansion. It trades memory and slight staleness for speed.

It enables training of deeper GNNs on large graphs by bounding per-batch computation.

**Example:** Using historical embeddings, a GNN approximates an out-of-batch neighbor's representation with its cached value from a prior step.

See also: SIGN Architecture, Mini-Batch Training (GNN).

#### HITS Algorithm

A link-analysis algorithm that assigns each node two scores, a hub score and an authority score, where good hubs point to good authorities and good authorities are pointed to by good hubs. The scores are computed by iterated mutual reinforcement.

It offers a two-sided alternative to PageRank that separates curating and authoritative roles.

**Example:** In a web graph, a directory page that links to many quality sites earns a high hub score, while those sites earn high authority scores.

See also: PageRank, Eigenvector Centrality.

#### Hits@K Metric

A ranking evaluation metric equal to the fraction of queries for which the correct answer appears among the top \( K \) ranked candidates. It measures how often the right answer is retrieved within a cutoff.

It is widely reported alongside mean reciprocal rank for knowledge-graph link prediction.

**Example:** A Hits@10 of 0.6 means the correct entity is in the top ten predictions for 60% of queries.

See also: Mean Reciprocal Rank (MRR), Link Prediction (KG).

#### Homophily

The tendency of connected nodes in a network to share attributes or labels, so neighbors are more similar than random pairs. It is the assumption that connection implies similarity.

Most standard GNNs and label propagation rely on homophily, and their performance degrades on heterophilous graphs.

**Example:** In a citation network, papers tend to cite others on the same topic, exhibiting label homophily.

See also: Label Propagation, Structural Equivalence.

#### Identity-Aware GNN

A GNN that injects an explicit marker for the node being computed, performing aggregation on a rooted, identity-augmented neighborhood so the model can recognize the focal node and surpass the 1-WL expressiveness limit. It distinguishes structures standard GNNs cannot.

It increases distinguishing power without resorting to full higher-order computation.

**Example:** An identity-aware GNN can count the cycles a node lies on, which a standard message-passing GNN cannot.

See also: Position-Aware GNN, Subgraph GNN, GNN Expressiveness.

#### In-Context Learning (Graphs)

A capability whereby a model performs a graph task by conditioning on a few demonstration examples provided at inference time, without updating its parameters. It transfers the prompting paradigm to graph problems.

It points toward graph foundation models that adapt to new tasks without retraining.

**Example:** A graph foundation model classifies nodes in a new graph from a handful of labeled example nodes given in the prompt, without fine-tuning.

See also: Graph Foundation Model, One-For-All (OFA) Model.

#### In-Degree

The number of edges directed into a node in a directed graph, counting incoming relationships. It measures how often a node is targeted.

In-degree distinguishes frequently referenced entities, such as highly cited papers or popular web pages.

**Example:** A landmark paper has high in-degree in a citation graph because many later papers cite it.

Contrast with: Out-Degree.

#### Independent Cascade Model

A diffusion model in which each newly activated node gets a single chance to activate each inactive neighbor independently with a given probability. Spread proceeds through independent contagion events.

It models influence as probabilistic, one-shot transmission along each edge.

**Example:** Under the independent cascade model, a user who shares a post has a fixed probability of causing each follower to reshare it once.

Contrast with: Linear Threshold Model.

#### Inductive KG Reasoning

Knowledge-graph reasoning that generalizes to entities, and sometimes relations, not seen during training by relying on relational structure rather than entity-specific embeddings. It enables prediction on dynamically growing knowledge graphs.

It contrasts with transductive embedding methods that require every entity to appear at training time.

**Example:** An inductive reasoner can answer queries about a newly added entity using only its relational connections, without retraining.

See also: ULTRA, InGram, Neural Bellman-Ford Net.

#### Inductive Learning

A learning setting in which the trained model can produce predictions for nodes or graphs unseen during training by relying on features and shared parameters rather than per-node lookup. It generalizes beyond the training graph.

It is essential for deploying GNNs on evolving graphs or entirely new graphs.

**Example:** A GraphSAGE model trained on one citation network can embed papers in a different, previously unseen network.

Contrast with: Transductive Learning.

#### Influence Maximization

The combinatorial problem of selecting a small set of seed nodes in a network to maximize the expected spread of influence under a chosen diffusion model. It is NP-hard but admits greedy approximation guarantees.

It formalizes viral marketing and the strategic seeding of information cascades.

**Example:** Influence maximization picks a handful of users to give free samples so that resulting word-of-mouth reaches the largest audience.

See also: Independent Cascade Model, Linear Threshold Model.

#### InGram

An inductive knowledge-graph embedding method that generates representations for unseen relations and entities at inference time by building a relation graph and learning from relational structure. It targets generalization to wholly new relations.

It addresses the harder inductive setting where even relation types are novel.

**Example:** InGram can score facts involving a relation type that never appeared during training by leveraging its position in a learned relation graph.

See also: Inductive KG Reasoning, ULTRA.

#### Inversion (Relation)

A relational pattern in which one relation is the reverse of another, so \( r_1(x, y) \) implies \( r_2(y, x) \), as with "parentOf" and "childOf". Capturing it requires that inverse relations be representable.

Modeling inversion lets a knowledge-graph embedding infer reciprocal facts consistently.

**Example:** RotatE represents inversion by assigning "childOf" the rotation opposite to "parentOf".

See also: Composition (Relation), RotatE.

#### Jumping Knowledge Network

A GNN architecture that combines the representations a node acquires at every layer, via concatenation, max-pooling, or attention, so each node can select the neighborhood range best suited to it. It decouples effective depth per node.

It addresses the fact that different nodes benefit from different receptive-field sizes.

**Example:** A jumping knowledge network lets a hub node draw on shallow-layer features while a peripheral node uses deeper-layer features.

See also: Skip Connection (GNN), Over-Smoothing.

#### K-Hop Neighborhood

The set of nodes reachable from a given node within \( k \) edges, encompassing all nodes at graph distance at most \( k \). It defines how far structural information extends around a node.

After \( k \) GNN layers, a node's representation depends on exactly its \( k \)-hop neighborhood.

**Example:** A node's two-hop neighborhood includes its neighbors and its neighbors' neighbors.

See also: Receptive Field (GNN), Layer Depth (GNN).

#### k-WL Test

A higher-dimensional generalization of the Weisfeiler-Lehman test that refines labels over \( k \)-tuples of nodes rather than single nodes, yielding strictly greater distinguishing power as \( k \) grows. It forms a hierarchy of increasingly expressive tests.

It provides the theoretical benchmark that higher-order GNNs aim to match.

**Example:** The 3-WL test distinguishes some graph pairs that the 1-WL test, and ordinary GNNs, cannot.

See also: 1-WL Test, Higher-Order GNN.

#### Katz Similarity

A node-similarity measure that sums the number of paths of all lengths between two nodes, weighting longer paths by a decaying factor \( \beta^\ell \). It rewards both short and abundant connections.

It generalizes adjacency-based proximity and serves as a target for proximity-preserving embeddings.

**Example:** Two researchers with many short connecting paths in a collaboration graph have high Katz similarity.

See also: Matrix Factorization (Graph), Node Embedding.

#### KG Completion

The task of inferring missing triples in an incomplete knowledge graph by predicting plausible head or tail entities for given relations. It addresses the inherent incompleteness of real knowledge bases.

It is the central evaluation task for knowledge-graph embedding methods.

**Example:** Given (Einstein, bornIn, ?), knowledge-graph completion ranks candidate cities and predicts "Ulm".

See also: Link Prediction (KG), Mean Reciprocal Rank (MRR).

#### KG Embedding Geometry

The geometric structure of the space in which knowledge-graph entities and relations are embedded, including whether relations act as translations, rotations, or bilinear forms, which determines the relational patterns a model can represent. It links model design to logical expressiveness.

Choosing an appropriate geometry decides whether symmetry, composition, and inversion can be captured.

**Example:** Translational geometry, as in TransE, naturally encodes composition but cannot represent symmetric relations.

See also: TransE, RotatE, Bilinear KG Model.

#### KG Entity

A node in a knowledge graph representing a distinct real-world object, concept, or instance, such as a person, place, or organization. Entities are the things that relations connect.

Learning entity embeddings is central to predicting missing facts in knowledge graphs.

**Example:** "Marie Curie" and "Nobel Prize" are entities in a knowledge graph linked by a relation.

See also: KG Relation, KG Triple.

#### KG Relation

A typed, directed edge label in a knowledge graph specifying the kind of connection between two entities, such as "bornIn" or "authorOf". Relations give knowledge graphs their multi-relational character.

Modeling relation-specific transformations is the core challenge of knowledge-graph embedding.

**Example:** The relation "capitalOf" links the entity "Paris" to the entity "France".

See also: KG Entity, KG Triple, Relation-Specific Weights.

#### KG Triple

The atomic fact in a knowledge graph, written as (head, relation, tail) or \( (h, r, t) \), asserting that a relation holds from a head entity to a tail entity. Triples are the units stored and predicted.

Knowledge-graph embedding models are trained to score observed triples higher than corrupted ones.

**Example:** The triple (Einstein, bornIn, Ulm) records that Einstein was born in Ulm.

See also: KG Completion, TransE.

#### Knowledge Graph

A graph-structured knowledge base in which nodes represent entities and directed, labeled edges represent typed relations between them, encoding facts as relationships. It captures structured, multi-relational world knowledge.

Knowledge graphs underpin search, question answering, and reasoning, and pose distinctive multi-relational learning challenges.

**Example:** A knowledge graph stores the fact "Einstein bornIn Ulm" as a relation edge between the Einstein and Ulm entities.

See also: KG Entity, KG Relation, KG Triple.

#### Label Propagation

A semi-supervised algorithm that spreads known node labels across edges to unlabeled nodes, iteratively setting each node's label distribution to a weighted average of its neighbors'. It assumes connected nodes share labels.

It provides a fast, parameter-free baseline exploiting homophily, and inspires propagation-based GNNs.

**Example:** Starting from a few labeled accounts, label propagation classifies the rest of a social network by repeatedly averaging neighbor labels.

See also: Homophily, APPNP.

#### LADIES Sampler

A layer-wise importance sampling method that, for each layer, samples a shared set of nodes for the whole mini-batch with probabilities favoring nodes connected to the previous layer's sampled nodes, controlling sample size and variance. It keeps the sampled computation dense and connected.

It addresses the redundancy and variance issues of naive per-node neighbor sampling.

**Example:** The LADIES sampler picks a fixed number of layer-two nodes likely to connect to the already-sampled layer-one nodes, avoiding wasted samples.

See also: Layer-Wise Sampling, Neighbor Sampling.

#### Laplacian Positional Encoding

A node feature derived from the eigenvectors of the graph Laplacian, providing each node with coordinates that encode its global position in the graph. It gives transformers and GNNs a sense of where nodes sit.

It supplies the positional information that self-attention, being permutation-symmetric, otherwise lacks on graphs.

**Example:** Concatenating the lowest-frequency Laplacian eigenvectors to node features lets a graph transformer distinguish nodes by global position.

See also: Random Walk Struct Encoding, Sign-Invariant Network, Graph Transformer.

#### Layer Depth (GNN)

The number of stacked message-passing layers in a GNN, which equals the number of aggregation rounds and thus the radius of each node's receptive field. Greater depth lets information travel farther.

Increasing depth expands reach but tends to induce over-smoothing and over-squashing.

**Example:** A four-layer GNN propagates information across four hops but risks blurring node representations together.

See also: Over-Smoothing, Receptive Field (GNN).

#### Layer Normalization (GNN)

A normalization technique that standardizes each node's representation across its feature dimensions independently of other nodes, making it insensitive to batch composition. It suits the variable batch and graph sizes common in GNNs.

It offers a batch-independent alternative to batch normalization for stabilizing GNN training.

**Example:** Layer normalization rescales each node's feature vector to zero mean and unit variance across its own dimensions.

See also: Batch Normalization (GNN), PairNorm.

#### Layer-Wise Sampling

A GNN sampling strategy that samples a fixed number of nodes per layer for an entire mini-batch, rather than independently expanding each node's neighbors, keeping the per-layer node count bounded. It prevents the exponential neighborhood growth of node-wise sampling.

It improves efficiency and reuse of sampled nodes across the batch.

**Example:** Layer-wise sampling selects 512 nodes at each layer that the whole batch shares, instead of separately sampling neighbors for every target node.

See also: LADIES Sampler, Neighbor Sampling.

#### LightGCN

A graph-based recommendation model that simplifies the graph convolutional network for collaborative filtering by removing feature transformations and nonlinearities, retaining only linear neighborhood aggregation over the user-item graph. The final embedding is a weighted sum across layers.

Its stripped-down design improves both accuracy and efficiency for recommendation.

**Example:** LightGCN refines user and item embeddings by repeatedly averaging neighbor embeddings on the interaction graph, with no learned weight matrices.

See also: Recommender System (Graph), Graph Convolutional Network.

#### LINE Embedding

A node-embedding method that directly optimizes embeddings to preserve first-order proximity (adjacent nodes) and second-order proximity (nodes with shared neighbors), without random walks. It scales to large networks via edge sampling.

It offers an explicit objective-based alternative to walk-based embeddings.

**Example:** LINE places two nodes with many common neighbors close together to preserve their second-order proximity.

See also: DeepWalk, Matrix Factorization (Graph).

#### Linear Threshold Model

A diffusion model in which each node activates once the cumulative weight of its already-active neighbors exceeds a node-specific random threshold. Influence spreads through accumulating peer pressure.

It captures adoption driven by the combined effect of multiple influencing neighbors.

**Example:** Under the linear threshold model, a person adopts a product only after enough of their friends have adopted it.

Contrast with: Independent Cascade Model.

#### Link Prediction

The task of predicting whether an edge should exist between two nodes, typically by scoring node-pair embeddings and ranking candidate links. It addresses incomplete or evolving graph structure.

It powers recommendation, knowledge-graph completion, and forecasting of future connections.

**Example:** A link-prediction model scores unobserved user pairs in a social network to suggest new friendships.

See also: Edge-Level Task, Binary Cross-Entropy (LP), Negative Sampling (LP).

#### Link Prediction (KG)

The knowledge-graph task of scoring and ranking candidate entities to complete a partial triple, equivalent to predicting missing relational edges. It is evaluated by how highly the correct entity ranks among candidates.

It is the operational form of knowledge-graph completion used in benchmarks.

**Example:** For the query (Paris, capitalOf, ?), knowledge-graph link prediction should rank "France" first among all entities.

See also: KG Completion, Hits@K Metric, Mean Reciprocal Rank (MRR).

#### LLM + GNN Integration

A family of methods combining large language models with graph neural networks, using the language model for text understanding or reasoning and the GNN for structural learning, on graphs whose nodes carry text. It unites semantic and relational signals.

It targets text-attributed graphs where both content and structure carry information.

**Example:** An LLM encodes each paper's abstract into features that a GNN then propagates across the citation graph for classification.

See also: Text-Attributed Graph, Graph Instruction Tuning.

#### Local Clustering Coefficient

A node-level measure equal to the fraction of pairs among a node's neighbors that are themselves connected, quantifying how close the node's neighborhood is to a clique. It ranges from 0 to 1.

It captures local cohesion and serves as a structural feature for nodes.

**Example:** A node whose neighbors are all mutually connected has a local clustering coefficient of 1.

See also: Transitivity, Clique.

#### Louvain Algorithm

A greedy, hierarchical community-detection method that iteratively moves nodes between communities to increase modularity, then aggregates communities into super-nodes and repeats. It scales to very large graphs.

Its speed and quality make it a standard baseline for finding communities.

**Example:** Running the Louvain algorithm on a large social graph quickly produces a multilevel hierarchy of nested communities.

See also: Modularity (Network), Girvan-Newman Algorithm.

#### Matrix Factorization (Graph)

A node-embedding technique that factorizes a graph proximity matrix, such as the adjacency or a Katz matrix, into a product of low-rank embedding matrices whose inner products approximate the proximities. It casts embedding as low-rank approximation.

It provides a linear-algebraic lens unifying many shallow embedding methods.

**Example:** Factorizing a node-proximity matrix as \( Z Z^\top \) yields embeddings \( Z \) whose dot products reconstruct proximity.

See also: Singular Value Decomp (SVD), Shallow Embedding.

#### Matrix Factorization (Rec)

A collaborative-filtering method that factorizes the user-item interaction matrix into low-dimensional user and item embedding matrices whose inner products approximate observed interactions. It learns latent preference factors.

It is the classical embedding-based baseline for recommendation, generalized by graph methods.

**Example:** Matrix factorization predicts a user's rating of a movie as the dot product of their learned user vector and the movie's item vector.

See also: Collaborative Filtering, LightGCN, Matrix Factorization (Graph).

#### Matrix Multiplication

A binary operation combining two matrices \( A \in \mathbb{R}^{m \times n} \) and \( B \in \mathbb{R}^{n \times p} \) into a product \( C \in \mathbb{R}^{m \times p} \), where each entry is the dot product of a row of \( A \) and a column of \( B \). It is associative and distributive but not commutative.

In graph machine learning, multiplying an adjacency matrix by a feature matrix propagates information along edges, forming the computational core of message passing.

**Example:** Computing \( A X \), where \( A \) is the adjacency matrix and \( X \) holds node features, sums each node's neighbor features in a single matrix multiplication.

#### Matrix Rank

The dimension of the vector space spanned by a matrix's rows or columns, equivalently the number of its nonzero singular values. It measures the amount of linearly independent information a matrix carries.

Low-rank structure in graph matrices justifies compressing them into compact embeddings.

**Example:** A community-structured adjacency matrix is approximately low rank, so a few embedding dimensions capture most of its connectivity.

#### Matrix Transpose

The operation that reflects a matrix across its main diagonal, mapping entry \( (i, j) \) to position \( (j, i) \), denoted \( A^\top \). A matrix equal to its transpose is symmetric.

In graph contexts, transposing a directed adjacency matrix reverses every edge direction, swapping in-edges and out-edges.

**Example:** For a directed citation graph, \( A^\top \) lets a node aggregate from papers that cite it rather than papers it cites.

#### Max Aggregation

A neighborhood aggregation operator that takes the element-wise maximum across neighbor messages, capturing the most salient feature value in each dimension. It emphasizes prominent neighbors and ignores their count.

It excels at detecting representative or extreme neighbor features but discards distributional detail.

**Example:** Max aggregation in GraphSAGE can flag the presence of a particular feature in any neighbor regardless of how many neighbors there are.

Contrast with: Sum Aggregation, Mean Aggregation.

#### Mean Aggregation

A neighborhood aggregation operator that averages the messages from a node's neighbors, normalizing away the effect of neighbor count. It captures the distribution of neighbor features but not their multiplicity.

It is robust to degree variation but cannot distinguish neighborhoods differing only in size.

**Example:** Mean aggregation gives the same result for a node with one neighbor of feature \( x \) as for one with three such neighbors.

Contrast with: Sum Aggregation, Max Aggregation.

#### Mean Reciprocal Rank (MRR)

A ranking evaluation metric equal to the average over queries of the reciprocal of the rank assigned to the correct answer, rewarding placing correct answers near the top. It ranges from 0 to 1.

It is a primary metric for knowledge-graph completion and link prediction.

**Example:** If the correct entity is ranked third for a query, that query contributes \( 1/3 \) to the mean reciprocal rank.

See also: Hits@K Metric, Link Prediction (KG).

#### Message Function

The component of a message-passing GNN that computes the information a node sends to or receives from a neighbor, typically as a function of the neighbor's representation and possibly the connecting edge. It defines what gets transmitted along each edge.

Designing the message function controls how edge features and neighbor states enter aggregation.

**Example:** A simple message function transforms a neighbor's feature vector by a learnable weight matrix before it is aggregated.

See also: Message Passing Neural Net, Aggregation Function.

#### Message Passing Neural Net

A general framework describing GNNs as repeated rounds in which each node computes messages from its neighbors, aggregates them, and updates its state, formalized by message, aggregation, and update functions. It encompasses most GNN variants.

It provides the common abstraction through which diverse GNN architectures are understood and implemented.

**Example:** In the message-passing framework, a graph convolutional layer sends each neighbor's transformed feature as a message and aggregates by a normalized sum.

See also: Message Function, Aggregation Function, Update Function.

#### Meta-Path

A composite relation defined as an ordered sequence of node and edge types connecting two endpoints in a heterogeneous graph, encoding a semantically meaningful indirect relationship. It defines typed higher-order neighborhoods.

Meta-paths let models reason about relationships that no single edge type expresses directly.

**Example:** The meta-path "author-paper-author" links two authors who co-wrote a paper.

See also: Heterogeneous Graph, HAN (Hetero Attention Net).

#### MinCutPool

A graph-pooling method that learns cluster assignments by optimizing a relaxed normalized minimum-cut objective alongside the task loss, coarsening the graph into well-separated clusters. It grounds hierarchical pooling in spectral clustering theory.

It produces pooling that respects community structure while remaining differentiable.

**Example:** MinCutPool groups densely connected regions of a graph into super-nodes guided by a min-cut criterion.

See also: DiffPool, Normalized Cut.

#### Mini-Batch Training (GNN)

A training strategy that updates GNN parameters using small subsets of nodes or graphs per step rather than the whole graph, enabling learning on graphs too large for full-batch computation. It requires sampling the relevant neighborhood for each batch.

It is essential for scaling GNNs beyond memory limits.

**Example:** Mini-batch training samples a few hundred target nodes and their neighborhoods per step instead of processing the entire million-node graph at once.

See also: Neighbor Sampling, Cluster-GCN, GraphSAINT.

#### Modularity (Network)

A scalar quality measure of a graph partition, equal to the fraction of edges falling within communities minus the expected fraction under a degree-preserving random model. Higher modularity indicates stronger community structure.

It serves as the objective that many community-detection algorithms optimize.

**Example:** A partition of a social network into tightly knit friend groups yields high modularity relative to a random rewiring.

See also: Community Detection, Louvain Algorithm.

#### Molecular Graph

A graph representation of a molecule in which nodes are atoms and edges are chemical bonds, each annotated with features such as element type, charge, and bond order. It is the standard input for chemistry GNNs.

Representing molecules as graphs lets GNNs predict chemical properties directly from structure.

**Example:** A water molecule is a molecular graph with one oxygen node bonded to two hydrogen nodes.

See also: Graph Classification, Drug Discovery with GNNs.

#### Molecule Generation

The task of producing novel, valid molecular graphs, often optimized for target properties such as binding affinity or solubility. It is a flagship application of graph generative models.

It promises to accelerate chemical and drug design by proposing candidate structures automatically.

**Example:** A generative model proposes new molecular graphs predicted to bind a disease target while remaining synthesizable.

See also: Graph Generative Model, GCPN, DiGress.

#### Multi-Head Attention (Graph)

An extension of graph attention that runs several independent attention computations ("heads") in parallel and concatenates or averages their outputs, stabilizing learning and capturing diverse relational patterns. Each head can focus on different neighbor aspects.

Multiple heads improve robustness and expressive capacity over a single attention function.

**Example:** A graph attention network with eight heads lets different heads attend to different subsets of a node's neighbors.

See also: Attention Mechanism (Graph), Graph Transformer.

#### Multi-Hop Query

A knowledge-graph query that requires traversing a chain of several relations to reach the answer, rather than a single relational step. It tests reasoning over relation composition.

Answering multi-hop queries on incomplete graphs motivates embedding-based reasoning that tolerates missing edges.

**Example:** "Cities in the country where Einstein was born" is a multi-hop query chaining "bornIn" and "locatedIn".

See also: Conjunctive Query, Composition (Relation), Query Embedding.

#### Multigraph

A graph that permits multiple distinct edges between the same pair of nodes, often representing repeated or differently typed interactions. Its adjacency structure must record edge multiplicity or identity.

Multigraphs capture settings where the same nodes interact in several ways simultaneously.

**Example:** A communication network where two people exchange both emails and phone calls is a multigraph with parallel edges.

See also: Heterogeneous Graph, Weighted Graph.

#### Negative Sampling

An efficient training technique that approximates an expensive softmax over all possible outputs by contrasting each observed positive pair with a few randomly sampled negative pairs. It dramatically speeds up embedding training.

It makes Skip-Gram-based graph embedding tractable on large vocabularies of nodes.

**Example:** Training node2vec, each true context node is contrasted against a handful of randomly drawn non-context nodes.

See also: Skip-Gram Model, Negative Sampling (LP).

#### Negative Sampling (LP)

A technique for link-prediction training that pairs each observed (positive) edge with one or more randomly sampled non-edges (negatives), making the otherwise overwhelmingly negative classification tractable and balanced. It avoids scoring all possible non-edges.

It is essential because real graphs are sparse, so true edges are vastly outnumbered.

**Example:** For each real friendship edge, negative sampling draws a random non-friend pair as a contrasting negative example.

See also: Binary Cross-Entropy (LP), Negative Sampling, Link Prediction.

#### Neighbor Sampling

A sampling method that, for each target node, draws a fixed-size random subset of its neighbors at each layer, bounding the computation per node and enabling mini-batch GNN training. It controls the exponential growth of multi-hop neighborhoods.

It is the technique that made inductive GNNs like GraphSAGE scalable.

**Example:** Neighbor sampling selects 10 neighbors per node at layer one and 10 of each of those at layer two, capping the receptive subgraph size.

See also: GraphSAGE, Mini-Batch Training (GNN), Layer-Wise Sampling.

#### Neighborhood Aggregation

The core GNN operation of combining information from a node's adjacent nodes into a summary used to update that node's representation. It is the mechanism by which structure influences learned features.

Repeated neighborhood aggregation lets information propagate across multiple hops of the graph.

**Example:** After one round of neighborhood aggregation, each node's representation reflects its own features and those of its direct neighbors.

See also: Aggregation Function, K-Hop Neighborhood.

#### Network Motif

A small connected subgraph pattern that recurs in a network significantly more often than expected in a randomized null model. Motifs are interpreted as functional building blocks.

Over-represented motifs hint at the local mechanisms shaping a network's organization.

**Example:** The feed-forward loop is an over-represented three-node motif in gene-regulation networks.

See also: Graphlet, Frequent Subgraph Mining.

#### NetworkX

A Python library for creating, manipulating, and analyzing graphs, offering algorithms for paths, centralities, and community detection on in-memory graph objects. It is a general-purpose graph-analysis toolkit.

It is commonly used for exploratory graph analysis and for loading small benchmark graphs.

**Example:** Calling NetworkX's `karate_club_graph()` loads Zachary's Karate Club network for analysis.

See also: SNAP, PyTorch Geometric (PyG).

#### Neural Bellman-Ford Net

A GNN-based knowledge-graph reasoning model that generalizes the Bellman-Ford shortest-path algorithm with learnable, relation-aware operators, computing path-based representations between entity pairs. It predicts links from the structure of paths connecting entities.

Its path-based, parameter-efficient design supports inductive reasoning on unseen entities.

**Example:** A Neural Bellman-Ford network scores a candidate fact by aggregating learnable representations of all relational paths between the two entities.

See also: Inductive KG Reasoning, Composition (Relation).

#### Neural Collaborative Filter

A recommendation model that replaces the inner-product scoring of matrix factorization with a neural network that learns a nonlinear interaction function between user and item embeddings. It increases the flexibility of preference modeling.

It showed that learned nonlinear interactions can outperform fixed dot-product scoring.

**Example:** A neural collaborative filter feeds concatenated user and item embeddings through a multilayer perceptron to predict an interaction score.

See also: Matrix Factorization (Rec), Collaborative Filtering.

#### Neural Network Layer

A parameterized transformation that maps an input vector or batch to an output, typically combining a linear map with a nonlinear activation. Stacking layers builds expressive function approximators.

In GNNs, one layer corresponds to one round of neighbor aggregation, so depth controls how far information travels.

**Example:** A single GNN layer transforms each node's representation using its features and those of its immediate neighbors.

See also: Layer Depth (GNN), Message Passing Neural Net.

#### Node (Vertex)

A fundamental entity in a graph representing an object such as a person, atom, or web page, connected to others by edges. Nodes may carry feature vectors and labels.

Nodes are the units over which GNNs compute representations and make predictions.

**Example:** In a molecular graph, each node is an atom carrying features such as element type and charge.

See also: Edge (Link), Attribute Graph.

#### Node Classification

The task of assigning a categorical label to each node, learned from a subset of labeled nodes and the graph structure. It is the most common node-level benchmark.

It evaluates how well a model combines node features with neighborhood information under homophily.

**Example:** Classifying each paper in a citation network into a subject area is node classification.

See also: Node-Level Task, Cross-Entropy (Node Class), Transductive Learning.

#### Node Degree

The number of edges incident to a node in an undirected graph, equal to the count of its neighbors. It is a basic measure of a node's local connectivity.

Degree appears in Laplacian normalization and helps explain why high-degree hubs behave differently under aggregation.

**Example:** The instructor and administrator nodes in the Karate Club graph have the highest degrees, reflecting their central social roles.

See also: Degree Distribution, In-Degree, Out-Degree.

#### Node Embedding

A learned mapping of each graph node to a low-dimensional continuous vector that preserves chosen aspects of its structural role or proximity. The vectors enable standard machine learning on graph nodes.

Embeddings turn discrete graph structure into features usable by downstream classifiers and predictors.

**Example:** A node embedding places two structurally similar users close together in vector space, easing link prediction.

See also: Embedding Space, Shallow Embedding, Encoder-Decoder Framework.

#### Node Feature Normalization

The preprocessing of node feature vectors, for example scaling each feature to zero mean and unit variance or normalizing each node's vector to unit norm, so features are on comparable scales before training. It stabilizes and speeds up optimization.

Consistent feature scales help GNN training converge and prevent dominant features from skewing aggregation.

**Example:** Node feature normalization rescales raw bag-of-words counts so no single high-frequency term dominates the input to the GNN.

See also: Batch Normalization (GNN), Layer Normalization (GNN).

#### Node-Level Task

A graph learning task whose prediction target is a property of individual nodes, such as a class label or continuous value. Predictions are made per node within one or more graphs.

It is one of the three canonical granularities of graph machine learning tasks.

**Example:** Predicting each user's interest category in a social network is a node-level task.

See also: Node Classification, Edge-Level Task, Graph-Level Task.

#### node2vec

A node-embedding method extending DeepWalk with a biased random walk controlled by parameters that interpolate between breadth-first and depth-first exploration, balancing local and global structure. It then applies Skip-Gram to the walks.

Its tunable walk bias lets embeddings emphasize community membership or structural role as desired.

**Example:** Tuning node2vec toward breadth-first walks produces embeddings that group tightly knit communities together.

See also: Biased Random Walk, BFS Strategy (node2vec), DFS Strategy (node2vec).

#### Normalized Cut

A graph-partitioning objective that minimizes the total weight of edges cut between groups while normalizing by each group's connectivity, avoiding trivially small partitions. Its relaxation is solved by spectral clustering.

It formalizes balanced partitioning and motivates the use of Laplacian eigenvectors for clustering.

**Example:** Minimizing the normalized cut splits an image's pixel graph into balanced, coherent regions for segmentation.

See also: Spectral Clustering, Graph Laplacian.

#### Normalized Graph Laplacian

A rescaled graph Laplacian, such as the symmetric form \( I - D^{-1/2} A D^{-1/2} \), whose eigenvalues lie in a bounded range independent of node degrees. Normalization makes spectral quantities comparable across graphs.

Its bounded spectrum stabilizes spectral filters and motivates the normalization in graph convolutional networks.

**Example:** The symmetric normalized graph Laplacian has eigenvalues in \( [0, 2] \), regardless of the graph's degree distribution.

See also: Graph Laplacian, Spectral Graph Convolution.

#### One-For-All (OFA) Model

A unified graph-learning framework that represents nodes, edges, and tasks across different graphs in a shared text-attributed space, enabling a single model to handle classification, link prediction, and graph-level tasks across domains. It seeks one model for all graph tasks.

It exemplifies the push toward general, cross-domain graph foundation models.

**Example:** A One-For-All model trained jointly on citation, molecular, and knowledge graphs answers tasks on each using a single set of parameters.

See also: Graph Foundation Model, In-Context Learning (Graphs).

#### Open Graph Benchmark

A collection of large-scale, realistic graph datasets with standardized splits and evaluation procedures for node, link, and graph-level tasks. It enables rigorous, reproducible comparison of graph methods.

It raised the bar for GNN evaluation beyond small, easily saturated benchmarks.

**Example:** The Open Graph Benchmark's ogbn-arxiv dataset poses node classification over a 170,000-paper citation network with a fixed split.

See also: TUDataset, SNAP.

#### Order Embedding

An embedding scheme that maps objects into a space equipped with a partial order, so that a containment or subset relationship is encoded geometrically, for instance one region dominating another coordinate-wise. It represents asymmetric "is-contained-in" relations.

It enables neural detection of subgraph relationships by turning containment into geometry.

**Example:** An order embedding places a subgraph's vector coordinate-wise below its supergraph's vector, encoding the containment relation.

See also: SPMiner, Subgraph Isomorphism.

#### Out-Degree

The number of edges directed out of a node in a directed graph, counting outgoing relationships. It measures how many other nodes a node points to.

Out-degree characterizes prolific referencing behavior, such as a survey paper citing many works.

**Example:** A survey article has high out-degree in a citation graph because it cites many earlier papers.

Contrast with: In-Degree.

#### Over-Smoothing

A degradation in deep GNNs whereby repeated neighborhood aggregation makes node representations converge toward indistinguishable values, erasing the differences needed for prediction. It worsens with increasing depth.

It explains why naively stacking many GNN layers often hurts rather than helps performance.

**Example:** After many layers, an over-smoothed GNN assigns nearly identical embeddings to all nodes in a connected component.

See also: PairNorm, Layer Depth (GNN), Over-Squashing.

#### Over-Squashing

A failure in which information from a node's exponentially growing distant neighborhood is compressed into a fixed-size vector as it passes through narrow graph bottlenecks, distorting long-range signals. It limits a GNN's ability to use far-away information.

It motivates graph rewiring, virtual nodes, and transformer-style global attention.

**Example:** In a tree-like graph, over-squashing prevents a root node from receiving undistorted information from its many distant leaves.

See also: GNN Bottleneck, Virtual Node Augmentation, Graph Transformer.

#### Overlapping Community

A community structure in which nodes may simultaneously belong to more than one community, reflecting multiple group memberships. It generalizes hard partitions into soft, shared memberships.

Overlapping models better capture real affiliations, where individuals participate in several groups at once.

**Example:** A person belonging to both a family circle and a workplace group occupies overlapping communities.

See also: BigCLAM Model, Community Detection.

#### PageRank

A node-importance algorithm that models a random surfer who follows edges and occasionally teleports to a random node, ranking nodes by their stationary visit probability. It is a damped variant of eigenvector centrality.

Originally designed to rank web pages, it remains a foundational graph-importance and proximity measure.

**Example:** PageRank assigns higher scores to web pages linked by many other high-scoring pages.

See also: Personalized PageRank, Random Walk, Teleportation (PageRank).

#### PairNorm

A normalization layer for GNNs that rescales node representations to keep the total pairwise feature distance constant across layers, directly counteracting the convergence of embeddings caused by over-smoothing. It preserves node distinguishability in deep models.

It enables training deeper GNNs without representations collapsing together.

**Example:** Inserting PairNorm between deep GNN layers keeps node embeddings spread apart rather than converging to a single point.

See also: Over-Smoothing, Layer Normalization (GNN).

#### Permutation Equivariance

The property that permuting a function's input elements permutes its outputs in the same way, characterizing how each GNN layer maps node features to node features. It composes through layers and resolves into invariance only at readout.

It ensures that per-node computations respect the absence of any canonical node ordering.

**Example:** A GNN layer is permutation-equivariant: swapping two nodes' positions swaps their output representations accordingly.

See also: Permutation Invariance, Graph Equivariance.

#### Permutation Invariance

The property that a function's output is unaffected by any reordering of its input elements, as required of GNN aggregation and graph-level readout over unordered neighbor sets. Sum, mean, and max are invariant operations.

It is the formal reason aggregation must not depend on the arbitrary order in which neighbors are listed.

**Example:** Sum aggregation is permutation-invariant because adding neighbor features yields the same result regardless of their order.

See also: Permutation Equivariance, Aggregation Function.

#### Personalized PageRank

A variant of PageRank in which the random surfer's teleportation jumps return to a fixed source node or set rather than a uniform distribution, producing importance scores relative to that source. It measures proximity to the seed.

It provides query-specific node rankings and underlies several propagation-based GNNs.

**Example:** Personalized PageRank from a single user node ranks other users by their relevance to that user, useful for recommendations.

See also: PageRank, APPNP.

#### PinSage

A large-scale graph recommendation model that combines GraphSAGE-style neighbor sampling with random-walk-based importance weighting to generate item embeddings on web-scale bipartite graphs. It was designed for billions of nodes in production.

It demonstrated that GNN-based recommendation scales to industrial graphs.

**Example:** PinSage embeds pins on a content-discovery platform by aggregating sampled neighbors weighted by random-walk visit counts.

See also: GraphSAGE, Neighbor Sampling, Recommender System (Graph).

#### Planar Graph

A graph that can be drawn in the plane so that no two edges cross. Planarity constrains structure and enables specialized efficient algorithms.

Planar structure arises in spatial and circuit graphs and bounds certain combinatorial properties.

**Example:** A map of countries, drawn as a graph with edges between bordering nations, is planar.

#### Position-Aware GNN

A GNN variant that augments node representations with information about their position in the graph, typically by measuring distances to a set of anchor nodes, so structurally identical nodes in different locations become distinguishable. It overcomes a limitation of standard message passing.

It enables tasks, such as link prediction, that require knowing where nodes sit relative to others.

**Example:** A position-aware GNN distinguishes two otherwise-identical nodes by their differing shortest-path distances to chosen anchor nodes.

See also: Identity-Aware GNN, Laplacian Positional Encoding.

#### Positive Semi-Definite Matrix

A symmetric matrix \( A \) for which \( x^\top A x \ge 0 \) holds for every vector \( x \), equivalently a matrix whose eigenvalues are all nonnegative. Such matrices define valid quadratic forms and kernels.

The graph Laplacian is positive semi-definite, guaranteeing nonnegative graph frequencies and a meaningful notion of signal smoothness.

**Example:** Because the Laplacian \( L \) is positive semi-definite, the smoothness measure \( x^\top L x \) is always nonnegative for any node signal \( x \).

#### Power Iteration

An iterative numerical method that repeatedly multiplies a vector by a matrix and renormalizes, converging to the matrix's dominant eigenvector. It avoids full eigendecomposition.

It is the practical algorithm for computing PageRank and other dominant-eigenvector centralities on large graphs.

**Example:** PageRank is computed by power iteration: repeatedly applying the transition matrix to a score vector until it stabilizes.

See also: PageRank, Eigenvector.

#### Power-Law Degree Distribution

A degree distribution of the form \( P(k) \propto k^{-\gamma} \), in which the probability of a node having degree \( k \) decays polynomially, producing many low-degree nodes and rare high-degree hubs. It indicates a heavy-tailed, scale-free structure.

Such distributions appear in many real networks and motivate models that reproduce hub formation.

**Example:** The World Wide Web's in-degree approximately follows a power law, with a few pages receiving an enormous share of links.

See also: Scale-Free Network, Preferential Attachment.

#### Pre-Training (GNN)

The initial training of a GNN on a large dataset or self-supervised objective to learn general representations before adapting it to a specific downstream task. It front-loads learning of broadly useful features.

It underlies transfer learning and graph foundation models.

**Example:** Pre-training a GNN with a self-supervised masked-feature objective on many molecules yields a model later fine-tuned for toxicity prediction.

See also: Fine-Tuning (GNN), Self-Supervised Learning, Transfer Learning (Graphs).

#### Preferential Attachment

A network growth mechanism in which new nodes connect to existing nodes with probability proportional to those nodes' current degree, so well-connected nodes attract still more connections. It produces "rich-get-richer" dynamics.

It is the generative principle that yields power-law degree distributions.

**Example:** A new researcher is more likely to cite already-famous papers, reinforcing their citation counts through preferential attachment.

See also: Barabási–Albert Model, Power-Law Degree Distribution.

#### Principal Neighborhood Agg

A GNN aggregation scheme that combines several aggregators (such as mean, max, min, and standard deviation) together with degree-based scalers, capturing complementary statistics of a neighborhood that any single aggregator would miss. It improves expressiveness and robustness across degree distributions.

It addresses the limitation that no single aggregator captures all relevant neighborhood information.

**Example:** Principal neighborhood aggregation concatenates mean, max, and standard-deviation summaries of a node's neighbors, scaled by degree, before updating.

See also: Aggregation Function, Sum Aggregation.

#### Protein-Protein Interaction

A graph machine-learning application centered on networks whose nodes are proteins and edges denote physical or functional interactions, used for tasks like function prediction and interaction inference. It is a standard inductive GNN benchmark domain.

It exemplifies node- and edge-level learning on biological networks.

**Example:** A GNN predicts a protein's function by aggregating signals over its protein-protein interaction neighborhood.

See also: Drug-Drug Interaction, Node Classification.

#### PyKEEN

A Python library for training and evaluating knowledge-graph embedding models, providing many model implementations, datasets, and standardized link-prediction evaluation. It centralizes knowledge-graph embedding experimentation.

It enables consistent comparison of models like TransE, DistMult, and RotatE.

**Example:** A researcher uses PyKEEN to train RotatE on a knowledge graph and report Hits@10 and mean reciprocal rank.

See also: TransE, RotatE, Mean Reciprocal Rank (MRR).

#### PyTorch Autograd

The automatic differentiation engine in PyTorch that records tensor operations in a dynamic computation graph and computes gradients by reverse traversal when `backward()` is called. It powers training without manual gradient derivation.

It transparently differentiates the irregular, data-dependent computation graphs that arise in message passing.

**Example:** Calling `loss.backward()` triggers autograd to populate `.grad` on every GNN parameter tensor.

See also: Automatic Differentiation, PyTorch Tensor.

#### PyTorch Geometric (PyG)

A library built on PyTorch that provides data structures, datasets, sampling utilities, and message-passing primitives for implementing graph neural networks. It standardizes GNN development on top of PyTorch.

It is among the most widely used frameworks for building and training GNNs.

**Example:** A developer subclasses PyTorch Geometric's message-passing base class to implement a custom GNN layer in a few lines.

See also: PyTorch Tensor, DGL.

#### PyTorch Tensor

A multidimensional array data structure in the PyTorch library that stores numerical data, supports GPU acceleration, and tracks operations for gradient computation. It is the basic unit of computation in PyTorch models.

Node features, adjacency information, and embeddings in GNN code are all represented as tensors.

**Example:** A node feature matrix of shape \( [N, F] \) is a PyTorch tensor passed into a GNN layer.

See also: PyTorch Autograd, PyTorch Geometric (PyG).

#### Query Embedding

An approach to answering complex knowledge-graph queries by embedding the entire query, including its logical operators, into the same space as entities so that answers are nearby points. It enables differentiable, scalable reasoning over incomplete graphs.

It generalizes single-triple link prediction to multi-step logical questions.

**Example:** A query embedding for "universities located in Europe" maps to a region of entity space whose nearby points are the answer universities.

See also: Box Embedding (Query2Box), Multi-Hop Query, Conjunctive Query.

#### R-GCN (Relational GCN)

A heterogeneous GNN that extends the graph convolutional network with a separate learnable weight matrix per relation type, aggregating neighbors grouped by the relation linking them. It handles multi-relational graphs like knowledge graphs.

Its per-relation parameters can proliferate, motivating basis decomposition to control their number.

**Example:** An R-GCN classifies entities in a knowledge graph by aggregating neighbors separately for each relation type with relation-specific weights.

See also: Relation-Specific Weights, Basis Decomposition (R-GCN), Heterogeneous GNN.

#### Random Walk

A stochastic process that traverses a graph by repeatedly moving from the current node to a randomly chosen neighbor. The sequence of visited nodes reveals structural proximity.

Random walks generate the node-context pairs used by embedding methods and define proximity in PageRank.

**Example:** Sampling many short random walks from each node produces neighbor sequences that DeepWalk treats like sentences.

See also: DeepWalk, Stationary Distribution.

#### Random Walk Struct Encoding

A node positional/structural encoding built from the probabilities that a random walk returns to the node after various numbers of steps, capturing local structural roles. It complements eigenvector-based encodings.

It supplies transformers with structure-aware features that are stable and easy to compute.

**Example:** A random walk structural encoding records a node's return probabilities at steps one through eight as extra features.

See also: Laplacian Positional Encoding, GPS (General Powerful GNN).

#### Receptive Field (GNN)

The set of input nodes that can influence a given node's final representation in a GNN, determined by the number of message-passing layers. With \( k \) layers it equals the node's \( k \)-hop neighborhood.

Understanding the receptive field clarifies how depth controls the scope of information a node sees.

**Example:** Adding a third layer to a GNN expands each node's receptive field from its two-hop to its three-hop neighborhood.

See also: K-Hop Neighborhood, Layer Depth (GNN).

#### Recommender System (Graph)

A recommendation approach that models users, items, and their interactions as a graph and uses graph learning to predict relevant items for each user. It treats recommendation as link prediction on a user-item graph.

Graph structure captures higher-order collaborative signals beyond direct interactions.

**Example:** A graph recommender predicts movies for a user by propagating preferences across the user-movie interaction graph.

See also: Collaborative Filtering, LightGCN, PinSage.

#### Relation-Specific Weights

Distinct learnable parameter sets assigned per relation type in a heterogeneous or relational GNN, so that each relation transforms neighbor messages differently. They preserve the distinct semantics of each relation.

Their number grows with the relation count, creating overfitting and scaling concerns that decomposition addresses.

**Example:** In an R-GCN, the "cites" and "authoredBy" relations each get their own weight matrix applied during aggregation.

See also: R-GCN (Relational GCN), Basis Decomposition (R-GCN).

#### Relational Database as Graph

A modeling perspective that treats a relational database's rows as nodes and its foreign-key references as edges, yielding a heterogeneous graph over which GNNs can learn directly from multi-table data. It avoids manual feature engineering across tables.

It is the foundation of relational deep learning, applying GNNs to enterprise tabular data.

**Example:** A retail database becomes a graph where customer rows link to order rows, which link to product rows via foreign keys.

See also: Relational Deep Learning, RelBench, Heterogeneous Graph.

#### Relational Deep Learning

A paradigm that applies deep learning, particularly GNNs, directly to relational databases by treating linked tables as a heterogeneous graph, learning predictive models without manual feature engineering across tables. It brings end-to-end learning to multi-table data.

It reframes common business prediction tasks as graph learning over database schemas.

**Example:** Relational deep learning predicts customer churn directly from a graph built from linked customer, order, and product tables.

See also: Relational Database as Graph, RelBench, RelGNN.

#### Relative Positional Encoding

A positional-encoding scheme that represents the relationship between two elements by their relative distance rather than absolute positions, applied in graph transformers via pairwise structural distances. It generalizes naturally across graphs of different sizes.

It supplies attention with distance-aware structure without committing to absolute node coordinates.

**Example:** A relative positional encoding biases attention between two nodes by a learned function of their shortest-path distance.

See also: Spatial Bias (Graph Attention), Laplacian Positional Encoding.

#### RelBench

A benchmark suite for relational deep learning that provides relational databases, predefined predictive tasks, and standardized evaluation for graph models operating on multi-table data. It enables comparable progress in the area.

It supplies the datasets and protocols against which relational GNNs are measured.

**Example:** RelBench includes an e-commerce database with tasks like predicting future purchases, evaluated under a temporal split.

See also: Relational Deep Learning, RelGNN, RelBench (Benchmark).

#### RelBench (Benchmark)

The benchmark component of the relational deep learning ecosystem, supplying relational databases, predictive tasks, and standardized temporal evaluation protocols for models that learn over multi-table data as graphs. It defines the empirical standard for the area.

It allows reproducible comparison of relational GNNs on realistic database tasks.

**Example:** A relational deep learning model is evaluated on RelBench by predicting future events from a temporally split relational database.

See also: Relational Deep Learning, RelBench, RelGNN.

#### RelGNN

A graph neural network architecture tailored to relational databases, designed to respect the schema's table types, foreign-key relationships, and temporal structure when learning over the database-derived graph. It specializes message passing for relational data.

It targets the distinctive heterogeneity and temporality of database graphs.

**Example:** RelGNN aggregates information across a database's customer-order-product links to predict each customer's next purchase.

See also: Relational Deep Learning, Relational Database as Graph, RelBench.

#### Residual Connection (GNN)

A skip connection that adds a layer's input to its output, so the layer learns a residual update rather than a full transformation. It stabilizes gradients and supports deeper architectures.

In GNNs, residual connections help counter the representation degradation caused by stacking many aggregation layers.

**Example:** A residual GNN layer computes \( h^{(k)} = h^{(k-1)} + f(h^{(k-1)}, \text{neighbors}) \), preserving prior information.

See also: Skip Connection (GNN), Over-Smoothing.

#### ROC-AUC Score

The area under the receiver operating characteristic curve, equal to the probability that a randomly chosen positive instance is ranked above a randomly chosen negative one, summarizing classifier ranking quality across thresholds. It ranges from 0.5 (random) to 1 (perfect).

It is a common metric for link prediction and binary node classification, robust to class imbalance.

**Example:** A link predictor with ROC-AUC 0.92 ranks a true edge above a random non-edge 92% of the time.

See also: Link Prediction, Binary Cross-Entropy (LP).

#### RotatE

A knowledge-graph embedding model that represents each relation as a rotation in complex vector space, training so that the tail equals the head rotated by the relation, \( t \approx h \circ r \) with \( |r| = 1 \). Rotation captures rich relational patterns.

It can model symmetry, antisymmetry, inversion, and composition simultaneously.

**Example:** RotatE represents "inverse" relations as opposite rotations, so applying a relation and then its inverse returns the original entity.

See also: TransE, Composition (Relation), KG Embedding Geometry.

#### SAN (Spectral Attention Net)

A graph transformer that incorporates learned Laplacian-eigenvector positional encodings into its attention so that nodes attend with awareness of their spectral position in the graph. It uses the graph's spectrum to inform global attention.

It illustrates the spectral approach to giving transformers a sense of graph structure.

**Example:** SAN feeds Laplacian eigenvector encodings into attention so spectrally distant nodes are treated differently.

See also: Laplacian Positional Encoding, Graph Transformer.

#### Scale-Free Network

A network whose degree distribution follows a power law, characterized by many low-degree nodes and a few highly connected hubs, with structure invariant across scales. Hubs dominate its connectivity.

Scale-free structure makes networks robust to random failures but vulnerable to targeted hub attacks.

**Example:** The Internet's autonomous-system graph is approximately scale-free, dominated by a small number of large hubs.

See also: Power-Law Degree Distribution, Barabási–Albert Model.

#### Scene Graph

A graph representation of a visual scene in which nodes are detected objects and edges encode their spatial or semantic relationships, such as "person riding bicycle". It structures image content relationally.

Scene graphs bridge computer vision and graph learning for relational image understanding.

**Example:** A scene graph for a photo links a "person" node to a "horse" node via a "riding" relation edge.

See also: Knowledge Graph, Heterogeneous Graph.

#### Self-Supervised Learning

A learning paradigm that derives training signals from the data itself by defining pretext tasks, requiring no human-provided labels, to learn general representations. On graphs it exploits structure and features as supervision.

It enables pretraining of GNNs on abundant unlabeled graph data.

**Example:** A self-supervised GNN learns node representations by predicting masked node features, using no external labels.

See also: Deep Graph Infomax (DGI), Graph Contrastive Learning, Pre-Training (GNN).

#### Shallow Embedding

A node-embedding approach in which each node's vector is a free parameter stored in a lookup table, learned directly without a feature-based encoder. It cannot generalize to nodes unseen during training.

Its transductive nature and lack of feature sharing motivate inductive, feature-based GNNs.

**Example:** DeepWalk and node2vec are shallow embeddings because they assign each node an independently optimized vector.

Contrast with: Inductive Learning, Graph Neural Network (GNN).

#### Shortest Path

A path between two nodes that minimizes total length, measured by edge count in unweighted graphs or summed weights in weighted graphs. It defines graph distance between the nodes.

Shortest paths underlie centrality measures and characterize how efficiently information can traverse a network.

**Example:** Breadth-first search finds the shortest unweighted path between two members of a social network.

See also: Betweenness Centrality, Breadth-First Search.

#### SIGN Architecture

A scalable GNN architecture that precomputes multiple fixed graph-operator-transformed feature matrices (such as different powers of the adjacency) offline, then trains a simple feed-forward model on their concatenation, avoiding message passing during training. It shifts graph computation to preprocessing.

This decoupling allows fast, mini-batchable training on very large graphs.

**Example:** The SIGN architecture precomputes one-, two-, and three-hop aggregated features for every node, then trains a multilayer perceptron on their concatenation.

See also: Historical Embeddings (SIGN), Simple Graph Conv (SGC), Mini-Batch Training (GNN).

#### Sign-Invariant Network

A neural module designed so its output is unchanged when the sign of an input eigenvector is flipped, resolving the arbitrary sign ambiguity of Laplacian eigenvectors used as positional encodings. It produces well-defined spectral features.

It ensures that spectral positional encodings yield consistent representations despite eigenvector sign indeterminacy.

**Example:** A sign-invariant network processes a Laplacian eigenvector so that \( v \) and \( -v \) give identical positional encodings.

See also: Basis-Invariant Network, Laplacian Positional Encoding.

#### Simple Graph Conv (SGC)

A simplified GNN that collapses multiple graph convolution layers into a single fixed linear operation by precomputing a power of the normalized adjacency applied to features, followed by one logistic-regression classifier. It removes intermediate nonlinearities.

It demonstrates that much of a GCN's performance comes from neighborhood smoothing rather than depth or nonlinearity.

**Example:** SGC precomputes \( \hat{A}^k X \) once, then trains a single linear classifier, matching deeper GCNs on many benchmarks at a fraction of the cost.

See also: Graph Convolutional Network, APPNP.

#### Singular Value Decomp (SVD)

A factorization of any matrix \( M \) into \( U \Sigma V^\top \), where \( U \) and \( V \) are orthogonal and \( \Sigma \) is diagonal with nonnegative singular values. It generalizes eigendecomposition to non-square matrices.

In graph ML, truncated SVD of an adjacency or proximity matrix produces low-dimensional node embeddings.

**Example:** Applying truncated SVD to a node-similarity matrix yields shallow embeddings, an approach generalized by methods like DeepWalk.

See also: Matrix Factorization (Graph), Shallow Embedding.

#### SIR Epidemic Model

A compartmental contagion model on a network in which each node is Susceptible, Infected, or Recovered, with infection spreading along edges and recovered nodes becoming immune. It captures one-time epidemic dynamics.

It links network structure to outbreak size and informs both disease and information-spread analysis.

**Example:** Simulating the SIR model on a contact network estimates how an outbreak's size depends on the network's connectivity.

See also: Independent Cascade Model, Influence Maximization.

#### Skip Connection (GNN)

A connection in a GNN that forwards a layer's input directly to a later point, combining it with the transformed output so earlier representations are preserved alongside aggregated ones. It eases training of deeper models.

Skip connections mitigate over-smoothing by retaining a node's pre-aggregation identity.

**Example:** Adding skip connections lets a deeper GNN retain each node's original features rather than washing them out across layers.

See also: Residual Connection (GNN), Jumping Knowledge Network.

#### Skip-Gram Model

A neural language model that learns word embeddings by predicting the surrounding context words from a target word, maximizing co-occurrence likelihood. It is adapted in graph embedding to predict context nodes from a target node.

It is the engine that converts random-walk sequences into node vectors in DeepWalk and node2vec.

**Example:** Applied to random walks, the Skip-Gram model learns node vectors that predict which nodes co-occur within a walk window.

See also: DeepWalk, Negative Sampling.

#### Small-World Network

A network exhibiting both high local clustering and short average path lengths between nodes, so most nodes are reachable in few hops despite dense local structure. It interpolates between regular and random graphs.

The small-world property explains why information and influence spread rapidly across real networks.

**Example:** A social network where friends cluster tightly yet any two people are linked by a short chain is small-world.

See also: Graph Diameter, Transitivity.

#### SNAP

The Stanford Network Analysis Platform, a system and dataset repository providing large real-world network datasets and efficient analysis tools. It is a longstanding source of benchmark graphs.

It supplies many of the real networks used in graph-learning research.

**Example:** Researchers download a large social network from the SNAP repository to benchmark a community-detection method.

See also: NetworkX, Open Graph Benchmark.

#### Social Network Analysis

The study of relationships among social actors using graph methods to measure influence, detect communities, and predict ties. It applies structural graph concepts to human and organizational interaction.

It is a major application area driving node classification, community detection, and link prediction.

**Example:** Social network analysis identifies influential users in a platform by computing centralities and detecting community structure.

See also: Community Detection, Betweenness Centrality.

#### Softmax Function

A function mapping a vector of real scores to a probability distribution, \( \text{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j} \), so outputs are positive and sum to one. It converts raw logits into class probabilities.

It produces normalized class scores for node and graph classification and normalizes attention coefficients over neighbors.

**Example:** A graph attention layer applies softmax over a node's neighbors so their attention weights sum to one.

See also: Cross-Entropy Loss, Attention Mechanism (Graph).

#### Spanning Tree

A subgraph of a connected graph that is a tree and includes every node, connecting them with the minimum number of edges. A weighted graph may have a minimum spanning tree of least total weight.

Spanning trees provide minimal connected backbones useful in network design and analysis.

**Example:** A minimum spanning tree of a road network connects all cities with the least total road length.

See also: Tree (Graph Theory).

#### Spatial Bias (Graph Attention)

An additive term in a graph transformer's attention scores that depends on the structural distance between two nodes, biasing attention toward or against nodes at particular graph distances. It injects topology into otherwise structure-blind attention.

It is the mechanism by which Graphormer encodes shortest-path information into global attention.

**Example:** Spatial bias lowers the attention weight between two atoms that are many bonds apart in a molecule.

See also: Graphormer, Relative Positional Encoding.

#### Spatial Domain (Graph)

The representation in which graph operations are defined directly over nodes and their local neighborhoods, aggregating information along edges without reference to eigenvectors. It contrasts with frequency-based formulations.

Most modern, scalable GNNs operate in the spatial domain via message passing.

**Example:** A graph convolutional network, though spectrally derived, is implemented in the spatial domain as normalized neighbor aggregation.

Contrast with: Spectral Domain (Graph).

#### Spectral Clustering

A clustering method that embeds nodes using the leading eigenvectors of the graph Laplacian and then applies a standard clustering algorithm such as k-means in that space. It leverages spectral structure to find groups.

It connects community detection to the eigenstructure that also underlies spectral graph convolutions.

**Example:** Spectral clustering of a graph's Laplacian eigenvectors recovers communities that are not linearly separable in the original space.

See also: Graph Laplacian, Normalized Cut.

#### Spectral Domain (Graph)

The representation of a graph signal in terms of its projections onto the eigenvectors of the graph Laplacian, analogous to the frequency domain of classical signals. Each eigenvalue plays the role of a graph frequency.

Operating in this domain enables principled definition of graph filters and convolutions.

**Example:** Transforming a node signal into the spectral domain reveals how much of it is smooth versus oscillatory across the graph.

Contrast with: Spatial Domain (Graph).

#### Spectral Graph Convolution

A convolution defined in the graph Fourier domain, where a signal is transformed by the eigenvectors of the graph Laplacian, filtered by a learned function of the eigenvalues, and transformed back. It generalizes Fourier-domain filtering to graphs.

It is the theoretical foundation from which spatial GNN layers like the graph convolutional network are derived.

**Example:** A spectral graph convolution applies a learned low-pass filter to suppress high-frequency variation across a graph signal.

See also: Spectral Domain (Graph), Chebyshev Polynomial Conv, Graph Laplacian.

#### SPMiner

A neural framework for frequent subgraph mining that embeds subgraphs into an order-embedding space to efficiently estimate pattern frequency and search for frequent patterns. It replaces exact, intractable counting with learned approximation.

It demonstrates how representation learning can accelerate classic combinatorial graph mining.

**Example:** SPMiner identifies frequent network motifs by counting how many target subgraphs are dominated by a pattern in order-embedding space.

See also: Order Embedding, Frequent Subgraph Mining.

#### Stationary Distribution

The long-run probability distribution over nodes visited by a random walk, invariant under one more step of the walk, satisfying \( \pi = \pi P \) for transition matrix \( P \). It describes where the walk spends time at equilibrium.

PageRank scores are precisely the stationary distribution of the teleporting random walk.

**Example:** Running a random walk for many steps and recording visit frequencies approximates the stationary distribution, equal to PageRank under teleportation.

See also: PageRank, Power Iteration.

#### Stochastic Depth (GNN)

A regularization technique that randomly skips entire GNN layers during training, letting information bypass them via identity connections, which both regularizes and eases training of deep networks. The full depth is used at inference.

In GNNs it can also reduce over-smoothing by limiting the number of aggregation rounds applied to some examples.

**Example:** Stochastic depth randomly drops a middle GNN layer during a training step, passing representations through unchanged.

See also: Skip Connection (GNN), Over-Smoothing, DropEdge.

#### Strongly Connected Component

A maximal set of nodes in a directed graph such that every node is reachable from every other node by following edge directions. Strongly connected components capture mutual reachability.

They reveal tightly coupled cyclic regions in directed networks such as the web.

**Example:** A cluster of web pages that all link to one another, directly or indirectly, forms a strongly connected component.

Contrast with: Weakly Connected Component.

#### Structural Equivalence

A notion of node similarity based on nodes occupying the same structural role, such as having similar connectivity patterns, even if they are far apart or share no neighbors. It contrasts with proximity-based similarity.

It clarifies what embedding methods capture: position-based versus role-based similarity.

**Example:** Two department managers in different parts of an organization chart are structurally equivalent despite no direct link.

Contrast with: Homophily.

#### Subgraph

A graph formed by selecting a subset of a larger graph's nodes together with some or all edges among them. It isolates a region of interest for analysis or computation.

Subgraphs define the local neighborhoods sampled during scalable GNN training and the patterns sought in mining.

**Example:** The induced subgraph of a node and its immediate neighbors forms that node's one-hop computation neighborhood.

See also: Ego Network, Subgraph GNN.

#### Subgraph GNN

A GNN that represents a graph as a collection of subgraphs, each obtained by a perturbation such as marking or deleting a node, runs a base GNN on each, and pools the results to gain expressiveness beyond 1-WL. It boosts distinguishing power through subgraph multiplicity.

It offers a practical route to higher expressiveness without full \( k \)-tuple computation.

**Example:** A subgraph GNN that processes one node-marked subgraph per node can distinguish graphs that confuse standard GNNs.

See also: Identity-Aware GNN, Higher-Order GNN.

#### Subgraph Isomorphism

The problem of determining whether a given pattern graph appears as a subgraph within a larger target graph, and locating its occurrences. It is NP-complete in general.

It underlies frequent subgraph mining and pattern matching, motivating learned approximations.

**Example:** Subgraph isomorphism checks whether a benzene-ring pattern occurs within a larger molecular graph.

See also: Frequent Subgraph Mining, Order Embedding.

#### Sum Aggregation

A neighborhood aggregation operator that adds the messages from all of a node's neighbors into a single vector. It retains information about both neighbor features and neighbor count.

Among basic aggregators it is the most expressive, underpinning the design of the graph isomorphism network.

**Example:** Sum aggregation lets a GNN distinguish a node with two identical neighbors from one with a single such neighbor.

Contrast with: Mean Aggregation, Max Aggregation.

#### Symmetric Matrix

A square matrix \( A \) equal to its own transpose, so \( A = A^\top \) and \( A_{ij} = A_{ji} \) for all entries. Symmetric real matrices have real eigenvalues and orthogonal eigenvectors.

The adjacency matrix of an undirected graph is symmetric, a property exploited by spectral methods that require real-valued eigendecompositions.

**Example:** The adjacency matrix of the undirected Karate Club graph is symmetric, since every friendship is mutual.

Contrast with: Graph (Directed).

#### Symmetry (Relation Pattern)

A relational pattern in which a relation holds in both directions, so \( r(x, y) \) implies \( r(y, x) \), as for "married_to". Whether a knowledge-graph model can express it is a key design criterion.

Models differ in their ability to capture symmetry, which affects which facts they can correctly infer.

**Example:** DistMult naturally represents the symmetric relation "spouseOf" because its scoring is symmetric in head and tail.

Contrast with: Antisymmetry (Relation).

#### Teleportation (PageRank)

The mechanism in PageRank by which the random surfer, with a fixed probability per step, jumps to a node drawn from a teleportation distribution instead of following an outgoing edge. It guarantees a unique, well-defined stationary distribution.

Teleportation prevents the walk from becoming trapped in sink nodes or disconnected regions.

**Example:** With a typical damping factor of 0.85, the surfer teleports to a random page 15% of the time, escaping dead-end pages.

See also: PageRank, Personalized PageRank.

#### Temporal GNN (TGN)

A class of GNNs for continuous-time dynamic graphs that maintains an evolving memory state per node, updated by timestamped interaction events and combined with message passing for prediction. It captures both temporal history and structure.

It enables learning on event streams where the timing of interactions is informative.

**Example:** A temporal GNN updates a user's memory vector each time they interact, then predicts their next likely connection.

See also: Dynamic Graph, TGAT, Temporal Graph.

#### Temporal Graph

A graph whose structure and/or features change over time, recorded either as a sequence of snapshots or a stream of timestamped events. It captures the time-evolving nature of real networks.

Modeling temporal graphs requires architectures that integrate structural and temporal information.

**Example:** A social network with timestamped friend-requests is a temporal graph whose edges appear over time.

See also: Dynamic Graph, Temporal GNN (TGN), Traffic Forecasting (Graph).

#### Text-Attributed Graph

A graph in which each node is associated with raw text, such as a document or description, in addition to structural connections. It combines natural-language content with relational topology.

It is the natural setting for integrating language models with graph neural networks.

**Example:** A citation network where each node carries its paper's title and abstract is a text-attributed graph.

See also: LLM + GNN Integration, Attribute Graph.

#### TGAT

A temporal graph attention network that encodes continuous time using a functional time-encoding and applies self-attention over a node's temporal neighborhood to produce time-aware embeddings. It treats interaction timestamps as continuous inputs.

It extends attention-based aggregation to capture when, not just whether, interactions occurred.

**Example:** TGAT embeds a node at a given time by attending over its past interactions, weighting them by their recency via time encoding.

See also: Temporal GNN (TGN), Graph Attention Network (GAT).

#### Tool-Use Graph

A representation of an agent's available tools and their input-output dependencies as a graph, enabling the agent to plan multi-tool workflows by reasoning over how tool outputs feed subsequent tools. It structures tool composition relationally.

It supports planning and orchestration when solving a task requires chaining several tools.

**Example:** A tool-use graph links a "search" tool's output to a "summarize" tool's input, letting an agent plan to search then summarize.

See also: Agent Memory as Graph.

#### Traffic Forecasting (Graph)

A graph machine-learning application that predicts future traffic conditions over a road network modeled as a graph of sensors or intersections, combining spatial graph structure with temporal dynamics. It is a benchmark for spatio-temporal GNNs.

It exemplifies the union of graph and time-series modeling.

**Example:** A spatio-temporal GNN forecasts highway speeds by propagating sensor readings over the road graph across time steps.

See also: Temporal GNN (TGN), Temporal Graph.

#### Train/Val/Test Split (Ind)

A data-splitting scheme for inductive learning in which validation and test nodes (or whole graphs) are entirely held out, unseen during training, testing generalization to new data. Test structure is not available at training time.

It evaluates a model's ability to embed and predict on genuinely novel nodes or graphs.

**Example:** An inductive split trains a GNN on one set of protein interaction graphs and evaluates it on separate graphs never seen during training.

See also: Inductive Learning, Train/Val/Test Split (Trans).

#### Train/Val/Test Split (Trans)

A data-splitting scheme for transductive learning in which all nodes of a single graph are present during training, but their labels are partitioned into training, validation, and test sets. The graph structure is shared across all splits.

It reflects the transductive setting where test nodes are known at training time but their labels are withheld.

**Example:** On the Cora citation graph, a transductive split labels some papers for training and hides the labels of validation and test papers within the same graph.

See also: Transductive Learning, Train/Val/Test Split (Ind).

#### Transductive Learning

A learning setting in which the model has access to the entire graph, including the unlabeled test nodes, during training, and makes predictions only for those specific nodes. It cannot generalize to entirely new nodes.

It characterizes shallow embeddings and standard single-graph node classification benchmarks.

**Example:** Classifying unlabeled members of the Karate Club, which were present in the graph during training, is transductive.

Contrast with: Inductive Learning.

#### TransE

A knowledge-graph embedding model that represents each relation as a translation in vector space, training so that head plus relation approximately equals tail, \( h + r \approx t \). It is simple, efficient, and influential.

Its translational geometry elegantly models composition and inversion but struggles with symmetric and one-to-many relations.

**Example:** TransE learns vectors where the embedding of "France" minus "Paris" approximates the "capitalOf" relation vector.

See also: TransR, RotatE, KG Embedding Geometry.

#### Transfer Learning (Graphs)

The reuse of knowledge learned on one graph or task to improve learning on a different but related graph or task, typically by transferring a pretrained model's parameters. It reduces data and compute needs on the target.

It is the practical goal motivating graph pretraining and foundation models.

**Example:** A GNN pretrained on a large molecular dataset is transferred to predict properties on a smaller, specialized molecule collection.

See also: Pre-Training (GNN), Fine-Tuning (GNN).

#### Transitivity

The global tendency of a network's connected triples to close into triangles, measured as the ratio of three times the number of triangles to the number of connected node triples. It quantifies overall clustering.

High transitivity reflects the social principle that a friend's friend tends also to be a friend.

**Example:** A friendship network with high transitivity contains many closed triangles where mutual acquaintances also know each other.

See also: Local Clustering Coefficient, Clique.

#### TransR

A knowledge-graph embedding model that, unlike TransE, projects entities into a relation-specific subspace before applying a translation, so each relation operates in its own space. It captures relations that act on different aspects of entities.

It increases expressiveness over TransE at the cost of more parameters per relation.

**Example:** TransR projects "Einstein" and "Ulm" into a "bornIn"-specific space, where the translation \( h + r \approx t \) holds.

See also: TransE, Type-Specific Projection.

#### Tree (Graph Theory)

A connected, undirected graph containing no cycles, equivalently a connected graph with exactly \( N - 1 \) edges for \( N \) nodes. Between any two nodes there is a unique path.

Trees model hierarchies and arise as the unrolled computation graphs of message-passing GNNs.

**Example:** The computation graph that a GNN builds around a node by unrolling its neighborhood is a tree.

See also: Spanning Tree, Cycle.

#### TUDataset

A collection of benchmark datasets for graph classification, comprising many small labeled graphs from domains such as chemistry, biology, and social networks. It is a standard testbed for graph-level methods.

Its many small graphs make it the classic benchmark for graph kernels and graph-classification GNNs.

**Example:** A graph classification model is evaluated on TUDataset's PROTEINS collection, where each graph represents a protein.

See also: Graph Classification, Graph Kernel.

#### Type-Specific Projection

A transformation that maps features of different node or relation types into a common space before joint processing, accommodating that distinct types may have differently structured features. It harmonizes heterogeneous inputs.

It is a standard first step in heterogeneous GNNs so that typed representations become comparable.

**Example:** A heterogeneous GNN applies a separate linear projection to author features and paper features before they interact.

See also: Heterogeneous GNN, TransR.

#### ULTRA

A foundation model for knowledge-graph reasoning that learns transferable representations of relational structure, conditioning on relation interactions so a single pretrained model generalizes to new knowledge graphs with unseen entities and relations. It aims for zero-shot transfer across graphs.

It moves knowledge-graph reasoning toward reusable, pretrained foundation models.

**Example:** ULTRA, pretrained on several knowledge graphs, performs link prediction on an entirely new graph without per-graph training.

See also: Inductive KG Reasoning, KG Foundation Model context, InGram.

#### Update Function

The component of a message-passing GNN that produces a node's new representation by combining its previous representation with the aggregated neighbor message, often via a neural layer with nonlinearity. It integrates self-information with neighborhood information.

The update function determines how a node balances its own state against incoming neighbor signals.

**Example:** An update function concatenates a node's prior embedding with its aggregated neighbor vector and passes the result through a linear layer and ReLU.

See also: Message Passing Neural Net, GraphSAGE.

#### Variational Autoencoder (VGAE)

A graph generative model that encodes nodes into a probabilistic latent space using a GNN encoder and reconstructs the graph's adjacency by decoding inner products of latent vectors. It learns a generative latent representation of graph structure.

It provides a principled probabilistic embedding useful for link prediction and graph generation.

**Example:** A variational graph autoencoder embeds nodes into a latent Gaussian space and reconstructs edges from the dot products of their latent samples.

See also: Graph Generative Model, Link Prediction.

#### Virtual Edge Augmentation

A graph-augmentation technique that inserts additional edges, such as connections between distant or structurally related nodes, to shorten path lengths and ease information flow. It alters the graph topology used for message passing.

It can relieve bottlenecks but risks introducing spurious relational signals.

**Example:** Adding virtual edges between two-hop neighbors lets a single GNN layer reach information normally two hops away.

See also: Virtual Node Augmentation, Over-Squashing.

#### Virtual Node Augmentation

A graph-augmentation technique that adds an artificial node connected to every real node, providing a global communication shortcut through which information can flow between distant nodes in one hop. It enlarges the effective receptive field cheaply.

It helps mitigate over-squashing and improves graph-level prediction by enabling global pooling within message passing.

**Example:** Adding a virtual node to a molecular graph lets distant atoms exchange information in a single message-passing step.

See also: Over-Squashing, Virtual Edge Augmentation.

#### Weakly Connected Component

A maximal set of nodes in a directed graph that becomes connected when all edge directions are ignored. It groups nodes linked by any chain of edges regardless of orientation.

Weak components describe overall reachability when direction is irrelevant.

**Example:** Two web pages linked only by a one-way hyperlink lie in the same weakly connected component but possibly different strongly connected ones.

Contrast with: Strongly Connected Component.

#### Weighted Graph

A graph in which each edge carries a numerical weight quantifying the strength, capacity, or cost of the relationship. Weights generalize the binary presence of an edge.

Edge weights modulate how strongly neighbors influence one another during aggregation.

**Example:** In a road network, edge weights represent travel times between intersections.

See also: Adjacency Matrix, Edge (Link).

#### Weisfeiler-Lehman (WL) Test

A classical algorithm that tests for graph non-isomorphism by iteratively refining node labels into hashes of their neighborhoods' labels and comparing the resulting label histograms of two graphs. It can prove two graphs are different but not always that they are the same.

It is the standard yardstick for the discriminative power of message-passing GNNs.

**Example:** The WL test relabels nodes by hashing their neighbor labels each round, distinguishing most but not all non-isomorphic graphs.

See also: 1-WL Test, GNN Expressiveness, Graph Isomorphism.

#### Weisfeiler-Lehman Kernel

A graph kernel that compares graphs by iteratively relabeling each node with a hash of its own and its neighbors' labels, then counting matching relabels across iterations. It mirrors the Weisfeiler-Lehman isomorphism test.

It is both a strong classical baseline and the conceptual precursor to GNN expressiveness analysis.

**Example:** The Weisfeiler-Lehman kernel deems two molecules similar when their iterated neighbor-label histograms largely coincide.

See also: Weisfeiler-Lehman (WL) Test, Graph Isomorphism Network.

#### 1-WL Test

The original, one-dimensional Weisfeiler-Lehman test, which refines each node's label using the multiset of its immediate neighbors' labels. It is the exact expressive upper bound for standard message-passing GNNs.

A GNN can distinguish two graphs only if the 1-WL test can, which motivates more powerful architectures.

**Example:** The 1-WL test fails to distinguish certain regular graphs, and so do standard message-passing GNNs.

See also: k-WL Test, Graph Isomorphism Network.

