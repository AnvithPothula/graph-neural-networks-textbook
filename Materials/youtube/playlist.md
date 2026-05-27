# YouTube Playlist: Stanford CS224W (2021)

**Playlist:** https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn
**Channel:** Stanford Online
**Total videos:** 60
**Offering:** Fall 2021 (the full public lecture series)

Each lecture was split into multiple short segments (typically 15–30 min each).
These videos correspond to the 2021 syllabus in `../syllabus/2021-syllabus.md`.

---

## How to Use This for Textbook Generation

When generating chapter content, you can reference individual lecture segments for:
- Jure Leskovec's specific explanations and intuitions
- The exact slide content and worked examples
- The order in which concepts are introduced

Search the playlist for a specific lecture using YouTube's search within playlist, or filter by the lecture number below.

---

## Lecture Video Breakdown (2021 Offering)

### Lecture 1 — Introduction: Machine Learning for Graphs
- 1.1 — Why graphs? Real-world applications (social, biological, knowledge networks)
- 1.2 — Different types of graphs; the tasks we care about
- 1.3 — Roadmap of the course

### Lecture 2 — Traditional Methods for ML on Graphs
- 2.1 — Node-level features: degree, centrality, clustering coefficient
- 2.2 — Link-level features: common neighbors, Jaccard, Adamic-Adar, Katz
- 2.3 — Graph-level features: graphlet kernel, WL kernel

### Lecture 3 — Node Embeddings
- 3.1 — Encoder-decoder framework; what makes a good embedding?
- 3.2 — DeepWalk: random walk objectives and skip-gram
- 3.3 — node2vec: biased random walks; p and q parameters

### Lecture 4 — Link Analysis: PageRank
- 4.1 — PageRank: flow formulation and matrix formulation
- 4.2 — Handling spider traps and dead ends; random teleportation
- 4.3 — Personalized PageRank; SimRank

### Lecture 5 — Label Propagation for Node Classification
- 5.1 — Semi-supervised learning; message passing for classification
- 5.2 — Belief propagation; collective classification framework

### Lecture 6 — Graph Neural Networks 1: GNN Model
- 6.1 — Limitations of shallow embeddings; need for deep graph encoders
- 6.2 — Neighborhood aggregation framework; computation graphs
- 6.3 — Matrix formulation of GNNs; how to train
- 6.4 — GraphSAGE; unsupervised training; inductive learning

### Lecture 7 — Graph Neural Networks 2: Design Space
- 7.1 — General GNN framework; message, aggregate, update
- 7.2 — GNN layers: linear, activation, batch norm, dropout
- 7.3 — Layer connectivity: stack, skip, jumping knowledge

### Lecture 8 — Applications of Graph Neural Networks
- 8.1 — Node classification, link prediction, graph classification applications
- 8.2 — Case studies: drug interaction, scene graphs, combinatorial optimization

### Lecture 9 — Theory of Graph Neural Networks
- 9.1 — GNN expressiveness; distinguishing graph structures
- 9.2 — Weisfeiler-Lehman graph isomorphism test
- 9.3 — GIN: most expressive MPNN; sum vs. mean vs. max aggregation

### Lecture 10 — Knowledge Graph Embeddings
- 10.1 — Knowledge graphs: structure, tasks, datasets (Freebase, Wikidata)
- 10.2 — TransE, TransR: translational models for KG embeddings
- 10.3 — Bilinear models: DistMult, ComplEx, RotatE; relation pattern analysis

### Lecture 11 — Reasoning over Knowledge Graphs
- 11.1 — Multi-hop reasoning; path queries over incomplete KGs
- 11.2 — Query embedding: embed queries as geometric objects
- 11.3 — Query2Box: box embeddings for conjunctive queries

### Lecture 12 — Frequent Subgraph Mining with GNNs
- 12.1 — Subgraph isomorphism; network motifs; graphlets
- 12.2 — Neural subgraph matching with GNNs (order embeddings)
- 12.3 — SPMiner: mining frequent subgraphs with GNNs

### Lecture 13 — GNNs for Recommender Systems
- 13.1 — Collaborative filtering; matrix factorization; NGCF
- 13.2 — LightGCN: simplified graph convolution for recommendation; PinSage

### Lecture 14 — Community Structure in Networks
- 14.1 — Network communities; modularity; Louvain algorithm
- 14.2 — Spectral community detection; graph partitioning
- 14.3 — Overlapping communities; BigCLAM model

### Lecture 15 — Deep Generative Models for Graphs
- 15.1 — Challenges of graph generation; evaluation metrics
- 15.2 — GraphRNN: sequential graph generation with RNNs
- 15.3 — GCPN: goal-directed molecular generation with RL + GNN

### Lecture 16 — Advanced Topics on GNNs
- 16.1 — GNNs beyond message passing; higher-order GNNs
- 16.2 — Scalable training: neighbor sampling, cluster sampling, GraphSAINT

### Lecture 17 — Scaling Up GNNs
- 17.1 — The scalability challenge: full-batch vs mini-batch
- 17.2 — Subgraph sampling methods; theoretical guarantees
- 17.3 — Cluster-GCN; GraphSAINT; practical comparison

### Lecture 18 — Guest Lecture: Petar Veličković
- Petar Veličković (DeepMind): GNNs as Algorithmic Reasoners
- Topic: How GNNs can learn to execute classical algorithms
- Slides: https://petar-v.com/talks/5G-CS224W.pdf

### Lecture 19 — Design Space of Graph Neural Networks / Conclusion
- Summary of the GNN design space: 5 dimensions
- Trends in graph learning
- Open research problems

---

## Additional Context: Why 60 Videos for 19 Lectures?

Each lecture was recorded in 3–4 segments of 15–30 minutes each. This was done to:
1. Make individual concepts searchable on YouTube
2. Enable asynchronous viewing at natural breakpoints
3. Allow students to replay specific concepts

When searching the playlist, the video titles follow the pattern:
`"CS224W: Machine Learning with Graphs | 2021 | Lecture X.Y - [Topic]"`

---

## Direct Playlist Search Tips

To find a specific topic quickly:
1. Open the playlist: https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn
2. Use Ctrl+F to search within the playlist page
3. Or search on YouTube: `CS224W 2021 [topic]`

Examples:
- "CS224W 2021 GIN" → finds Lecture 9.3 on Graph Isomorphism Networks
- "CS224W 2021 PageRank" → finds Lecture 4
- "CS224W 2021 knowledge graph" → finds Lectures 10 and 11
