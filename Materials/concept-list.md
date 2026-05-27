# GNN Textbook: Concept List (~260 Key Concepts)

This file lists the key concepts for the learning graph generator. Each concept belongs to one of the taxonomy categories below.

Used as input to `/skill learning-graph-generator`.

**Target:** 260+ concepts covering a broader scope than CS224W, including prerequisite math, classical graph theory topics dropped from the 2025 course, and modern frontier topics.

---

## Taxonomy Categories

1. **Prerequisites** — math, probability, linear algebra, PyTorch basics
2. **Graph Fundamentals** — basic graph theory and properties
3. **Classical Graph Algorithms** — PageRank, centrality, community detection
4. **Node Embeddings** — shallow embedding methods
5. **GNN Architecture** — GNN model components and variants
6. **GNN Theory** — expressiveness, limitations, theoretical results
7. **Graph Transformers** — attention-based graph architectures
8. **Knowledge Graphs** — KG embeddings and reasoning
9. **Heterogeneous Graphs** — multi-relational and typed graphs
10. **Applications** — recommender systems, drug discovery, temporal graphs
11. **Scalability** — sampling, mini-batching, distributed training
12. **Generative Models** — graph generation methods
13. **Advanced Topics** — LLMs+GNNs, foundation models, agents
14. **Training & Optimization** — loss functions, regularization, augmentation
15. **Tools & Frameworks** — PyTorch Geometric, NetworkX, OGB, benchmarks

---

## Concept List

### Prerequisites
1. Matrix multiplication
2. Matrix transpose
3. Eigenvalue decomposition
4. Eigenvector
5. Symmetric matrix
6. Positive semi-definite matrix
7. Singular value decomposition (SVD)
8. Matrix rank
9. Dot product
10. Cosine similarity
11. Gradient descent
12. Backpropagation
13. Chain rule (calculus)
14. Automatic differentiation
15. PyTorch tensor
16. PyTorch autograd
17. Neural network layer (torch.nn.Module)
18. Softmax function
19. Cross-entropy loss
20. Adam optimizer

### Graph Fundamentals
21. Graph (undirected)
22. Graph (directed / digraph)
23. Node (vertex)
24. Edge (link)
25. Adjacency matrix
26. Degree (node degree)
27. In-degree
28. Out-degree
29. Degree distribution
30. Power-law degree distribution
31. Path (graph path)
32. Shortest path
33. Diameter (graph)
34. Connected component
35. Strongly connected component
36. Weakly connected component
37. Bipartite graph
38. Heterogeneous graph
39. Multigraph
40. Weighted graph
41. Attribute graph (feature-rich graph)
42. Subgraph
43. Ego network
44. Clique
45. Cycle
46. Tree (graph theory)
47. Spanning tree
48. Planar graph
49. Graph isomorphism
50. Graph homomorphism
51. Small-world network
52. Scale-free network
53. Erdős–Rényi random graph
54. Barabási–Albert model
55. Preferential attachment
56. Giant component
57. Transitivity (global clustering coefficient)

### Classical Graph Algorithms
58. Clustering coefficient (local)
59. Betweenness centrality
60. Closeness centrality
61. Eigenvector centrality
62. PageRank
63. Personalized PageRank (PPR)
64. HITS algorithm (Hubs and Authorities)
65. Random walk
66. Stationary distribution
67. Power iteration
68. Teleportation (PageRank damping)
69. Community detection
70. Modularity (network)
71. Louvain algorithm
72. Girvan-Newman algorithm
73. Spectral clustering
74. Normalized cut
75. Overlapping community
76. BigCLAM model
77. Network motif
78. Graphlet
79. Graphlet degree vector (GDV)
80. Graph kernel
81. Weisfeiler-Lehman kernel
82. Label propagation
83. Belief propagation
84. Influence maximization
85. Linear threshold model
86. Independent cascade model
87. SIR epidemic model
88. Breadth-first search (BFS)
89. Depth-first search (DFS)
90. Katz similarity

### Node Embeddings
91. Node embedding
92. Embedding space
93. Encoder-decoder framework (graph)
94. Shallow embedding
95. Matrix factorization (graph)
96. DeepWalk
97. node2vec
98. Biased random walk (node2vec)
99. BFS strategy (node2vec)
100. DFS strategy (node2vec)
101. Skip-gram model
102. Negative sampling
103. LINE (Large-scale Information Network Embedding)
104. Transductive learning
105. Inductive learning
106. Structural equivalence
107. Homophily

### GNN Architecture
108. Graph Neural Network (GNN)
109. Message passing neural network (MPNN)
110. Message function
111. Aggregation function
112. Update function
113. Graph Convolutional Network (GCN)
114. GraphSAGE
115. Graph Attention Network (GAT)
116. Attention mechanism (graph)
117. Multi-head attention (graph)
118. Graph Isomorphism Network (GIN)
119. Sum aggregation
120. Mean aggregation
121. Max aggregation
122. Neighborhood aggregation
123. K-hop neighborhood
124. Receptive field (GNN)
125. Layer depth (GNN)
126. Skip connection (GNN)
127. Residual connection (GNN)
128. Jumping Knowledge Network (JK-Net)
129. Graph-level readout
130. Global mean pooling
131. Global sum pooling
132. DiffPool (differentiable pooling)
133. MinCutPool
134. Node-level task
135. Edge-level task
136. Graph-level task
137. Link prediction
138. Node classification
139. Graph classification
140. Graph regression
141. Spectral graph convolution
142. Chebyshev polynomial convolution
143. Graph Laplacian
144. Normalized graph Laplacian
145. Spectral domain (graph)
146. Spatial domain (graph)
147. Virtual node augmentation
148. Virtual edge augmentation

### GNN Theory
149. Weisfeiler-Lehman (WL) test
150. 1-WL test
151. k-WL test
152. GNN expressiveness
153. Graph isomorphism problem
154. Distinguishing power (GNN)
155. Over-smoothing
156. Over-squashing
157. Bottleneck (GNN)
158. Equivariance (graph)
159. Invariance (graph)
160. Permutation invariance
161. Permutation equivariance
162. Position-aware GNN
163. Identity-aware GNN (ID-GNN)
164. Higher-order GNN
165. Subgraph GNN

### Graph Transformers
166. Graph Transformer
167. Graphormer
168. SAN (Spectral Attention Network)
169. GPS (General, Powerful, Scalable GNN)
170. Laplacian eigenvector (positional encoding)
171. Random walk structural encoding (RWSE)
172. Sign-invariant network (SignNet)
173. Basis-invariant network (BasisNet)
174. Spatial bias (graph attention)
175. Centrality encoding (Graphormer)
176. Edge encoding (Transformer)
177. Relative positional encoding (graph)

### Knowledge Graphs
178. Knowledge graph (KG)
179. Entity (KG)
180. Relation (KG)
181. Triple (KG)
182. KG completion
183. Link prediction (KG)
184. TransE
185. TransR
186. DistMult
187. ComplEx
188. RotatE
189. Geometric interpretation (KG embeddings)
190. Bilinear model (KG)
191. Symmetry (relation pattern)
192. Antisymmetry (relation pattern)
193. Inversion (relation pattern)
194. Composition (relation pattern)
195. Query embedding
196. Box embedding (Query2Box)
197. Multi-hop query
198. Conjunctive query
199. Neural Bellman-Ford Network (NBFNet)
200. Inductive KG reasoning
201. ULTRA (Universal Transferable Reasoning)
202. InGram (inductive KG embedding)

### Heterogeneous Graphs
203. Heterogeneous GNN
204. R-GCN (Relational GCN)
205. Heterogeneous Graph Transformer (HGT)
206. HAN (Heterogeneous Attention Network)
207. Meta-path
208. Relation-specific weight matrix
209. Basis decomposition (R-GCN)
210. Type-specific projection

### Applications
211. Recommender system (graph-based)
212. Collaborative filtering
213. Matrix factorization (recommendation)
214. Neural Collaborative Filtering (NCF)
215. LightGCN
216. PinSage
217. Drug-drug interaction prediction
218. Protein-protein interaction network
219. Molecular graph
220. Drug discovery with GNNs
221. Social network analysis
222. Fraud detection (graph)
223. Traffic forecasting (graph)
224. Scene graph
225. Relational database as graph
226. Temporal graph
227. Dynamic graph
228. Temporal GNN (TGN)
229. TGAT (Temporal Graph Attention)
230. Frequent subgraph mining
231. Subgraph isomorphism
232. Order embedding (subgraph matching)
233. SPMiner

### Scalability
234. Mini-batch training (GNN)
235. Neighbor sampling
236. GraphSAINT
237. Cluster-GCN
238. LADIES sampler
239. Layer-wise sampling
240. Historical embeddings (SIGN)
241. Graph partitioning

### Generative Models
242. Graph generative model
243. GraphRNN
244. GCPN (Graph Convolutional Policy Network)
245. Molecule generation
246. Variational autoencoder (graph / VGAE)
247. DiGress (discrete diffusion for graphs)
248. Evaluation metrics for graph generation (validity, uniqueness, novelty, FCD)

### Advanced Topics
249. In-context learning on graphs (PRODIGY)
250. Conformalized GNN (uncertainty quantification)
251. Graph foundation model
252. LLM + GNN integration
253. Text-attributed graph
254. Graph instruction tuning
255. One-for-All (OFA) graph model
256. Agent memory as graph
257. Tool-use graph (agent workflows)
258. Relational Deep Learning (RDL)
259. RelBench
260. RelGNN

### Training & Optimization
261. Cross-entropy loss (node classification)
262. Binary cross-entropy (link prediction)
263. Negative sampling (link prediction)
264. Data augmentation (graph)
265. Dropout (GNN)
266. Batch normalization (GNN)
267. Layer normalization (GNN)
268. DropEdge
269. PairNorm
270. Early stopping (GNN)
271. Train/val/test split (transductive)
272. Train/val/test split (inductive)
273. Curriculum learning (graph)

### Tools & Frameworks
274. PyTorch Geometric (PyG)
275. NetworkX
276. Open Graph Benchmark (OGB)
277. SNAP (Stanford Network Analysis Project)
278. DeepSNAP
279. DGL (Deep Graph Library)
280. TUDataset
281. PyKEEN (KG embeddings library)
282. RelBench (relational benchmark)
