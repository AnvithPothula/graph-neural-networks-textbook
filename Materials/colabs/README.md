# CS224W Colab Assignments

6 hands-on Colab notebooks that build practical GNN skills progressively. These are the primary coding materials for the textbook's code examples and can be referenced when generating chapter content.

---

## Colab 0 — Introduction to Graph ML Libraries

**Out:** Week 1 (with Lecture 2)
**Topic:** NetworkX and PyTorch Geometric basics

**Direct links:**
- [2025 Colab 0](https://drive.google.com/file/d/1JAJeJMR_djDtiKy-NbFkIMke72Sw-LeR/view?usp=sharing)
- [2021 Colab 0](https://colab.research.google.com/drive/16tqEHKOLUgYvXKx1V3blfYGpQb1_09MG?usp=sharing)
- [AndrewSpano solution](https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_0.ipynb)

**Skills covered:**
- Create and manipulate graphs with NetworkX (directed, undirected, weighted)
- Compute graph statistics: degree distribution, clustering coefficient, connected components
- Introduction to PyTorch Geometric (PyG): `Data`, `Dataset`, `DataLoader` objects
- Explore the KarateClub dataset
- Visualize graphs and their properties

---

## Colab 1 — Node Embeddings

**Out:** Week 1 | **Due:** Week 3 (after Theory of GNNs)
**Topic:** Learning node embeddings from scratch; shallow embedding methods

**Direct links:**
- [2025 Colab 1](https://colab.research.google.com/drive/1cNsHg6NClQyZiQEgRDCKoqofiik3y1XN?usp=sharing)
- [2021 Colab 1](https://colab.research.google.com/drive/1p2s0on6nibUYhJnONBWEAwpBlue37Tcc?usp=sharing)
- [AndrewSpano solution](https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_1.ipynb)

**Skills covered:**
- Implement a shallow embedding (lookup table) using PyTorch
- Train node embeddings on KarateClub to predict community membership
- Manual gradient descent on embedding parameters
- Visualize embeddings in 2D using PCA/t-SNE
- Understand the encoder-decoder framework for graph embeddings

---

## Colab 2 — Graph Neural Networks

**Out:** Week 3 | **Due:** Week 5 (before Knowledge Graphs)
**Topic:** GCN and GraphSAGE on OGB benchmarks (node classification + graph classification)

**Direct links:**
- [2025 Colab 2](https://colab.research.google.com/drive/1DqySwyevHcM7OE1Sh3xWGyKD0Jcr95R5)
- [2021 Colab 2](https://colab.research.google.com/drive/1BRPw3WQjP8ANSFz-4Z1ldtNt9g7zm-bv?usp=sharing)
- [AndrewSpano solution](https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_2.ipynb)

**Skills covered:**
- Build GCN and GraphSAGE models using PyTorch Geometric
- **Node classification** on `ogbn-arxiv` (citation network, ~170K nodes)
- **Graph classification** on `ogbg-molhiv` (molecular property prediction, ~41K graphs)
- Use the Open Graph Benchmark (OGB) evaluation pipeline
- Track performance with official leaderboard metrics

---

## Colab 3 — GraphSAGE, GAT, and Edge Prediction

**Out:** Week 5 | **Due:** Week 7 (before Advanced Architectures)
**Topic:** Node classification with GraphSAGE/GAT; link prediction; intro to DeepSNAP

**Direct links:**
- [2025 Colab 3](https://colab.research.google.com/drive/11F8K9lnVlGRNOeFWfyfeOim0NdxOdtae?usp=sharing)
- [2021 Colab 3](https://colab.research.google.com/drive/1bAvutxJhjMyNsbzlLuQybzn_DXM63CuE)
- [AndrewSpano solution](https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_3.ipynb)

**Skills covered:**
- Implement GraphSAGE and GAT on the Cora citation dataset
- **Link prediction** on Cora: predict held-out edges using dot product + MLP head
- Introduction to DeepSNAP library for flexible graph task handling
- Compare GCN vs. GraphSAGE vs. GAT on node classification

---

## Colab 4 — Heterogeneous Graphs

**Out:** Week 7 | **Due:** Week 10 (before Final Project)
**Topic:** Heterogeneous GNNs with DeepSNAP on ACM academic network

**Direct links:**
- [2025 Colab 4](https://colab.research.google.com/drive/1AaNEIaIZhRNMueJDdrnNLdwiYuwwfFP9)
- [2021 Colab 4](https://colab.research.google.com/drive/1X4uOWv_xkefDu_h-pbJg-fEkMfR7NGz9?usp=sharing)
- [AndrewSpano solution](https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_4.ipynb)

**Skills covered:**
- Build a heterogeneous graph from the ACM dataset (paper, author, subject nodes)
- Implement Heterogeneous GCN and GAT using DeepSNAP's `HeteroGraph`
- Node property prediction on heterogeneous graphs
- Understand type-specific message passing

---

## Colab 5 — Neighbor Sampling & Subgraph Sampling

**Out:** Week 9 | **Due:** Week 11 (end of course)
**Topic:** Scalable GNN training via neighbor and cluster sampling

**Direct links:**
- [2025 Colab 5](https://colab.research.google.com/drive/1S6LFPJxYHtBkWFgA4Yc5E173y59_rWpl)
- [2021 Colab 5](https://colab.research.google.com/drive/17Pe4o_oSsD2J-wTb_xGtYJQsyCawK6sJ?usp=sharing)
- [AndrewSpano solution](https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs/blob/main/CS224W_Colab_5.ipynb)

**Skills covered:**
- Implement neighbor sampling (GraphSAGE-style) with PyG's `NeighborLoader`
- Implement cluster/subgraph sampling with `ClusterData` and `ClusterLoader`
- Compare different sampling ratios on large graphs
- Understand training efficiency vs. accuracy trade-offs
- Scale GNN training to graphs with millions of nodes

---

## Summary Table

| Colab | Topic | Dataset | Key PyG/Library API |
|---|---|---|---|
| 0 | Graph libraries intro | KarateClub | NetworkX, PyG basics |
| 1 | Node embeddings | KarateClub | `torch.nn.Embedding` |
| 2 | GCN/SAGE | ogbn-arxiv, ogbg-molhiv | `GCNConv`, `SAGEConv`, OGB |
| 3 | SAGE/GAT + link pred | Cora | `GATConv`, `DeepSNAP` |
| 4 | Heterogeneous GNNs | ACM | `HeteroGraph`, `HeteroConv` |
| 5 | Scalable training | Large graphs | `NeighborLoader`, `ClusterData` |
