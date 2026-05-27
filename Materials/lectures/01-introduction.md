# Lecture 1: Introduction to Graph Machine Learning

## Overview

This lecture introduces the fundamental question: *why graphs?* It establishes that many real-world datasets are naturally represented as graphs—social networks, the web, molecular structures, knowledge bases—and that leveraging this relational structure can dramatically improve ML predictions.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/01-intro.pdf
**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/01-intro.pdf
**YouTube (2021):** Search "CS224W Lecture 1" in playlist https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn

---

## Key Concepts

1. **Why graphs?** — Relational structure is ubiquitous. Graphs encode entities (nodes) and their relationships (edges).
2. **Types of graphs:** undirected, directed, weighted, heterogeneous, attribute graphs, bipartite
3. **Graph tasks:** node-level, edge-level, graph-level (classification, regression, generation)
4. **Real-world graph examples:**
   - Social networks (Facebook, Twitter): user nodes, friendship edges
   - Citation networks (Cora, ArXiv): paper nodes, citation edges
   - Molecular graphs: atom nodes, bond edges
   - Knowledge graphs: entity nodes, relation edges
   - Web graph: page nodes, hyperlink edges
   - Protein-protein interaction (PPI) networks
5. **Why is graph ML hard?** — No fixed structure, arbitrary size, no spatial locality, nodes have different number of neighbors
6. **Representation learning on graphs:** the goal is to map nodes/edges/graphs to low-dimensional embeddings that preserve structure and enable downstream ML tasks
7. **Types of ML tasks on graphs:**
   - Node classification (predict label of node given its neighborhood)
   - Link prediction (predict if edge exists between two nodes)
   - Community detection (find groups of closely related nodes)
   - Network similarity (how similar are two graphs?)
   - Graph classification (classify entire graph)

## Graph Properties to Know

- **Degree distribution** P(k): fraction of nodes with degree k
- **Clustering coefficient** C: fraction of connected triangles among node's neighbors
- **Connected components:** the isolated parts of a graph
- **Path length:** average shortest path between nodes
- **Giant component:** the largest connected component
- **Power-law degree distribution:** characteristic of real-world networks (scale-free)
- **Small-world property:** low average path length despite high clustering

## Course Roadmap

1. Traditional ML on graphs (hand-crafted features)
2. Graph representation learning (node embeddings)
3. Deep learning on graphs (GNNs)
4. Knowledge graphs and reasoning
5. Applications (recommender systems, drug discovery, etc.)
6. Scalability, generative models, advanced topics

## Textbook Chapter Notes

This lecture maps to **Chapter 1: Introduction to Graph ML**. Key elements to cover:
- Motivating examples with real datasets
- Mathematical definition of a graph
- Graph properties (with code to compute them using NetworkX)
- Overview of graph ML tasks
- MicroSim idea: interactive graph viewer where user adds/removes nodes and sees properties update live

## Suggested Code Example (NetworkX)

```python
import networkx as nx

# Create a simple social network
G = nx.karate_club_graph()

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print(f"Clustering coefficient: {nx.average_clustering(G):.3f}")
print(f"Average shortest path: {nx.average_shortest_path_length(G):.3f}")
```
