# Reference Books for GNN Textbook

Three foundational textbooks freely available online. These are the recommended readings listed on the CS224W course page and provide deeper mathematical treatment of the topics covered in this textbook.

---

## 1. Graph Representation Learning

**Author:** William L. Hamilton
**Publisher:** Morgan & Claypool (Synthesis Lectures on AI and Machine Learning), 2020
**Free online:** https://www.cs.mcgill.ca/~wlh/grl_book/

### Coverage
The most directly relevant textbook to this project. Written by one of the creators of GraphSAGE, it covers:

- **Chapter 1–2:** Graph statistics, graph theory background, machine learning background
- **Chapter 3:** Neighborhood reconstruction methods (DeepWalk, node2vec, matrix factorization)
- **Chapter 4:** Multi-relational data and KGs; TransE, ComplEx, RotatE
- **Chapter 5:** Generative models for graphs (GraphRNN, VGAE, GCPN)
- **Chapter 6:** Deep learning for graphs; GCN derivation; spectral vs. spatial
- **Chapter 7:** Graph neural networks and applications
- **Appendix:** Background on linear algebra, probability, ML

### When to Reference
- For mathematical derivations of GCN and node2vec
- For the formal encoder-decoder framework
- For KG embedding proofs
- For generative model formalisms

---

## 2. Networks, Crowds, and Markets

**Authors:** David Easley and Jon Kleinberg
**Publisher:** Cambridge University Press, 2010
**Free online:** http://www.cs.cornell.edu/home/kleinber/networks-book/

### Coverage
A broader treatment covering the social science and economics perspective on networks:

- **Part I:** Graph theory and social networks (strong/weak ties, triadic closure)
- **Part II:** Game theory and strategic behavior on networks
- **Part III:** Markets and auctions
- **Part IV:** Information networks (web graph, PageRank, hyperlink analysis)
- **Part V:** Network dynamics (epidemics, diffusion, cascading behavior)
- **Part VI:** Population heterogeneity and evolution

### When to Reference
- For PageRank and web graph analysis (Chapter 14–15)
- For network epidemics and influence maximization (Chapter 21–22)
- For community structure and clustering (Chapter 3)
- For historical context on network analysis

### Key Chapters for This Textbook
- Ch. 2–3: Graph connectivity, strong/weak ties
- Ch. 13–14: Web search and PageRank
- Ch. 19–21: Cascading behavior, diffusion of innovations

---

## 3. Network Science

**Author:** Albert-László Barabási
**Free online (with interactive content):** http://networksciencebook.com
**Also available as PDF chapters:** http://networksciencebook.com

### Coverage
Beautiful, visually rich treatment of complex networks from a physics/complexity science perspective:

- **Chapter 1:** Introduction to networks
- **Chapter 2:** Graph theory basics
- **Chapter 3:** Random networks (Erdős–Rényi model)
- **Chapter 4:** Scale-free property (power-law degree distributions)
- **Chapter 5:** Barabási–Albert model (preferential attachment)
- **Chapter 6:** Evolving networks
- **Chapter 7:** Degree correlations (assortative/disassortative)
- **Chapter 8:** Network robustness and fragility
- **Chapter 9:** Communities (Girvan-Newman, modularity)
- **Chapter 10:** Spreading phenomena (epidemics, SIS/SIR models)

### When to Reference
- For real-world network properties (Chapter 4: scale-free networks)
- For preferential attachment and network growth models
- For community detection background (Chapter 9)
- For epidemic spreading / influence maximization (Chapter 10)

### Key Insight for Textbook
This book explains *why* graph ML is important: real-world networks have non-trivial structure (power-law degrees, small-world properties, community structure) that classical ML ignores.

---

## 4. Additional Online Resources

### CS224W Course Notes PDF
- **URL:** https://archives.leni.sh/stanford/CS224w.pdf
- Community-compiled notes from CS224W lectures
- Covers all major lecture topics in condensed form
- Good for quick reference on notation and formulas

### PyTorch Geometric Documentation
- **URL:** https://pytorch-geometric.readthedocs.io/
- Official API docs for all GNN layers, datasets, and transforms
- Includes tutorials for node classification, link prediction, and graph classification

### Open Graph Benchmark (OGB)
- **URL:** https://ogb.stanford.edu/
- Standardized benchmark datasets and evaluation
- Leaderboards for all major GNN tasks
- Created by the CS224W team

### Stanford SNAP (Stanford Network Analysis Platform)
- **URL:** https://snap.stanford.edu/
- Jure Leskovec's research group website
- Contains datasets, tools, and research papers
- Source of many CS224W datasets

### Deep Graph Library (DGL)
- **URL:** https://www.dgl.ai/
- Alternative to PyG; good documentation and tutorials
- Some models are easier to implement in DGL vs. PyG

---

## Recommended Reading Order

For building a GNN textbook, reference these in this order:

1. **Barabási (Network Science):** Chapters 1–5 — understand why networks matter
2. **Easley & Kleinberg:** Chapters 13–15 — PageRank and web analysis
3. **Hamilton (GRL):** Chapters 1–7 — core technical content for most chapters
4. **CS224W Notes PDF** — for compact notation and formula reference

