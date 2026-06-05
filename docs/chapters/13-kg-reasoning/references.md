# References: Reasoning over Knowledge Graphs

1. [Knowledge Graph](https://en.wikipedia.org/wiki/Knowledge_graph) - Wikipedia - Covers the structure, history, and major applications of knowledge graphs, including Freebase, Wikidata, and commercial deployments. Provides essential background on the triple-store data model underpinning all KG reasoning methods.

2. [Ontology (information science)](https://en.wikipedia.org/wiki/Ontology_(information_science)) - Wikipedia - Explains formal ontologies and description logics, which form the theoretical basis for logical query languages over knowledge graphs. Connects box-embedding geometric semantics to classical AI knowledge representation.

3. [Belief propagation](https://en.wikipedia.org/wiki/Belief_propagation) - Wikipedia - Describes message-passing inference on graphical models, the algorithmic ancestor of path-based KG reasoning methods such as NBFNet. Understanding sum-product message passing clarifies why Bellman-Ford-style graph propagation is a natural fit for multi-hop inference.

4. Knowledge Graphs: Fundamentals, Techniques, and Applications - Mayank Kejriwal, Craig A. Knoblock, Pedro Szekely - MIT Press - Comprehensive graduate-level treatment of knowledge graph construction, embedding, and reasoning. Chapter 9 covers multi-hop query answering and situates Query2Box and BetaE within the broader ontological reasoning literature.

5. Introduction to Statistical Relational Learning - Lise Getoor, Ben Taskar (eds.) - MIT Press - Foundational collection on probabilistic logic and relational learning, directly motivating differentiable rule-learning approaches (DRUM, Neural LP) covered in the path-based reasoning section. Essential context for understanding how neural KG methods relate to classical statistical relational AI.

6. [Query2Box: Reasoning over Knowledge Graphs in Vector Space Using Box Embeddings](https://arxiv.org/abs/2002.05969) - arXiv - Introduces axis-aligned box embeddings to represent answer sets of conjunctive queries, with a learnable intersection operator. The geometric AND-as-intersection intuition is the conceptual anchor for the entire chapter's treatment of logical query answering.

7. [Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs](https://arxiv.org/abs/2010.11465) - arXiv - Extends the query-embedding paradigm to handle negation by representing entity sets as Beta distributions. Demonstrates that complement distributions naturally encode NOT, enabling full first-order existential positive query answering.

8. [Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction](https://arxiv.org/abs/2106.06935) - arXiv - Proposes NBFNet, which initializes per-query node representations and propagates them via learnable Bellman-Ford iterations. Achieves state-of-the-art link prediction on FB15k-237 and WN18RR while remaining interpretable through path score extraction.

9. [KG-Reasoning Papers With Code](https://paperswithcode.com/task/knowledge-graph-completion) - Papers With Code - Tracks reproducible benchmarks for KG completion and multi-hop reasoning, including Query2Box, BetaE, CQD, and NBFNet results on FB15k-237, WN18RR, and the EPFO query benchmark. Useful for comparing methods on standardized leaderboards with linked code.

10. [Stanford CS224W: Machine Learning with Graphs — Knowledge Graph Reasoning (Lecture 10)](https://web.stanford.edu/class/cs224w/) - Stanford University - Lecture slides and notes covering multi-hop path queries, the GQE-to-Query2Box progression, and BetaE, with visual walkthroughs of box intersection and Beta distribution complement operations. Directly complements the derivations in this chapter.
