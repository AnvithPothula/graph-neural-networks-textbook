# References: Knowledge Graph Embeddings

1. [Knowledge Graph](https://en.wikipedia.org/wiki/Knowledge_graph) - Wikipedia - Covers the definition, history, and major examples of knowledge graphs including Freebase, Wikidata, and Google's Knowledge Graph. Provides essential background on the open-world and closed-world assumptions that motivate embedding-based link prediction.

2. [Link Prediction](https://en.wikipedia.org/wiki/Link_prediction) - Wikipedia - Explains the general link prediction problem in networks, covering both structural heuristics (common neighbors, Jaccard coefficient) and learning-based approaches. Situates KG link prediction as a special case where typed, directed relations replace unlabeled edges.

3. [Wikidata](https://en.wikipedia.org/wiki/Wikidata) - Wikipedia - Describes Wikidata, the largest freely available knowledge graph, with over 100 million statements. Illustrates the (head, relation, tail) triple format used throughout KG embedding research and provides a concrete example of the scale and incompleteness that motivates embedding-based completion.

4. Knowledge Graphs: Fundamentals, Techniques, and Applications - Mayank Kejriwal, Craig A. Knoblock, Pedro Szekely - MIT Press - A comprehensive graduate-level treatment of knowledge graph construction, representation, and reasoning. Chapters on embedding-based methods give theoretical grounding for TransE, bilinear families, and evaluation metrics including filtered MRR and Hits@K.

5. Graph Representation Learning - William L. Hamilton - Morgan & Claypool (Synthesis Lectures on AI and Machine Learning) - Dedicates a full chapter to knowledge graph embeddings, deriving TransE, DistMult, ComplEx, and RotatE under a unified bilinear scoring framework. Ideal companion reading for this chapter; freely available as a draft at cs.mcgill.ca/~wlh/grl_book/.

6. [Translating Embeddings for Modeling Multi-relational Data (TransE)](https://arxiv.org/abs/1301.3666) - arXiv - The original TransE paper by Bordes et al. (2013) that introduced the h + r ≈ t objective for link prediction. Despite its simplicity TransE still serves as the baseline against which all subsequent geometric KG embedding methods are compared.

7. [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/abs/1902.10197) - arXiv - Sun et al. (2019) introduce RotatE and the canonical four relation-pattern taxonomy (symmetry, antisymmetry, inversion, composition). Proves that RotatE is the minimal model that covers all four patterns and introduces self-adversarial negative sampling.

8. [Complex Embeddings for Simple Link Prediction (ComplEx)](https://arxiv.org/abs/1606.06357) - arXiv - Trouillon et al. (2016) show that extending DistMult to complex-valued embeddings is sufficient to handle asymmetric relations while retaining closed-form scoring. The Hermitian dot product derivation is one of the most elegant theoretical results in KG embedding.

9. [PyKEEN — Python KGE Library Documentation](https://pykeen.readthedocs.io/en/stable/) - PyKEEN / ReadTheDocs - Official documentation for PyKEEN, which implements 40+ KG embedding models with standardized training pipelines, filtered-ranking evaluation, and hyperparameter optimization. The model gallery pages include mathematical definitions, paper citations, and ready-to-run code for TransE, RotatE, ComplEx, and DistMult.

10. [Papers With Code: Knowledge Graph Completion Benchmark](https://paperswithcode.com/task/knowledge-graph-completion) - Papers With Code - Leaderboard tracking state-of-the-art results on FB15k-237 and WN18RR with reproducible code links. Provides up-to-date context on where the geometric embedding families covered in this chapter stand relative to newer transformer-based and GNN-based approaches.
