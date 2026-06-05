# References: Advanced GNN Topics: In-Context Learning and Uncertainty

1. [Self-supervised learning](https://en.wikipedia.org/wiki/Self-supervised_learning) - Wikipedia - Covers the general framework of self-supervised learning in which models generate supervisory signals from the data's own structure; foundational context for graph SSL methods like DGI and GraphCL.

2. [Contrastive learning](https://en.wikipedia.org/wiki/Contrastive_learning) - Wikipedia - Explains the contrastive learning paradigm, including positive/negative pair construction and the InfoNCE loss; directly relevant to GraphCL and GRACE covered in Section 24.4.

3. [Conformal prediction](https://en.wikipedia.org/wiki/Conformal_prediction) - Wikipedia - Describes the distribution-free uncertainty quantification framework that produces prediction sets with guaranteed coverage; provides the statistical foundation for conformalized GNNs in Section 24.6.

4. Deep Learning on Graphs - Ma and Tang - Cambridge University Press - Comprehensive treatment of graph representation learning including self-supervised methods; Chapter 8 covers graph contrastive learning and mutual information maximization with worked derivations.

5. Probabilistic Machine Learning: Advanced Topics - Murphy - MIT Press - Rigorous coverage of uncertainty quantification methods including conformal prediction and Bayesian deep learning; the conformal inference sections (Chapter 15) complement the GNN-specific treatment in Section 24.6.

6. [Deep Graph Infomax (DGI) — arXiv:1809.10341](https://arxiv.org/abs/1809.10341) - arXiv - The original DGI paper by Veličković et al. (ICLR 2019); introduces mutual-information maximization between patch representations and a global summary vector as a self-supervised graph pretext task.

7. [GraphCL: Graph Contrastive Representation Learning — arXiv:2010.13902](https://arxiv.org/abs/2010.13902) - arXiv - You et al. (NeurIPS 2020) systematically study four graph augmentation strategies (node dropping, edge perturbation, attribute masking, subgraph sampling) and show which augmentations transfer to which graph types.

8. [PRODIGY: Enabling In-Context Learning Over Graphs — arXiv:2305.12600](https://arxiv.org/abs/2305.12600) - arXiv - Huang et al. (NeurIPS 2023) introduce prompt graphs that encode support examples directly into the graph structure, enabling few-shot in-context learning on node and edge classification tasks without fine-tuning.

9. [Uncertainty Quantification over Graph with Conformalized GNNs — arXiv:2305.14535](https://arxiv.org/abs/2305.14535) - arXiv - Huang et al. (NeurIPS 2023) propose DAPS, a diffusion-smoothed nonconformity score that exploits graph homophily to produce tighter prediction sets than naive conformal prediction on ogbn-arxiv and ogbn-products.

10. [Graph Self-Supervised Learning: A Survey — arXiv:2103.00111](https://arxiv.org/abs/2103.00111) - arXiv - Liu et al. (2022) survey over 30 graph SSL methods organized by pretext task type (generative, predictive, contrastive, hybrid); a useful map for understanding where DGI, GraphCL, and GRACE sit in the broader landscape.
