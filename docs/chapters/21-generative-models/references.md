# References: Deep Generative Models for Graphs

1. [Variational Autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) - Wikipedia - Explains the VAE framework (encoder-decoder architecture, ELBO objective, reparameterization trick) that underlies VGAE and other latent-space graph generative models.

2. [Generative Adversarial Network](https://en.wikipedia.org/wiki/Generative_adversarial_network) - Wikipedia - Covers the adversarial training paradigm used as a baseline and component in several graph generation approaches, including MolGAN and hybrid GAN-VAE molecular models.

3. [Molecular graph](https://en.wikipedia.org/wiki/Molecular_graph) - Wikipedia - Defines the graph-theoretic representation of molecules (atoms as nodes, bonds as edges) that is the canonical target domain for evaluating graph generative models in drug discovery benchmarks.

4. Deep Learning - Ian Goodfellow, Yoshua Bengio, Aaron Courville - MIT Press - Chapters 14–20 cover directed and undirected generative models, VAEs, and GANs; provides the probabilistic foundations needed before specializing to graph-structured outputs.

5. Graph Representation Learning - William L. Hamilton - Morgan & Claypool (Synthesis Lectures on AI and ML) - Chapter 7 surveys graph generative models from VAE-based to sequential approaches; accessible treatment with consistent notation aligned to the GNN literature.

6. [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models (You et al., 2018)](https://arxiv.org/abs/1802.08773) - arXiv - The original GraphRNN paper; introduces the BFS-ordered sequential formulation and demonstrates that edge-level RNNs outperform prior Kronecker and SBM baselines on structural graph metrics.

7. [Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation (You et al., 2018)](https://arxiv.org/abs/1806.02473) - arXiv - Introduces GCPN, which frames molecule construction as an MDP on a partial graph and applies PPO with chemistry-aware reward shaping; first model to jointly optimize validity and targeted property scores.

8. [Variational Graph Auto-Encoders (Kipf & Welling, 2016)](https://arxiv.org/abs/1611.07308) - arXiv - Foundational VGAE paper; derives the graph ELBO using a GCN encoder and inner-product decoder, establishing the standard approach for learning continuous latent representations of graph structure.

9. [DiGress: Discrete Denoising Diffusion for Graphs (Vignac et al., 2022)](https://arxiv.org/abs/2209.14734) - arXiv - Introduces discrete diffusion over node and edge categorical attributes; achieves state-of-the-art FCD on MOSES and GuacaMol by operating directly in graph space rather than converting to sequences or continuous embeddings.

10. [Benchmarking Graph Neural Networks for Molecular Generation — Papers With Code: Molecule Generation](https://paperswithcode.com/task/molecule-generation) - Papers With Code - Aggregates leaderboard results, datasets (ZINC, MOSES, GuacaMol), and linked implementations for the major molecular graph generation models, making it easy to compare validity, uniqueness, novelty, and FCD scores across methods.
