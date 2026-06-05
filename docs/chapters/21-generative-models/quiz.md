# Quiz: Deep Generative Models for Graphs

Test your understanding of GraphRNN, GCPN, VGAE, DiGress, and graph generation for drug discovery.

---

#### Question 1

GraphRNN generates graphs by treating graph generation as a sequential process. What is the core idea?

<div class="upper-alpha" markdown>
1. It generates all edges simultaneously using a single forward pass through a GNN encoder-decoder
2. It uses a VAE to sample a latent code and then decodes it into a complete adjacency matrix
3. It generates the adjacency matrix row by row: at each step, a graph-level RNN produces the state for a new node, then a sequence-level RNN generates the connection vector (which existing nodes the new node connects to)
4. It generates graphs by applying random perturbations to a template graph structure
</div>

??? question "Show Answer"
    The correct answer is **C**. GraphRNN models the autoregressive process of building a graph node by node. A graph-level RNN \(\mathbf{h}_t = f(\mathbf{h}_{t-1}, S_t)\) maintains the state after adding node t and its connections. A sequence-level RNN then generates the adjacency vector \(S_t\): for each previous node j < t, the sequence RNN predicts whether node t connects to node j. The graph-level state is updated after the full adjacency vector is generated. This converts graph generation into a tractable sequential prediction problem.

    **Concept Tested:** GraphRNN, Graph Generative Model

---

#### Question 2

GCPN (Graph Convolutional Policy Network) frames molecule generation as a reinforcement learning problem. What is the reward signal?

<div class="upper-alpha" markdown>
1. The number of atoms successfully added to the molecule without violating valence constraints
2. The negative log-likelihood of the generated molecule under a pre-trained language model
3. The similarity between the generated molecule and a reference drug compound
4. A combination of chemical validity rewards (is the molecule chemically valid?) and property rewards (does it have desired properties like drug-likeness, binding affinity, low toxicity?)
</div>

??? question "Show Answer"
    The correct answer is **D**. GCPN uses a GNN as the policy network that decides which atoms/bonds to add to the growing molecule. The reward signal combines: (1) a chemical validity reward (penalty for creating molecules that violate valence rules or are not syntactically valid SMILES); (2) property-based rewards from a property predictor (drug-likeness QED score, synthesizability SA score, binding affinity from docking simulations); (3) adversarial rewards from a discriminator that distinguishes generated from real molecules. PPO optimizes the policy against this compound reward.

    **Concept Tested:** GCPN, Drug Discovery with GNNs

---

#### Question 3

A Variational Graph Autoencoder (VGAE) encodes a graph into a latent distribution and decodes it back. What is the decoder's output and how does it reconstruct the graph?

<div class="upper-alpha" markdown>
1. The decoder directly outputs the learned node embeddings from the encoder
2. The decoder reconstructs the adjacency matrix by scoring each potential edge as \(\sigma(\mathbf{z}_u \cdot \mathbf{z}_v)\) — the dot product of latent node vectors passed through sigmoid
3. The decoder reconstructs node feature vectors by inverting the GNN encoder
4. The decoder generates node labels using a softmax classifier on the latent codes
</div>

??? question "Show Answer"
    The correct answer is **B**. VGAE's decoder scores edge probability between nodes \(u\) and \(v\) as \(\sigma(\mathbf{z}_u \cdot \mathbf{z}_v)\) where \(\mathbf{z}_u, \mathbf{z}_v \in \mathbb{R}^d\) are sampled latent codes. The reconstructed adjacency \(\hat{A} \approx \sigma(ZZ^\top)\) — the outer product of the full latent matrix. The encoder uses a GNN to compute the mean \(\mu_u\) and log-variance \(\log \sigma^2_u\) for each node, and the training loss is \(\text{ELBO} = \text{reconstruction loss} + \text{KL divergence}\). VGAE was the first method to show that GNN encoders + inner-product decoders form a powerful framework for link prediction and graph generation.

    **Concept Tested:** Variational Autoencoder (VGAE)

---

#### Question 4

DiGress generates molecular graphs using a discrete diffusion process. How does it differ from continuous diffusion models like DDPM?

<div class="upper-alpha" markdown>
1. DiGress adds and removes edges and node types using a categorical Markov chain over the discrete graph space rather than adding continuous Gaussian noise to real-valued features
2. DiGress uses score matching instead of variational inference for training
3. DiGress operates on the graph's Laplacian eigenspectrum rather than the adjacency matrix
4. DiGress generates graphs top-down from coarse structure to fine details using a hierarchical diffusion schedule
</div>

??? question "Show Answer"
    The correct answer is **A**. Molecular graphs are discrete: atoms are categorical (C, N, O, F...) and bonds are categorical (none, single, double, aromatic). Continuous diffusion models add Gaussian noise — appropriate for images but not for discrete graphs. DiGress defines a discrete forward process that progressively corrupts a molecular graph by randomly switching atom types and edge types toward a uniform (or marginal) distribution, and trains a GNN denoiser to reverse this corruption. The discrete Markov chain operates directly on the graph's categorical structure.

    **Concept Tested:** DiGress, Graph Generative Model

---

#### Question 5

Drug-drug interaction (DDI) prediction using GNNs models interactions between drugs in a network. What does the graph structure represent in a DDI network?

<div class="upper-alpha" markdown>
1. Nodes are protein targets and edges represent drugs that bind to the same target
2. Nodes are molecular atoms and edges are chemical bonds within each drug
3. Nodes are patient records and edges represent co-prescription of two drugs
4. Nodes are drugs and edges connect drugs that have known interactions (positive DDIs) — the GNN predicts which additional edges (unreported interactions) should exist
</div>

??? question "Show Answer"
    The correct answer is **D**. In a DDI network, each node represents a drug compound and each edge represents a known interaction — typically adverse polypharmacy effects (e.g., drug A increases the blood level of drug B, drug A and B combined cause cardiac arrhythmia). The GNN learns drug representations by aggregating over known interactions, then scores potential new DDIs using the encoder-decoder framework. Accurately predicting DDIs is critical for patient safety, as polypharmacy (taking multiple drugs simultaneously) is common in elderly patients.

    **Concept Tested:** Drug-Drug Interaction, Drug Discovery with GNNs

---

#### Question 6

Graph generation metrics evaluate whether generated graphs resemble real molecular graphs. Which metric measures whether generated molecules could realistically be synthesized in a laboratory?

<div class="upper-alpha" markdown>
1. Drug-likeness (QED) score, which measures similarity to known drugs via Lipinski's rule of five
2. Uniqueness, which measures whether the generated molecule appears in the training set
3. Synthetic Accessibility (SA) score, which estimates how feasible laboratory synthesis is on a 1–10 scale based on the complexity of the molecular structure
4. Validity score, which measures whether the SMILES string representing the molecule is parseable
</div>

??? question "Show Answer"
    The correct answer is **C**. The SA (Synthetic Accessibility) score (Ertl & Schuffenhauer 2009) estimates synthetic feasibility on a scale of 1 (easy to synthesize) to 10 (very hard) using structural fragments from PubChem. Validity tests whether a molecule is chemically legal (correct valence, no radicals); QED measures drug-likeness; novelty measures how many generated molecules are not in the training set; uniqueness measures the diversity of the generated set. All these metrics together assess whether generated molecules are (1) valid, (2) novel, (3) diverse, and (4) practically synthesizable.

    **Concept Tested:** Graph Generation Metrics, Molecule Generation

---

#### Question 7

GCPN's policy network uses a GCN to process the current state of the molecular graph being built. What action space does the RL agent operate over?

<div class="upper-alpha" markdown>
1. The agent selects which element to remove from the molecule at each step
2. The agent selects which atom type to add and which existing atom to connect it to (or selects "done" to terminate generation)
3. The agent selects a 3D conformation for the current molecular structure
4. The agent selects which training batch to use for the next gradient update
</div>

??? question "Show Answer"
    The correct answer is **B**. GCPN's action at each step is a pair (s, t, b) where s is the scaffold node to attach to, t is the type of the new atom (C, N, O, ...) or functional group, and b is the bond type (single, double, aromatic). The agent can also select a "terminate" action to stop adding atoms. Each action grows the molecular graph by one atom and one or more bonds. The GCN state captures the current partial molecule's graph structure, allowing the policy to make chemically informed decisions about where and what to add next.

    **Concept Tested:** GCPN, Molecule Generation

---

#### Question 8

Protein-protein interaction (PPI) networks are used in GNN drug discovery pipelines. What do nodes and edges represent in a PPI network used for drug target identification?

<div class="upper-alpha" markdown>
1. Nodes are amino acid residues and edges are hydrogen bonds between residues
2. Nodes are proteins (gene products) and edges represent experimentally verified physical binding between the protein pairs
3. Nodes are drugs and edges represent drugs that affect the same protein pathway
4. Nodes are diseases and edges connect diseases sharing a common protein marker
</div>

??? question "Show Answer"
    The correct answer is **B**. In a PPI network, each node is a protein (identified by gene name, UniProt ID, etc.) and each edge represents an experimentally observed or computationally predicted physical interaction between the two proteins. GNNs on PPI networks can identify "essential proteins" (high-degree or high-betweenness nodes whose dysfunction causes disease), predict protein function by propagating known functional labels, and identify drug targets by finding proteins in disease-associated modules. STRING and BioGRID are standard PPI databases used in these analyses.

    **Concept Tested:** Protein-Protein Interaction, Drug Discovery with GNNs

---

#### Question 9

Why is the molecular graph representation superior to SMILES strings for machine learning on molecules?

<div class="upper-alpha" markdown>
1. Molecular graphs have smaller memory footprint than SMILES strings for large molecules
2. SMILES strings cannot encode ring structures; molecular graphs handle all topologies natively
3. SMILES cannot be parsed by standard tokenizers, making language model training impossible
4. SMILES is a sequential text representation that requires linearizing a 2D structure, losing spatial relationships and making it hard to encode symmetry; molecular graphs directly encode atoms as nodes and bonds as edges, enabling GNNs to use the 2D topology natively with permutation equivariance
</div>

??? question "Show Answer"
    The correct answer is **D**. SMILES encodes a molecule as a 1D string (e.g., "c1ccccc1" for benzene), requiring a canonical ordering of atoms — an arbitrary choice that introduces tokenization artifacts. Different valid SMILES for the same molecule look completely different as strings, requiring data augmentation. Molecular graphs represent atoms as nodes with features (element, charge, hybridization) and bonds as edges with features (bond type, aromaticity), capturing the 2D topology directly. GNNs on molecular graphs are natively permutation equivariant — no canonical ordering needed.

    **Concept Tested:** Molecular Graph, Drug Discovery with GNNs

---

#### Question 10

GraphRNN's training objective is the negative log-likelihood of the training graph sequences. What is one major limitation of GraphRNN compared to DiGress for molecular generation?

<div class="upper-alpha" markdown>
1. GraphRNN generates graphs by ordering nodes, making its output dependent on the arbitrary BFS/DFS node ordering used during training — valid molecules can be generated in exponentially many different orderings, creating inconsistency
2. GraphRNN has higher computational complexity than DiGress for all graph sizes
3. GraphRNN cannot model atom types; DiGress natively encodes categorical atom features
4. GraphRNN requires labeled training data while DiGress is fully unsupervised
</div>

??? question "Show Answer"
    The correct answer is **A**. GraphRNN conditions generation on a specific node ordering (e.g., BFS order starting from a random node). The same molecular graph can be traversed in many valid orderings, each producing a different training sequence. This ordering ambiguity means the model must learn multiple ways to generate the same molecule — wasting capacity. DiGress avoids ordering by operating on the full graph jointly: the diffusion process corrupts and denoises the entire graph simultaneously using a permutation equivariant GNN, eliminating the node ordering problem entirely.

    **Concept Tested:** GraphRNN, DiGress, Graph Generative Model
