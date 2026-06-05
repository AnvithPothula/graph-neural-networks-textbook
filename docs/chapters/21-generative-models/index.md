---
title: "Chapter 21: Deep Generative Models for Graphs"
description: "Deep generative models for graphs — GraphRNN, GCPN, VGAE, and DiGress — applied to molecular generation and drug discovery."
generated_by: claude skill chapter-content-generator
date: 2026-05-30 11:04:35
version: 0.08
---

# Chapter 21: Deep Generative Models for Graphs

**Part 5: Scalability and Generation**

## Summary

Covers deep generative models for graphs — GraphRNN, GCPN, VGAE, and DiGress — with particular focus on molecular graph generation for drug discovery applications.

## Concepts Covered

This chapter covers the following 11 concepts from the learning graph:

1. Drug-Drug Interaction
2. Protein-Protein Interaction
3. Molecular Graph
4. Drug Discovery with GNNs
5. Graph Generative Model
6. GraphRNN
7. GCPN
8. Molecule Generation
9. Variational Autoencoder (VGAE)
10. DiGress
11. Graph Generation Metrics

## Prerequisites

This chapter builds on:

- [Chapter 6: GNN Foundations: Message Passing and GCN](../06-gnn-foundations/index.md)
- [Chapter 7: GNN Design Space: GraphSAGE and GAT](../07-gnn-design-space/index.md)
- [Chapter 20: Scaling GNNs to Billion-Node Graphs](../20-scaling-gnns/index.md)

---

!!! mascot-welcome "Generating Something New"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waves hello">
    Every chapter so far has been about *analyzing* graphs that already exist — classifying their nodes, predicting their links, detecting their communities. This chapter asks a fundamentally different question: can we *generate* new graphs that have never been seen before, yet are valid, useful, and structurally coherent? The most compelling application is molecule design — a new molecule is a graph, and generating better molecules is one of the most consequential applications of GNNs in existence. We will build up four distinct generative paradigms: autoregressive sequence models, reinforcement learning, variational autoencoders, and diffusion models. Each makes a different bet about what structure to impose on the generation process, and understanding those bets is the core of this chapter.

## 21.1 The Problem: Designing Molecules as Graphs

Drug discovery is one of the most expensive scientific endeavors in history. Bringing a single new drug from initial compound to regulatory approval takes a median of 12 years and costs approximately $2.6 billion — and the failure rate at clinical trials exceeds 90%. The central bottleneck is finding the right molecule: a compound with high biological activity against a target protein, low toxicity, good pharmacokinetics, and synthetic accessibility. The chemical space of drug-like molecules is estimated at \( 10^{60} \) compounds, far beyond the capacity of any experimental screening program.

A molecule is naturally represented as a **molecular graph**: atoms correspond to nodes annotated with element type (carbon, nitrogen, oxygen, …), and bonds correspond to edges annotated with bond type (single, double, triple, aromatic). This graph structure captures the covalent bonding topology that determines a molecule's three-dimensional shape and chemical behavior. The same representation extends naturally to two important network types that GNNs are applied to in the broader drug discovery pipeline.

A **drug-drug interaction (DDI)** network models how different drugs influence each other's pharmacological effects when co-administered. Nodes represent drugs; edges represent known interactions (synergistic, antagonistic, or side-effect-inducing). Predicting new DDI edges from known network topology enables safety screening before clinical trials — a link prediction task on a graph where the nodes themselves are molecules.

A **protein-protein interaction (PPI)** network maps the physical and functional contacts between proteins in a biological cell. Nodes represent proteins; edges represent experimentally confirmed interactions. GNN-based node classification on PPI networks predicts which proteins are involved in specific disease pathways, guiding the selection of drug targets. The PPI graph for a single human cell type contains roughly 20,000 nodes and several hundred thousand edges.

The unifying goal across all three representations — molecular graphs, DDI networks, and PPI networks — is to learn distributions over graph space so that new, biologically valid structures can be sampled, evaluated, and refined. This is the **graph generation problem**.

## 21.2 Formalizing Graph Generation

Before examining specific architectures, it is worth being precise about what it means to "generate" a graph. Let \( \mathcal{G} = \{G_1, G_2, \ldots, G_n\} \) be a set of observed graphs — for example, all known drug molecules in a database. The goal is to learn a probability distribution \( p_\theta(G) \) over graphs such that:

\[
\text{(1) } G \sim p_\theta(G) \text{ resembles graphs in } \mathcal{G}; \quad
\text{(2) } G \sim p_\theta(G) \text{ is novel (not in } \mathcal{G}\text{)}
\]

Additionally, for **conditional generation**, we want \( p_\theta(G \mid y) \) where \( y \) encodes a target property (e.g., "bioactive against EGFR kinase with QED \( > 0.8 \)"). Conditioning converts unconditional sampling into goal-directed design.

Graph generation is substantially harder than image or text generation for three structural reasons, which we define here before examining how each model addresses them:

- **Discrete structure.** Node and edge types are categorical, not continuous. Gradient-based optimization cannot move smoothly through discrete choices; models must define stochastic categorical decisions or embed discrete choices in continuous latent spaces.
- **Variable size.** A text sentence has a fixed length or a sequence terminator; a graph has no natural length. Generative models must simultaneously decide the number of nodes and which pairs to connect.
- **Permutation invariance.** The same molecular graph can be represented by \( n! \) different orderings of its atoms. A generative model must produce the same distribution \( p_\theta(G) \) regardless of which ordering is used — or it must commit to a canonical ordering and accept that the model's implicit prior is over orderings, not graphs.

!!! mascot-thinking "Permutation Invariance is Genuinely Hard"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinks carefully">
    The permutation invariance problem is worth sitting with. When you train a model to generate the adjacency matrix of aspirin, the "correct" output is not a single matrix — it is 21! ≈ 10^19 different matrices, all equally valid representations of the same molecule. A naive MLE approach that treats one particular atom ordering as the ground truth is training on one out of 10^19 equivalent representations and penalizing all the others. The practical responses — BFS ordering (GraphRNN), SMILES linearization, or permutation-invariant diffusion (DiGress) — are each engineering different compromises with this fundamental symmetry.

## 21.3 GraphRNN: Sequential Graph Generation

**GraphRNN** (You et al., 2018) addresses the permutation problem by committing to a canonical ordering: nodes are added in breadth-first search (BFS) order from a random starting node. Under BFS ordering, when node \( v_t \) is added at step \( t \), it can only connect to nodes \( v_1, \ldots, v_{t-1} \) that were added before it. This converts the graph generation problem into a sequence generation problem: at each step, generate a new node and a binary sequence indicating which previous nodes to connect to.

GraphRNN implements this with two recurrent neural networks operating at different granularities. The **graph-level RNN** maintains a hidden state \( h_t \) that summarizes the graph generated so far, and at each step produces a context vector used to initialize the edge-level RNN. The **edge-level RNN** then generates a binary sequence \( (e_{t,t-1}, e_{t,t-2}, \ldots, e_{t,1}) \) — one bit per previous node — indicating which edges to add from the new node \( v_t \). Together:

\[
h_t = \text{GRU}^{\text{graph}}(h_{t-1}, \text{SOS}), \qquad
(e_{t,j})_{j=1}^{t-1} = \text{GRU}^{\text{edge}}\!\left(h_t\right)
\]

where SOS is a start-of-sequence token and \( \text{GRU}^{\text{edge}} \) outputs one Bernoulli logit per step. The BFS ordering is critical for tractability: under BFS, each new node connects only to nodes in its BFS subtree, so the edge sequence has length at most equal to the BFS bandwidth rather than \( t-1 \) for large graphs.

Training uses **teacher forcing** (standard in sequence models): at each step, the true graph state — not the model's sample — is fed as input to the GRU. This stabilizes training but introduces a train-inference mismatch: at inference, the model samples its own outputs autoregressively and may drift from the training distribution. Generation terminates when the graph-level RNN emits an end-of-sequence token.

The following code illustrates the core of a simplified GraphRNN implementation. Note the key architecture choices before reading: `hidden_dim` is the GRU state size shared between both RNNs; `max_prev_node` is the BFS bandwidth (maximum lookaback window); and `node_embedding_dim` is the size of the node context vector passed from the graph-level to the edge-level GRU.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLevelGRU(nn.Module):
    """Graph-level RNN: one step per node; outputs context for edge generation."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h):
        # x: (batch, 1, input_dim) — edge sequence embedding from previous step
        # h: (1, batch, hidden_dim) — graph-level hidden state
        out, h_new = self.gru(x, h)
        context = self.output_linear(out.squeeze(1))  # (batch, output_dim)
        return context, h_new


class EdgeLevelGRU(nn.Module):
    """Edge-level RNN: generates binary edge decisions for one new node."""
    def __init__(self, context_dim, hidden_dim, max_prev_node):
        super().__init__()
        self.gru_cell = nn.GRUCell(context_dim + 1, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, 1)
        self.max_prev_node = max_prev_node

    def forward(self, context, target_edges=None):
        # context: (batch, context_dim) — from graph-level GRU
        # target_edges: (batch, max_prev_node) — teacher-forcing targets (None at inference)
        batch = context.size(0)
        h = torch.zeros(batch, self.gru_cell.hidden_size, device=context.device)
        edge_logits = []
        # Start token: 0 (no previous edge)
        token = torch.zeros(batch, 1, device=context.device)
        for step in range(self.max_prev_node):
            inp = torch.cat([context, token], dim=-1)
            h = self.gru_cell(inp, h)
            logit = self.output_linear(h)  # (batch, 1)
            edge_logits.append(logit)
            if target_edges is not None:
                # Teacher forcing: use ground truth edge as next input
                token = target_edges[:, step:step+1].float()
            else:
                # Autoregressive sampling
                token = torch.bernoulli(torch.sigmoid(logit))
        return torch.cat(edge_logits, dim=-1)  # (batch, max_prev_node)


def graphrnn_loss(edge_logits, edge_targets):
    """Binary cross-entropy per edge position, averaged over valid positions."""
    # edge_logits, edge_targets: (batch, max_prev_node)
    return F.binary_cross_entropy_with_logits(edge_logits, edge_targets.float())
```

GraphRNN's main limitation is its **lack of global structural awareness**. Because the graph-level GRU is a fixed-size vector, it struggles to track long-range topological constraints — for example, ensuring that a molecule's ring system closes correctly 10 nodes after the ring was started. For small molecules (≤ 40 atoms), GraphRNN achieves validity rates above 95%; for larger drug-like molecules, validity drops substantially.

## 21.4 GCPN: Reinforcement Learning for Goal-Directed Generation

GraphRNN optimizes the likelihood of training graphs but does not directly optimize for molecular properties. **Graph Convolutional Policy Network (GCPN, You et al., 2018)** reframes molecule generation as a **reinforcement learning** problem where the agent is rewarded for generating molecules with desired biochemical properties.

!!! mascot-tip "RL Lets You Optimize Properties You Can't Backpropagate Through"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage gives a tip">
    Many molecular property predictors — docking simulators, QED calculators, retrosynthetic feasibility checkers — are not differentiable. You cannot simply backpropagate through them. Reinforcement learning sidesteps this: the model generates a complete molecule, evaluates it externally, and receives a scalar reward. The policy gradient (REINFORCE or PPO) then updates the GNN policy to generate higher-reward molecules. This is why GCPN is so powerful for goal-directed design: it can optimize any computable property, not just properties with accessible gradients.

The GCPN formulation defines four components:

- **State \( s_t \)**: the partial molecular graph built so far, represented as a GCN embedding of all atoms and bonds present at step \( t \).
- **Action \( a_t \)**: either (i) add a new atom of type \( z \in \{\mathrm{C}, \mathrm{N}, \mathrm{O}, \mathrm{F}, \ldots\} \) connected to atom \( u \) in the current graph, or (ii) add a bond between two existing atoms \( (u, v) \), or (iii) terminate generation.
- **Policy \( \pi_\theta(a_t \mid s_t) \)**: a GCN-based policy network that reads the current molecular graph and outputs a distribution over actions. The GCN embeds all atoms; a softmax over pairs of node embeddings selects which atoms to connect.
- **Reward \( r(G) \)**: a multi-component reward evaluating the final molecule on: (a) drug-likeness (QED score, 0–1), (b) synthetic accessibility (SA score, penalizes hard-to-synthesize structures), (c) a target property like predicted binding affinity, and (d) validity penalties for violating valence rules.

The reward function combines these components. For a generated molecule \( G \):

\[
r(G) = \lambda_1 \cdot \text{QED}(G) - \lambda_2 \cdot \text{SA}(G) + \lambda_3 \cdot f_{\text{target}}(G) + \lambda_4 \cdot \mathbf{1}[\text{valid}(G)]
\]

where \( \lambda_1, \ldots, \lambda_4 \) are tunable weights balancing the trade-offs between drug-likeness, synthesizability, target activity, and chemical validity. GCPN also applies a **validity mask** that forces the action distribution to zero for chemically impossible actions (e.g., adding a fifth bond to a carbon), avoiding wasted exploration on invalid chemistry.

The following code sketches the GCPN policy network and the core RL training loop. The `MolecularGCN` component embeds the current graph state; `PolicyHead` computes action logits over atom-pair choices.

```python
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class MolecularGCN(nn.Module):
    """GCN encoder for the current partial molecule graph."""
    def __init__(self, atom_feature_dim, hidden_dim, embed_dim, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(atom_feature_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, embed_dim))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        return self.convs[-1](x, edge_index)  # (num_atoms, embed_dim)


class PolicyHead(nn.Module):
    """Computes action logits: score every (source_atom, target_atom) pair."""
    def __init__(self, embed_dim, num_atom_types):
        super().__init__()
        # Score for connecting existing atoms or adding a new atom
        self.bond_score = nn.Bilinear(embed_dim, embed_dim, 1)
        self.atom_type_head = nn.Linear(embed_dim, num_atom_types)

    def forward(self, node_embeddings, graph_embedding):
        # Bond action: score all (i, j) pairs in current graph
        n = node_embeddings.size(0)
        src = node_embeddings.unsqueeze(1).expand(-1, n, -1)
        tgt = node_embeddings.unsqueeze(0).expand(n, -1, -1)
        bond_logits = self.bond_score(
            src.reshape(-1, src.size(-1)),
            tgt.reshape(-1, tgt.size(-1))
        ).reshape(n, n)
        # Atom type action: which element to add next
        atom_logits = self.atom_type_head(graph_embedding)
        return bond_logits, atom_logits


def reinforce_loss(log_probs, rewards, baseline=0.0):
    """Policy gradient loss with optional baseline for variance reduction."""
    # log_probs: list of log P(action) per step
    # rewards: scalar reward for the completed episode
    advantages = rewards - baseline
    return -torch.stack(log_probs).sum() * advantages
```

In practice, GCPN trains with **proximal policy optimization (PPO)** rather than vanilla REINFORCE to improve stability, and initializes from a policy pre-trained with imitation learning on known molecules to avoid starting from random chemistry. GCPN has been applied to generate inhibitors for the EGFR kinase target and to optimize solubility and drug-likeness (QED) simultaneously, achieving property distributions substantially better than random sampling from chemical space.

## 21.5 VGAE: Latent Space Representations for Graphs

Both GraphRNN and GCPN are **autoregressive**: they construct graphs one element at a time in sequence. A complementary approach is to learn a **latent space** over entire graphs: encode a graph into a continuous vector, perform optimization or interpolation in that vector space, then decode a new graph from any point in the space. This is the **Variational Graph Autoencoder (VGAE)** paradigm (Kipf & Welling, 2016).

VGAE follows the standard variational autoencoder (VAE) framework adapted for graphs. The encoder is a GCN that produces a distribution over latent codes rather than a point estimate. For each node \( v \), it outputs a mean vector \( \mu_v \in \mathbb{R}^d \) and a log-variance vector \( \log \sigma^2_v \in \mathbb{R}^d \). Latent codes are sampled using the reparameterization trick:

\[
z_v = \mu_v + \epsilon \odot \sigma_v, \quad \epsilon \sim \mathcal{N}(0, I_d)
\]

The decoder is a simple inner product: the probability of edge \( (u, v) \) is the sigmoid of the dot product between their latent codes:

\[
\hat{A}_{uv} = \sigma(z_u^\top z_v)
\]

The model is trained to maximize the Evidence Lower Bound (ELBO):

\[
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(Z|A,X)}\!\left[\log p(A \mid Z)\right] - \text{KL}\!\left(q(Z \mid A, X) \;\|\; p(Z)\right)
\]

where the first term rewards accurate edge reconstruction and the second term keeps the posterior \( q(Z \mid A, X) \) close to the standard normal prior \( p(Z) = \mathcal{N}(0, I) \).

The following code implements VGAE using PyTorch Geometric's built-in VGAE module. The key parameters are: `GCNEncoder` produces node-level mean and log-variance; `out_channels` is the latent dimension \( d \); and `recon_loss` computes the reconstruction term of the ELBO using negative sampling for efficiency.

```python
import torch
from torch_geometric.nn import VGAE, GCNConv

class GCNEncoder(torch.nn.Module):
    """Two-layer GCN producing per-node mean and log-variance for the latent code."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_shared = GCNConv(in_channels, 2 * out_channels)  # shared first layer
        self.conv_mu = GCNConv(2 * out_channels, out_channels)       # mean head
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)   # log-variance head

    def forward(self, x, edge_index):
        x = self.conv_shared(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# VGAE wraps the encoder and provides the reparameterization trick + inner-product decoder
model = VGAE(GCNEncoder(in_channels=data.num_node_features, out_channels=16))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_vgae(data):
    model.train()
    optimizer.zero_grad()
    # Encode: sample z via reparameterization
    z = model.encode(data.x, data.train_pos_edge_index)
    # Reconstruction loss: -log p(A|Z) approximated via positive + negative edges
    # recon_loss uses negative sampling: compares sampled negative edges against positive ones
    recon_loss = model.recon_loss(z, data.train_pos_edge_index)
    # KL divergence: keeps posterior q(Z|A,X) close to N(0,I)
    kl_loss = (1 / data.num_nodes) * model.kl_loss()
    loss = recon_loss + kl_loss
    loss.backward()
    optimizer.step()
    return float(loss)

# At inference: decode by predicting edge probabilities between node pairs
@torch.no_grad()
def predict_links(data, threshold=0.5):
    z = model.encode(data.x, data.edge_index)
    # predict_links_probabilities returns sigmoid(z_u @ z_v) for all node pairs
    return model.decode(z, data.edge_index)
```

VGAE has an important dual interpretation: in addition to graph generation (sample \( z \sim \mathcal{N}(0, I) \), decode to adjacency), it is a state-of-the-art **link prediction** method. The inner-product decoder directly produces edge probabilities, and VGAE node embeddings trained on the ELBO objective consistently outperform transductive methods on citation and social network link prediction benchmarks. VGAE's primary limitation is scalability: the decoder computes \( O(n^2) \) edge probabilities, making it impractical for graphs with more than a few thousand nodes.

!!! mascot-encourage "VGAE Bridges Generation and Prediction"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage offers encouragement">
    VGAE is one of those models that feels like it is doing two different things but is actually doing one coherent thing very well: learning a continuous embedding of graphs in which proximity in latent space corresponds to structural similarity. The fact that this same embedding produces both good link predictions and plausible sampled graphs is not a coincidence — it is evidence that the ELBO objective has successfully compressed the graph's topology into the latent code. The latent space interpolation experiment (morphing one molecule into another by linear interpolation of their latent codes) is one of the most visually compelling demonstrations of this.

## 21.6 DiGress: Discrete Diffusion for Graphs

The most recent and best-performing graph generation paradigm applies **diffusion models** — previously dominant in image generation — to the discrete domain of graphs. **DiGress** (Vignac et al., 2022) defines a discrete diffusion process directly over node types and edge types, circumventing the need for a sequential autoregressive factorization or an inner-product decoder.

Diffusion models operate in two phases: a **forward process** that gradually corrupts a real data point into pure noise, and a **reverse process** (learned) that denoises a noisy sample back toward the data distribution. For images, both processes operate in continuous space using Gaussian perturbations. For molecular graphs, node types and edge types are categorical, so DiGress uses **discrete Markov noise**: at each noise step \( t \), a random subset of nodes change their atom type and a random subset of edges change their bond type (or are added/deleted), following a transition matrix \( Q_t \) designed so that the marginal distribution at maximum noise is uniform over all atom/bond types.

The forward process defines, for a graph \( G_0 = (A_0, X_0) \) with adjacency \( A_0 \) and node features \( X_0 \):

\[
q(G_t \mid G_{t-1}) = \prod_{v} q(x_v^{(t)} \mid x_v^{(t-1)}) \prod_{(u,v)} q(a_{uv}^{(t)} \mid a_{uv}^{(t-1)})
\]

where each marginal is a Markov transition over the finite set of node types or edge types respectively. The transition matrix \( Q_t \) is designed to interpolate between the data distribution (at \( t = 0 \)) and a uniform categorical distribution (at \( t = T \)), analogously to how Gaussian diffusion interpolates toward \( \mathcal{N}(0, I) \).

The **reverse process** learns to predict \( G_0 \) from \( G_t \): given a noisy graph at time \( t \), a neural network \( f_\theta(G_t, t) \) predicts the clean graph \( \hat{G}_0 \). DiGress uses an **equivariant graph transformer** as the denoiser backbone — a transformer architecture with node-pair attention that is invariant to permutations of the node ordering. This is the key design choice that distinguishes DiGress from simpler baselines: because the denoiser is permutation-equivariant, the model learns a distribution \( p_\theta(G) \) that is truly permutation-invariant, resolving the symmetry problem without committing to a canonical ordering.

Training minimizes a cross-entropy loss between the predicted clean node/edge types and the true ones:

\[
\mathcal{L} = \mathbb{E}_{t, G_0, G_t}\!\left[-\log p_\theta(G_0 \mid G_t, t)\right]
\]

where the expectation is over noise levels \( t \sim \text{Uniform}(1, T) \), training graphs \( G_0 \), and noisy samples \( G_t \sim q(G_t \mid G_0) \). At inference, generation starts from a fully noisy graph \( G_T \sim \text{Uniform} \) and iteratively applies the reverse process for \( T \) steps.

DiGress is not trivial to implement from scratch; the following code sketches the key forward-process noise application using the marginal transition matrix, which is the most instructive component for understanding the discrete diffusion framework.

```python
import torch
import torch.nn.functional as F

def marginal_transition(x_0, Q_bar_t):
    """
    Apply cumulative noise transition to clean node features.
    x_0: (batch, num_nodes) — integer node type indices
    Q_bar_t: (num_node_types, num_node_types) — cumulative transition matrix at time t
             Q_bar_t[i,j] = P(x_t = j | x_0 = i)
    Returns: x_t sampled from q(x_t | x_0)
    """
    num_node_types = Q_bar_t.size(0)
    # One-hot encode x_0
    x_0_onehot = F.one_hot(x_0, num_classes=num_node_types).float()  # (batch, n, K)
    # Transition probabilities: (batch, n, K) @ (K, K) -> (batch, n, K)
    probs = x_0_onehot @ Q_bar_t  # q(x_t | x_0) as a distribution
    # Sample from the categorical distribution
    x_t = torch.distributions.Categorical(probs=probs).sample()
    return x_t


def build_linear_noise_schedule(num_steps, num_node_types, noise_fraction=0.1):
    """
    Linear noise schedule: Q_t interpolates between identity and uniform distribution.
    At t=0: no noise (Q=I). At t=T: uniform over all types.
    """
    Q_bars = []
    for t in range(1, num_steps + 1):
        alpha_t = 1.0 - noise_fraction * t / num_steps
        # Q_bar_t = alpha_t * I + (1 - alpha_t) * (1/K) * 11^T
        uniform = torch.ones(num_node_types, num_node_types) / num_node_types
        identity = torch.eye(num_node_types)
        Q_bar = alpha_t * identity + (1 - alpha_t) * uniform
        Q_bars.append(Q_bar)
    return Q_bars  # list of T transition matrices
```

DiGress achieves state-of-the-art performance on standard molecular generation benchmarks, including QM9 (small organic molecules) and ZINC250k (drug-like molecules), significantly outperforming GraphRNN, GCPN, and VGAE on both validity and distributional fidelity metrics.

## 21.7 Evaluating Graph Generative Models

With four distinct generation paradigms in hand, we now need tools to compare them rigorously. Graph generation evaluation is multi-dimensional: a model that generates valid molecules but only reproduces training set molecules is useless for drug discovery; a model that generates novel molecules that are all chemically invalid is equally useless. The standard evaluation suite for molecular generation covers the following metrics, which we define before examining the benchmark results.

**Validity** measures the fraction of generated molecules that obey chemical valence rules (e.g., carbon has at most 4 bonds, nitrogen at most 3). Validity is computed by parsing each generated graph with the RDKit cheminformatics library. A model that generates random adjacency matrices achieves near-0% validity; GraphRNN achieves ≥ 95% on small molecule datasets by design (the sequential construction adds bonds one at a time with validity masking).

**Uniqueness** measures the fraction of valid generated molecules that are structurally distinct from each other. A degenerate model that generates only the most common training molecule (aspirin, say) achieves 0% uniqueness. Uniqueness is computed by canonicalizing each molecule to a SMILES string and checking for duplicates.

**Novelty** measures the fraction of valid, unique generated molecules that do not appear in the training set. A model that memorizes training data achieves 0% novelty. Novelty is the key metric for drug discovery: a model that regenerates known molecules adds no scientific value.

**Fréchet ChemNet Distance (FCD)** measures the distributional fidelity of the generated molecule set relative to the reference set. FCD embeds each molecule through ChemNet (a chemistry-aware neural network) and computes the Fréchet distance between the Gaussian fit to the generated distribution and the Gaussian fit to the reference distribution — analogously to the Fréchet Inception Distance (FID) used for image generation. Lower FCD indicates closer alignment with the reference chemical property distribution.

**Graph statistics** provide a model-agnostic check of structural fidelity: degree distribution, clustering coefficient, and orbit counts of generated graphs are compared to those of the training set using maximum mean discrepancy (MMD) or Jensen-Shannon divergence. These statistics are especially important for non-molecular graph generation tasks (e.g., generating social networks or citation graphs) where domain-specific chemical validity metrics do not apply.

The following table summarizes how the four model families compare across these dimensions, drawing on the evaluations fully described above.

| Model | Validity | Uniqueness | Novelty | FCD ↓ | Notes |
|-------|:--------:|:----------:|:-------:|:-----:|-------|
| GraphRNN | 97.6% | 99.9% | 99.7% | 2.91 | Sequential; slow on large molecules |
| GCPN | 99.0% | 99.9% | 99.7% | 0.44 | RL-optimized; goal-directed capable |
| VGAE | 81.4% | 99.3% | 99.4% | 5.62 | Scalability-limited; fast training |
| DiGress | **99.0%** | **99.8%** | **99.8%** | **0.05** | Permutation-invariant; SOTA distribution |

!!! mascot-warning "High Validity Alone Does Not Mean Good Generation"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warns about overinterpreting validity">
    GraphRNN's 97.6% validity looks nearly as good as GCPN's 99.0% — but the FCD scores (2.91 vs. 0.44) tell a very different story about chemical property diversity. A model can achieve near-perfect validity by defaulting to structurally simple molecules that trivially satisfy valence rules but have no drug-like properties. Always report all four metrics together; validity alone is a necessary but far from sufficient condition for useful molecular generation. DiGress's FCD of 0.05 means its generated distribution is essentially indistinguishable from the training distribution in chemical property space — a much stronger claim than validity alone.

## 21.8 The Drug Discovery Pipeline with GNNs

We close the chapter by connecting generative models to the full drug discovery pipeline, where DDI and PPI networks play roles alongside molecular graph generation. A production GNN-based drug discovery system typically has three stages:

- **Stage 1 — Target identification.** PPI network analysis with GNN node classification identifies which proteins are implicated in a disease pathway. Nodes with high centrality in the disease-associated subgraph are prioritized as drug targets.
- **Stage 2 — Hit generation.** A generative model (GCPN or DiGress) generates candidate molecules conditioned on predicted binding affinity to the identified target. GNN property predictors (trained on existing drug databases) score each candidate for ADMET properties (absorption, distribution, metabolism, excretion, toxicity).
- **Stage 3 — Safety screening.** The DDI network is queried to check whether any high-scoring candidate molecule has predicted strong interactions with existing drugs already prescribed for the target patient population. GNN-based DDI edge prediction (link prediction on the DDI graph) flags potentially dangerous combinations before clinical testing.

This pipeline integrates four of the chapter's core concepts: molecular graphs (the representation), graph generative models (Stage 2), PPI networks (Stage 1), and DDI networks (Stage 3). The key insight is that GNNs serve different architectural roles in each stage — discriminative (property prediction, link prediction) in Stages 1 and 3, generative in Stage 2 — but the underlying graph neural network machinery is shared.

#### Diagram: Drug Discovery GNN Pipeline


<iframe src="../../sims/ch21-drug-discovery-pipeline/main.html" width="100%" height="522px" scrolling="no"></iframe>
[Run Drug Discovery GNN Pipeline Fullscreen](../../sims/ch21-drug-discovery-pipeline/main.html)

## 21.9 Common Pitfalls

**Optimizing validity at the expense of novelty.** Applying an aggressive chemical validity mask during generation (GCPN-style) trivially produces high-validity molecules — but validity masking can also suppress chemical diversity by forcing the policy toward safe, predictable bond choices. Monitor novelty and FCD alongside validity; if novelty drops below 90%, the model may be collapsing toward a small set of "safe" scaffolds.

**Training VGAE on unbalanced positive/negative edges.** The inner-product decoder outputs sigmoid probabilities for all \( O(n^2) \) node pairs, but real molecular graphs are sparse — typically fewer than 0.1% of pairs are bonded. Without negative sampling or positive-class upweighting in the reconstruction loss, the decoder learns to predict 0 for all pairs (trivially low binary cross-entropy on an imbalanced dataset). Always use balanced positive/negative sampling or focal loss when training VGAE.

**Comparing model families without controlling for training data.** GraphRNN, GCPN, VGAE, and DiGress have been evaluated on different subsets and splits of molecular databases in the original papers. FCD, validity, and novelty scores are not directly comparable across papers unless the exact same training/test split and reference set are used. When reproducing results or comparing methods, always use a standardized benchmark (GuacaMol or MOSES) with fixed splits.

**Using SMILES-based generation metrics on graph-based models.** Some evaluation pipelines convert generated graphs to SMILES strings and use SMILES-based metrics (e.g., SMILES uniqueness by string comparison). This can undercount duplicates: two different SMILES strings may represent the same molecule under canonical form. Always canonicalize SMILES through RDKit before computing uniqueness and novelty.

## 21.10 Further Reading

- **You et al. (2018) — GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models.** The original sequential graph generation paper; Section 3.3 on BFS ordering is the key technical contribution. [arXiv:1802.08773](https://arxiv.org/abs/1802.08773)

- **You et al. (2018) — Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation.** The RL formulation for goal-directed generation; the reward function design and the chemistry validity mask are the most reusable engineering contributions. [arXiv:1806.02473](https://arxiv.org/abs/1806.02473)

- **Kipf & Welling (2016) — Variational Graph Auto-Encoders.** Short paper (4 pages) that introduces VGAE; notable for how much it achieves with a simple inner-product decoder. The link prediction results on Cora and Citeseer set the standard for subsequent work. [arXiv:1611.07308](https://arxiv.org/abs/1611.07308)

- **Vignac et al. (2022) — DiGress: Discrete Denoising Diffusion for Graphs with Transformers.** The equivariant transformer denoiser and the discrete Markov noise process are the technical core; Appendix B provides full transition matrix derivations. [arXiv:2209.14734](https://arxiv.org/abs/2209.14734)

- **Brown et al. (2019) — GuacaMol: Benchmarking Models for de Novo Molecular Design.** The standardized evaluation benchmark for molecular generation; defines the validity, uniqueness, novelty, and FCD metrics used in §21.7 and provides the reference dataset and evaluation code. [arXiv:1811.09621](https://arxiv.org/abs/1811.09621)

- **Polykovskiy et al. (2020) — Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models.** An alternative standardized benchmark with the MOSES dataset (1.9M drug-like molecules) and additional metrics (internal diversity, scaffold similarity). [arXiv:1811.12823](https://arxiv.org/abs/1811.12823)

- **Gaudelet et al. (2021) — Utilizing Graph Machine Learning within Drug Discovery and Development.** A comprehensive survey of GNN applications across the full drug discovery pipeline (target identification, hit generation, lead optimization, ADMET prediction); useful for understanding where each generation paradigm fits in practice. [arXiv:2012.05716](https://arxiv.org/abs/2012.05716)

## 21.11 Exercises

**Remembering**

1. Define the three fundamental challenges of graph generation (discrete structure, variable size, permutation invariance) and explain why each makes graph generation harder than image generation.

2. List the four evaluation metrics for molecular graph generation described in §21.7. For each metric, state whether higher or lower values are better and what a degenerate model that generates only single-atom graphs would score.

**Understanding**

3. GraphRNN uses BFS node ordering to reduce sequence length. Explain why BFS ordering limits the edge sequence length and what would go wrong if nodes were ordered randomly instead of by BFS.

4. VGAE decodes edge probabilities using an inner product \( \hat{A}_{uv} = \sigma(z_u^\top z_v) \). Explain what geometric property of the latent space this decoder assumes, and describe one class of molecular graphs (in terms of their topology) for which this decoder would consistently fail to reconstruct edges accurately.

**Applying**

5. Modify the GCPN reward function \( r(G) \) from §21.4 to additionally penalize molecules with molecular weight above 500 Da (Lipinski's Rule of Five) and reward molecules with hydrogen bond donors \( \leq 5 \). Write the updated reward expression with appropriate \( \lambda \) weights.

6. Using the `build_linear_noise_schedule` function from §21.6, compute the transition matrix \( Q_{\bar{t}} \) for \( t = T/2 \) with \( T = 1000 \) steps, `num_node_types = 6`, and `noise_fraction = 0.1`. What fraction of the diagonal entries are greater than 0.9 at \( t = T/2 \)?

**Analyzing**

7. Compare GraphRNN and DiGress on the permutation invariance problem. GraphRNN commits to BFS ordering; DiGress uses a permutation-equivariant transformer. Describe one concrete failure mode for GraphRNN that arises from BFS-ordering bias, and explain how DiGress's equivariant backbone avoids this failure.

8. VGAE achieves 81.4% validity on molecular generation benchmarks, substantially below GraphRNN (97.6%) and GCPN (99.0%). Analyze why the inner-product decoder, while elegant for link prediction, systematically produces invalid molecules at a higher rate than sequential generative models.

**Evaluating**

9. A pharmaceutical company has two requirements: (a) generated molecules must have ≥ 99% validity, and (b) the chemical property distribution of generated molecules must closely match known drug-like compounds (low FCD). Based on the benchmark table in §21.7, which model best satisfies both requirements? Justify your choice and identify any remaining limitation of that model for production drug discovery.

10. DiGress achieves the best FCD (0.05) among all models but requires \( T = 1000 \) denoising steps at inference, each involving a full graph transformer forward pass. A researcher proposes using \( T = 50 \) steps to speed up inference. Predict the effect on each of the four evaluation metrics (validity, uniqueness, novelty, FCD) and justify your prediction based on how discrete diffusion works.

**Creating**

11. Design a **conditional DiGress** model for targeted drug generation. Your design must: (a) specify how the target protein's binding pocket (represented as a 3D point cloud) is encoded and injected as conditioning signal into the graph transformer denoiser, (b) describe the modified training objective that conditions generation on the protein context, and (c) propose an evaluation protocol that measures whether generated molecules actually bind to the specified protein target.

12. Propose a hybrid generative model that combines GCPN's RL reward optimization with DiGress's permutation-invariant diffusion backbone. Concretely: (a) describe how you would define the RL action space for a diffusion-based model (hint: the "action" is a choice of denoising trajectory), (b) design a reward function that guides the diffusion reverse process toward high-QED molecules, and (c) identify the main technical challenge in backpropagating through the discrete sampling steps and propose one approach to address it.

---

!!! mascot-celebration "Graphs That Have Never Existed Before"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrates">
    You have now seen four ways to build machines that create novel graphs: GraphRNN's sequential storytelling, GCPN's reward-driven chemistry, VGAE's continuous latent compression, and DiGress's diffusion-based denoising. Each is a different answer to the same deep question — what structure should we impose on the generation process? — and the drug discovery pipeline shows how discriminative GNNs (for PPI and DDI analysis) and generative GNNs (for molecule design) combine into something far more powerful than either alone. The field moves fast here; by the time you read this, there will be new architectures that outperform DiGress. But the fundamental tradeoffs — sequential vs. latent, discrete vs. continuous, goal-directed vs. unconditional — will still be the right frame for evaluating them.

[See Annotated References](./references.md)
