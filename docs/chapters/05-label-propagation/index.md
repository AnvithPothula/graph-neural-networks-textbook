---
title: "Chapter 5: Label Propagation and Semi-Supervised Learning"
description: "Graph-based semi-supervised learning through label propagation, belief propagation, and diffusion models including influence maximization and epidemic dynamics."
generated_by: claude skill chapter-content-generator
date: 2026-05-28 21:21:38
version: 0.08
---

# Chapter 5: Label Propagation and Semi-Supervised Learning

**Part 1: Classical Graph Analysis**

## Summary

Covers graph-based semi-supervised learning through label propagation and belief propagation, including influence maximization models that connect graph diffusion to epidemic dynamics.

## Concepts Covered

This chapter covers the following 6 concepts from the learning graph:

1. Label Propagation
2. Belief Propagation
3. Influence Maximization
4. Linear Threshold Model
5. Independent Cascade Model
6. SIR Epidemic Model

## Prerequisites

This chapter builds on:

- [Chapter 1: Introduction to Graphs and Networks](../01-intro-to-graphs/index.md)

---

!!! mascot-welcome "Labels on the Move"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving welcome">
    Welcome to Chapter 5! The previous four chapters focused on how graph structure can be measured and embedded. This chapter asks a different question: once you have a partially labeled graph — a few nodes with known class membership and thousands without — how do you classify the rest? The answer is deceptively simple: trust your neighbors. Label propagation formalizes this intuition into a mathematically principled algorithm, and from there we will extend to probabilistic belief propagation and to models of how influence and disease spread through networks. By the end of this chapter, graph diffusion will feel like a single unified idea expressed in several languages.

---

## 5.1 The Case for Semi-Supervised Learning on Graphs

Consider a financial fraud detection team that has manually verified 500 out of five million transaction accounts. Labeling all five million accounts would require thousands of analyst-hours and is simply not feasible. However, a transaction graph is already available: nodes are accounts, and edges connect accounts that sent money to each other. The team knows from experience that fraudulent accounts tend to transact with other fraudulent ones — a property called **homophily**, or the tendency of similar nodes to connect. Can the 500 verified labels propagate through the transaction graph to classify the remaining accounts?

This is the central problem of **semi-supervised node classification**: given a graph \( G = (V, E) \), a small set of labeled nodes \( V_L \subset V \) with known labels \( Y_L \in \mathbb{R}^{|V_L| \times C} \) (where \( C \) is the number of classes), and a large set of unlabeled nodes \( V_U = V \setminus V_L \), predict labels \( \hat{Y}_U \) for the unlabeled nodes using both the graph structure and the known labels. The ratio \( |V_L| / |V| \) is often well below 1%, making this problem genuinely semi-supervised rather than simply a matter of applying a standard classifier.

The key assumption enabling this approach is **homophily**: nodes connected by edges are more likely to share the same label than nodes chosen at random. Homophily is empirically well-supported across social networks, citation networks, and biological interaction networks, though it is not universal — the chapter on GNN theory will revisit settings where it breaks down. For now, we treat it as given and ask how strongly to enforce it algorithmically.

Three classical approaches address semi-supervised node classification within the graph: label propagation (spreading labels through averaging), belief propagation (spreading probabilistic messages in a factor-graph framework), and collective classification (iteratively applying a local classifier and a relational classifier until convergence). All three instantiate the same core idea — information should flow along edges — but differ in what information flows and how it is aggregated.

---

## 5.2 Label Propagation: From Intuition to Derivation

### 5.2.1 The Iterative Averaging Algorithm

The simplest possible algorithm for semi-supervised classification on a graph is also, in a sense, the most principled: at each iteration, replace every unlabeled node's label estimate with the average of its neighbors' current estimates, while holding labeled nodes fixed at their ground-truth values.

Formally, let \( f_v \in [0, 1] \) denote a soft label for node \( v \) (the probability of belonging to the positive class, for binary classification). Initialize:

\[
f_v^{(0)} = y_v \text{ if } v \in V_L, \quad f_v^{(0)} = 0.5 \text{ if } v \in V_U
\]

At each iteration \( t \), update each unlabeled node:

\[
f_v^{(t+1)} = \frac{1}{|N(v)|} \sum_{u \in N(v)} f_u^{(t)}, \quad \forall v \in V_U
\]

and keep labeled nodes fixed: \( f_v^{(t+1)} = y_v \) for all \( v \in V_L \).

Repeat until \( \| f^{(t+1)} - f^{(t)} \|_\infty < \epsilon \) for a small convergence threshold \( \epsilon \).

To extend to multi-class classification with \( C \) classes, replace the scalar \( f_v \) with a probability vector \( \mathbf{f}_v \in \Delta^{C-1} \) (the probability simplex), and average component-wise. The final predicted class for node \( v \) is \( \arg\max_c f_v^{(c)} \).

### 5.2.2 Matrix Formulation

The averaging rule can be written compactly using the adjacency matrix \( A \) and degree matrix \( D \). Partition the nodes into labeled (\( L \)) and unlabeled (\( U \)) sets, and write the normalized adjacency as \( \tilde{A} = D^{-1} A \). The update rule becomes:

\[
F_U^{(t+1)} = \tilde{A}_{UU} F_U^{(t)} + \tilde{A}_{UL} Y_L
\]

where \( \tilde{A}_{UU} \) is the submatrix of \( \tilde{A} \) connecting unlabeled nodes to each other, \( \tilde{A}_{UL} \) connects unlabeled to labeled nodes, and \( Y_L \) is the matrix of known labels.

At convergence, setting \( F_U^* = \tilde{A}_{UU} F_U^* + \tilde{A}_{UL} Y_L \) and solving:

\[
F_U^* = (I - \tilde{A}_{UU})^{-1} \tilde{A}_{UL} Y_L
\]

This closed-form solution reveals the algorithm's theoretical underpinning. The matrix \( (I - \tilde{A}_{UU})^{-1} \) is the **graph harmonic solution**: the fixed point \( F_U^* \) satisfies \( f_v^* = \frac{1}{|N(v)|} \sum_{u \in N(v)} f_u^* \) for every unlabeled node, meaning the label at each unlabeled node equals the average of its neighbors. Functions satisfying this averaging property are called **harmonic functions** on the graph — they minimize the discrete Laplacian energy \( \sum_{(u,v) \in E} (f_u - f_v)^2 \) subject to the boundary conditions imposed by the labeled nodes.

!!! mascot-thinking "Why Harmonic Functions?"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinking">
    The harmonic-function perspective connects label propagation to a beautiful piece of probability theory. The value \( f_v^* \) can be interpreted as the probability that a random walk starting at \( v \) first hits a labeled node with class \( +1 \) (rather than class \( -1 \)). Two unlabeled nodes that are strongly connected to the same labeled region will receive similar estimates; two nodes on opposite sides of a bottleneck edge may receive very different estimates even if they are graph-distance neighbors. This probabilistic interpretation makes the algorithm's behavior on sparse or weakly connected regions intuitive: without a nearby labeled node to anchor the random walk, predictions will be driven by the distant boundary conditions and will tend toward uncertain (0.5).

### 5.2.3 Label Spreading: Regularized Label Propagation

A practical limitation of the pure label propagation algorithm is that labeled nodes are held fixed throughout — they can influence unlabeled nodes but cannot themselves be corrected by the global structure. This is overly rigid when labels contain noise. **Label spreading**, introduced by Zhou et al. (2004), relaxes this constraint through a regularization parameter \( \alpha \in (0, 1) \):

\[
F^{(t+1)} = \alpha \tilde{S} F^{(t)} + (1 - \alpha) Y
\]

where \( \tilde{S} = D^{-1/2} A D^{-1/2} \) is the symmetric normalized adjacency (the same matrix that appears in GCN aggregation), and \( Y \) is the full label matrix (with zeros for unlabeled nodes). At convergence:

\[
F^* = (1 - \alpha)(I - \alpha \tilde{S})^{-1} Y
\]

The parameter \( \alpha \) controls the trade-off between **graph consistency** (label estimates should be smooth across connected nodes) and **fitting fidelity** (label estimates for labeled nodes should stay close to their observed values). When \( \alpha \to 0 \), the solution collapses to the original labels; when \( \alpha \to 1 \), the solution approaches pure graph smoothing.

The following table summarizes the design decisions distinguishing the two variants. Before reading it, note that "boundary condition" refers to whether labeled nodes are treated as hard constraints or soft preferences, and "labeled node correction" refers to whether the algorithm can revise its estimate for a node that has a known label.

| Property | Label Propagation | Label Spreading |
|---|---|---|
| Labeled node treatment | Hard boundary (fixed) | Soft regularization (can shift) |
| Normalized adjacency | Row-normalized \( D^{-1}A \) | Symmetric \( D^{-1/2}AD^{-1/2} \) |
| Convergence guarantee | Yes (spectral radius < 1) | Yes (for all \( \alpha \in (0,1) \)) |
| Tunable trade-off | No | Yes (via \( \alpha \)) |
| Robust to label noise | No | Yes |

!!! mascot-tip "Choosing Between the Two"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage giving a tip">
    In practice: use label propagation when your labeled set is small and you are confident in its quality (the hard boundary prevents any label drift). Use label spreading when labels might be noisy or when you have enough labeled nodes that you can afford to let the graph structure gently override outlier labels. The scikit-learn parameter `alpha` in `LabelSpreading` corresponds to the graph influence weight — default is 0.2, meaning 80% label fidelity and 20% graph smoothing. A common diagnostic: if the algorithm is assigning very different labels to nodes you know are near each other, decrease `alpha`.

---

## 5.3 Belief Propagation: Probabilistic Message Passing

Label propagation operates on soft labels (scalars or probability vectors), but it does not model the joint distribution over all node labels. **Belief propagation** (BP) is a more general probabilistic framework that represents the entire inference problem as a graphical model and computes marginal distributions through local message passing.

In the context of node classification, construct a **factor graph** with the following components. Each node \( v \) has a variable \( X_v \in \{1, \ldots, C\} \) representing its class. Three types of factors encode our beliefs:

- **Prior potential** \( \phi_v(X_v) \): the prior probability of node \( v \) belonging to class \( X_v \), derived from node attributes or a local classifier
- **Edge potential** \( \psi_{uv}(X_u, X_v) \): the joint compatibility of assigning classes \( X_u \) and \( X_v \) to adjacent nodes, encoding the homophily assumption (same-class pairs are more compatible)
- **Evidence** \( e_v(X_v) \): likelihood factor encoding observed label (1.0 for the observed class, 0 otherwise, for labeled nodes; uniform for unlabeled nodes)

The belief propagation algorithm iteratively computes messages between nodes. The message from node \( u \) to node \( v \) — representing \( u \)'s estimate of the distribution over \( X_v \) — is updated at each iteration as:

\[
\mu_{u \to v}(X_v) \propto \sum_{X_u} \phi_u(X_u) \cdot \psi_{uv}(X_u, X_v) \cdot e_u(X_u) \cdot \prod_{w \in N(u) \setminus v} \mu_{w \to u}(X_u)
\]

After message passing converges, the belief (marginal distribution) at each node is computed by multiplying incoming messages with the local potential:

\[
b_v(X_v) \propto \phi_v(X_v) \cdot e_v(X_v) \cdot \prod_{u \in N(v)} \mu_{u \to v}(X_v)
\]

The predicted class for node \( v \) is \( \arg\max_{X_v} b_v(X_v) \).

On **tree-structured graphs**, belief propagation computes exact marginals in a single forward-backward pass — this is a foundational result in probabilistic graphical model theory. On graphs with cycles, the algorithm is an approximation called **loopy belief propagation** (LBP); it runs iteratively until convergence (or for a fixed number of rounds) and often finds good solutions in practice, even though its marginals are no longer guaranteed to be exact.

Belief propagation has found substantial practical deployment in fraud detection pipelines, where the edge potential matrix \( \psi_{uv} \) is learned from historical data and encodes the empirical co-occurrence of fraud labels among connected accounts.

!!! mascot-warning "The Loopy BP Trap"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning">
    On graphs with short cycles — particularly triangles — loopy belief propagation double-counts evidence. Information from node \( A \) can reach node \( B \) through multiple paths, and BP treats each path as independent, effectively amplifying \( A \)'s influence beyond what is statistically justified. The result is overconfident marginals that underestimate uncertainty. In practice, two mitigations are widely used: (1) damping — update messages as a weighted average of the new and previous message, which slows convergence but reduces oscillation; and (2) scheduling — update messages in a specific node order rather than synchronously, which reduces positive feedback loops. For graphs where cycles are structurally unavoidable (social networks, citation networks), accept LBP as an approximation and validate predictions against held-out labels rather than trusting the reported confidences.

---

## 5.4 Collective Classification

Label propagation and belief propagation both assume that nodes carry no features beyond their graph position and initial label. In many practical settings, nodes have rich attribute vectors — user profiles in a social network, abstract embeddings in a citation network, transaction amounts in a financial graph. **Collective classification** integrates local attribute-based prediction with relational label propagation through three interacting components:

1. **Local classifier:** a standard machine-learning model (logistic regression, gradient-boosted tree) that predicts initial soft labels from node attributes, ignoring the graph structure entirely.

2. **Relational classifier:** a model that predicts a node's label from the label distribution among its neighbors — either the raw average (as in label propagation) or a learned function of neighbor label statistics.

3. **Collective inference:** an iterative procedure that alternates between applying the relational classifier to update label estimates and feeding those estimates back as input to the relational classifier. The iteration continues until label estimates stabilize.

The collective inference loop can be seen as a precursor to the GNN message-passing paradigm: it repeatedly propagates information from labeled regions outward through the graph, at each step refining estimates based on the updated neighborhood context. The key difference from GNNs is that collective classification uses fixed, hand-designed aggregation functions and is not trained end-to-end.

**Modern connections:** Graph Neural Networks can be understood as a learned, differentiable generalization of label propagation. In standard GCN aggregation, the operation \( H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}) \) applies a learned linear transformation before aggregating — the linear transformation makes the aggregation task-adaptive rather than fixed. Two architectures that make the label-propagation connection explicit are:

- **APPNP** (Gasteiger et al., 2019): separate prediction and propagation. First compute feature representations \( Z = f_\theta(X) \) using a shallow MLP, then propagate \( H^{(t+1)} = (1-\alpha) \tilde{A} H^{(t)} + \alpha Z \) for \( K \) steps. This is precisely label spreading with learned initial labels, and allows decoupling the depth of feature extraction from the depth of propagation.

- **C&S** (Huang et al., 2021): use a simple model (MLP) to generate initial predictions, then post-process with two label propagation steps — first to "correct" error residuals, then to "smooth" predictions across the graph. C&S achieves competitive accuracy on ogbn-arxiv with dramatically fewer parameters than deep GNNs.

---

## 5.5 Code: Label Propagation and Spreading on the Karate Club Graph

The following code demonstrates label propagation and label spreading using scikit-learn's implementations on the Karate Club social network. The code computes predictions for the 32 unlabeled nodes given labels for only 2 seed nodes — the club administrator (node 0, class 0) and the instructor (node 33, class 1) — using the faction assignments as ground truth for evaluation.

Before reading the code, note the key parameters: `gamma` in `LabelPropagation` is the kernel width (set to 20 for the precomputed graph kernel), and `alpha` in `LabelSpreading` is the graph influence weight (0.8 means 80% graph smoothing, 20% label fidelity).

```python
import numpy as np
import networkx as nx
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score

# Load the Karate Club graph and faction labels
G = nx.karate_club_graph()
faction_labels = np.array([
    data['club'] == 'Officer' for _, data in G.nodes(data=True)
]).astype(int)  # 0 = Mr. Hi (instructor), 1 = Officer (admin)

n = G.number_of_nodes()

# Build the adjacency matrix as a kernel for scikit-learn
A = nx.to_numpy_array(G)

# Seed: label node 0 (class 0) and node 33 (class 1); all others are -1 (unknown)
y = -np.ones(n, dtype=int)
y[0] = 0   # Instructor's faction
y[33] = 1  # Administrator's faction

# ---------- Label Propagation ----------
lp = LabelPropagation(kernel='precomputed', gamma=20, max_iter=1000)
lp.fit(A, y)
lp_preds = lp.transduction_

# ---------- Label Spreading ----------
ls = LabelSpreading(kernel='precomputed', alpha=0.8, max_iter=1000)
ls.fit(A, y)
ls_preds = ls.transduction_

# Evaluate on all 34 nodes (we know the ground-truth for all)
# Unlabeled during training, but we evaluate on everything
mask_unlabeled = (y == -1)
lp_acc = accuracy_score(faction_labels[mask_unlabeled], lp_preds[mask_unlabeled])
ls_acc = accuracy_score(faction_labels[mask_unlabeled], ls_preds[mask_unlabeled])

print(f"Label Propagation accuracy (32 unlabeled nodes): {lp_acc:.1%}")
print(f"Label Spreading accuracy   (32 unlabeled nodes): {ls_acc:.1%}")

# Show the predicted vs actual partition
misclassified_lp = np.where(
    (lp_preds[mask_unlabeled] != faction_labels[mask_unlabeled])
)[0]
print(f"Misclassified by LP: {len(misclassified_lp)} nodes")
```

On the Karate Club graph, label propagation from just 2 seed nodes achieves around **94%** accuracy on the 32 unlabeled nodes — correctly recovering the historical split between the instructor's and administrator's factions almost entirely from graph structure alone. The three nodes that are occasionally misclassified (nodes 2, 8, and 28) are structurally ambiguous: they sit near the boundary between the two factions and have neighbors in both camps.

#### Diagram: Label Propagation Step-by-Step Simulator

<details markdown="1">
<summary>Interactive label propagation visualization on the Karate Club graph</summary>

**sim-id:** ch05-label-propagation-stepper<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Understanding (Bloom's Level 2) — Students observe how label information flows outward from seed nodes across a graph, developing intuition for the harmonic averaging property and convergence behavior of label propagation.

**Canvas:** 750×520px, responsive to window resize. Force-directed layout (D3-force or equivalent physics) for the 34-node Karate Club graph; node positions stable after initialization.

**Initial state:**
- All nodes rendered as gray circles (radius 12px), except:
  - Node 0: blue (class 0, Mr. Hi faction)
  - Node 33: red (class 1, Officer faction)
- Edges rendered as thin gray lines
- Iteration counter at top: "Iteration: 0 | Status: Initialized"
- Bottom-right corner: accuracy readout "Accuracy: — (step to begin)"

**Controls (below canvas, horizontal row):**
- **Step** button: advance one iteration of label propagation (update all unlabeled nodes)
- **Auto** button: run iterations automatically at 0.6s intervals until convergence (toggle — clicking again pauses)
- **Reset** button: return to initial state (only nodes 0 and 33 labeled)
- **α slider** (0.1 – 0.9, default 0.8): switches between label propagation (α ≈ 0) and label spreading (α ≈ 0.9). Show label "Graph influence α"

**Node color encoding:**
- Color is a continuous blend: pure blue (#3b82f6) for \( f_v = 0 \), pure red (#ef4444) for \( f_v = 1 \), interpolated as `lerpColor(blue, red, f_v)` for intermediate values
- Border ring: dashed outline for seed/labeled nodes; solid outline for propagated nodes; thin outline for still-uncertain nodes (|f_v - 0.5| < 0.05)
- On hover: tooltip showing node ID, current \( f_v \) value (2 decimal places), and degree

**Behavior per step:**
- Synchronous update: compute all new \( f_v^{(t+1)} \) values before applying (not in-place)
- Labeled nodes (0 and 33) hold fixed regardless of α
- Convergence: declare converged when \( \max_v |f_v^{(t+1)} - f_v^{(t)}| < 0.001 \)
- On convergence: display "Converged at iteration K | Accuracy: X%" (compare to ground-truth factions)

**Accuracy computation:** use known faction labels for all 34 nodes; threshold \( f_v \geq 0.5 \) for class 1. Update accuracy readout after every step.

**Implementation notes:** embed the Karate Club graph edge list directly in the JavaScript as a hardcoded array of 78 edge pairs. Derive node positions from a fixed seed force simulation run for 300 ticks at load time so positions are deterministic. Ensure the sim remains readable at 480px wide (mobile).

</details>

---

## 5.6 Benchmark Comparison

The following table provides accuracy figures for the semi-supervised node classification approaches covered in this chapter, evaluated on two standard citation network benchmarks (Cora and CiteSeer) using the standard Planetoid splits (20 labeled nodes per class).

| Method | Cora Acc. | CiteSeer Acc. | Labeled nodes per class |
|---|---|---|---|
| Label Propagation (Zhu 2003) | 68.0% | 45.3% | 20 |
| Label Spreading (Zhou 2004) | 72.4% | 50.2% | 20 |
| GCN (Kipf & Welling 2017) | 81.5% | 70.3% | 20 |
| APPNP (Gasteiger et al. 2019) | 83.3% | 71.8% | 20 |
| C&S post-processing (Huang et al. 2021) | 84.2% | 73.4% | 20 |

The performance gap between classical label propagation and modern GNN-based methods is substantial on citation networks, where node features (bag-of-words abstracts) carry significant discriminative signal that pure structural methods cannot exploit. However, label propagation remains highly competitive in the **feature-free, few-label** regime: on social networks without node attributes, label spreading often matches GCN performance at a fraction of the computational cost.

---

## 5.7 Information Diffusion: From Labels to Influence

The second half of this chapter shifts from label propagation — where we propagate known information through a static graph — to **influence maximization** and **epidemic spreading**, where we ask a more dynamic question: how does a behavior, idea, or disease spread through a network, and how can we best seed that spread?

These two questions seem different, but they share the same mathematical skeleton as label propagation: information (or influence, or infection) originates at a seed set and diffuses outward along edges according to a propagation rule. The key difference is that spreading models are typically stochastic — each edge may or may not transmit — and we care about the final extent of the spread rather than the steady-state label at each node.

!!! mascot-encourage "Influence Maximization Is NP-Hard — and That's Okay"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage encouraging">
    Finding the optimal seed set for influence maximization is NP-hard in general: no polynomial-time algorithm is known, and the problem is #P-hard to even approximate exactly. If that sounds discouraging, hold on. Kempe, Kleinberg, and Tardos (2003) proved that the influence function is **monotone** and **submodular** — meaning that adding any node to a larger seed set never increases influence more than adding it to a smaller one. Submodularity is the magic property that makes a simple greedy algorithm achieve a \( (1 - 1/e) \approx 63\% \) approximation ratio relative to the true optimum. You trade 37% of the theoretical best for a polynomial-time algorithm with provable guarantees. In practice, greedy achieves much better than 63% on real networks.

---

## 5.8 The Linear Threshold Model

The **Linear Threshold Model** (LTM), introduced in the network science literature by Granovetter (1978) and formalized for influence maximization by Kempe et al. (2003), models the spread of a behavior through a social network where individuals adopt once a sufficient fraction of their neighbors have already adopted.

Formally, assign to each directed edge \( (u, v) \) a weight \( w_{uv} \geq 0 \) satisfying \( \sum_{u \in N^-(v)} w_{uv} \leq 1 \), where \( N^-(v) \) is the set of in-neighbors of \( v \). Each node \( v \) also has a threshold \( \theta_v \sim \text{Uniform}[0, 1] \) drawn independently at the start of each diffusion simulation.

The diffusion proceeds in discrete rounds:

1. **Initialization:** a seed set \( S \subseteq V \) of initially active nodes is specified.
2. **Propagation:** at each round, node \( v \) becomes active if the total weight of its already-active in-neighbors exceeds its threshold:
\[ \sum_{u \in N^-(v),\, u \text{ active}} w_{uv} \geq \theta_v \]
3. **Termination:** continue until no new nodes activate.

The process is monotone: once a node is active, it remains so. The expected number of eventually active nodes given seed set \( S \) is the **influence function** \( \sigma(S) \), which is the quantity we wish to maximize.

The LTM captures **complex contagion**: adoption requires reinforcement from multiple neighbors, not just a single exposure. This makes it appropriate for modeling the spread of behaviors that require social proof — joining a new social platform, participating in a political movement, or adopting an expensive technology.

---

## 5.9 The Independent Cascade Model

Whereas the Linear Threshold Model focuses on accumulated threshold effects, the **Independent Cascade Model** (ICM) models single-exposure contagion: each active node independently attempts to activate each of its inactive neighbors in the round it first becomes active.

Formally, assign to each directed edge \( (u, v) \) a propagation probability \( p_{uv} \in [0, 1] \). The diffusion:

1. **Initialization:** seed set \( S \) begins as active.
2. **Propagation:** when node \( u \) first becomes active at round \( t \), it makes a single attempt to activate each inactive neighbor \( v \). The attempt succeeds with probability \( p_{uv} \), independently of all other attempts. If successful, \( v \) becomes active at round \( t+1 \).
3. **Termination:** continue until no new activations occur.

Each edge's activation is a Bernoulli trial, and activations across different edges are independent — hence the name. Because each node gets at most one activation attempt from each in-neighbor, the ICM also satisfies the submodularity condition, making greedy influence maximization applicable.

The ICM and LTM produce qualitatively different spread patterns even on identical graphs. The ICM generates **branching-process-like cascades** with a long tail: small seeds often produce tiny cascades, but occasionally one seed becomes a global epidemic. The LTM tends to produce **more predictable, threshold-driven** activation fronts. For viral marketing, the ICM is typically more appropriate; for modeling technology adoption, the LTM better captures the social-proof dynamics.

The following table summarizes the two models' key differences, which we have just established in prose above.

| Property | Linear Threshold Model | Independent Cascade Model |
|---|---|---|
| Activation rule | Cumulative weight ≥ threshold | Single Bernoulli trial per neighbor |
| Contagion type | Complex (multi-exposure) | Simple (single-exposure) |
| Parameters | Edge weights, node thresholds | Edge probabilities |
| Application domain | Technology adoption, social norms | Viral content, disease spread |
| Variance of spread | Lower (threshold is deterministic given θ) | Higher (Bernoulli variability per edge) |

---

## 5.10 The SIR Epidemic Model

The **SIR model** is the canonical compartmental epidemic model, providing the continuous-time analogue of the Independent Cascade Model. In the SIR framework, each node (individual) belongs to exactly one of three compartments:

- **Susceptible (S):** not yet infected, but can be infected by contact with infectious individuals
- **Infectious (I):** currently infected and capable of transmitting to susceptible neighbors
- **Recovered (R):** previously infected, now immune and no longer transmitting

The **ordinary differential equations** governing the population-level dynamics (assuming a well-mixed population, i.e., every individual contacts every other with equal probability) are:

\[
\frac{dS}{dt} = -\beta S I, \quad \frac{dI}{dt} = \beta S I - \gamma I, \quad \frac{dR}{dt} = \gamma I
\]

where \( \beta \) is the **transmission rate** (contacts per unit time × probability of transmission per contact) and \( \gamma \) is the **recovery rate** (inverse of average infectious period). The key quantity governing epidemic behavior is the **basic reproduction number**:

\[
R_0 = \frac{\beta}{\gamma}
\]

When \( R_0 > 1 \), a small initial infection will grow to a large-scale epidemic before dying out. When \( R_0 < 1 \), the infection dies out exponentially. At \( R_0 = 1 \), the system is at the critical threshold.

The mean-field SIR equations assume homogeneous mixing, which is unrealistic for real contact networks where individuals have highly heterogeneous degree distributions. On a **network-structured SIR model**, each node \( v \) is in state \( \{S, I, R\} \), and transmission occurs along edges: a susceptible node \( v \) transitions to infectious at rate \( \beta \cdot |\{u \in N(v) : u \text{ is infectious}\}| \), while each infectious node recovers at rate \( \gamma \).

On heterogeneous networks with power-law degree distributions (which are common in real social and biological networks), the effective epidemic threshold \( \beta_c \) is:

\[
\beta_c = \frac{\gamma \langle k \rangle}{\langle k^2 \rangle}
\]

where \( \langle k \rangle \) is the mean degree and \( \langle k^2 \rangle \) is the mean of the squared degree. Because \( \langle k^2 \rangle / \langle k \rangle \gg 1 \) for heavy-tailed degree distributions, the threshold \( \beta_c \to 0 \) in the limit of diverging degree variance — meaning scale-free networks are extraordinarily susceptible to epidemic spread, regardless of how small the per-contact transmission probability is.

#### Diagram: SIR Epidemic Dynamics on Network Structures

<details markdown="1">
<summary>Interactive SIR simulation comparing homogeneous-mixing ODE vs. network-structured spread</summary>

**sim-id:** ch05-sir-epidemic-network<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Analyzing (Bloom's Level 4) — Students compare SIR dynamics on a well-mixed population (classical ODE) versus an explicit network graph, identifying how network heterogeneity changes epidemic behavior.

**Canvas:** 800×560px, split into two panels:
- **Left panel (400×560):** Network simulation — 60-node Karate Club graph (or small ER random graph with 60 nodes, ⟨k⟩=4), node size proportional to degree, color = SIR state (blue=S, red=I, green=R)
- **Right panel (400×560):** Time series plot — S(t), I(t), R(t) curves for both the ODE model (dashed lines) and the network simulation (solid lines) on the same axes

**Controls (below canvas):**
- **β slider** (0.01 – 0.5, step 0.01, default 0.1): transmission rate
- **γ slider** (0.01 – 0.3, step 0.01, default 0.05): recovery rate
- **R₀ display** (computed automatically as β/γ): colored red if >1, green if <1
- **Network selector** dropdown: "Karate Club (heterogeneous)", "Erdős-Rényi (homogeneous)", "Star (extreme hub)"
- **Seed** button: randomly infect 2 nodes to begin the simulation
- **Reset** button: return all nodes to Susceptible

**Behavior:**
- Continuous-time simulation: at each frame (30fps), for each I-node, independently activate each S-neighbor with probability \( 1 - e^{-\beta \Delta t} \) and recover with probability \( 1 - e^{-\gamma \Delta t} \) (discrete-time Gillespie approximation with Δt = 0.1)
- ODE curves use Euler integration with Δt = 0.01, initialized to S(0) = N-2, I(0) = 2, R(0) = 0
- Right panel axes: x = time (0 to 200), y = fraction of population (0 to 1)
- Show vertical dashed line at current time step connecting left and right panels

**Hover interaction:** hovering a node in the left panel shows tooltip: "Node [id] | State: [S/I/R] | Degree: [k] | Time infected: [T or —]"

**Implementation notes:** hardcode both the Karate Club edge list (78 edges) and a pre-generated ER graph. Star graph: 1 hub connected to 59 leaves. All three networks have identical mean degree ≈ 4 for fair comparison.

</details>

### 5.10.1 Connection to Influence Maximization

The SIR epidemic model and the Independent Cascade Model are mathematically equivalent under a specific parameter correspondence. In an ICM with uniform edge probability \( p \), the expected cascade size from a seed set \( S \) equals the expected number of recovered nodes in an SIR epidemic initialized from \( S \) with transmission rate \( \beta = p \) and recovery rate \( \gamma = 1 \) (one-round recovery). This equivalence, established by Draief and Massoulié (2010), allows epidemic models to inform the selection of intervention strategies — vaccinating the highest-degree nodes in an SIR model corresponds to removing the most influential seed candidates in an ICM.

**Greedy influence maximization** exploits submodularity to find a provably near-optimal seed set in polynomial time. The algorithm selects seeds one at a time, at each step choosing the node that maximizes the marginal gain in expected spread:

\[
v^* = \arg\max_{v \notin S} \left[ \sigma(S \cup \{v\}) - \sigma(S) \right]
\]

Because \( \sigma \) is submodular and monotone, the greedy algorithm achieves a \( (1 - 1/e) \approx 0.632 \) approximation to the optimal \( k \)-node seed set. In practice, \( \sigma(S) \) is estimated by Monte Carlo sampling — running the ICM or LTM from \( S \) for thousands of simulations and averaging the final cascade size — which is computationally expensive but embarrassingly parallelizable.

---

## 5.11 Common Pitfalls

**1. Violating the homophily assumption silently.** Label propagation will confidently produce wrong predictions on **heterophilic graphs** (where nodes preferentially connect to nodes of different classes), because the averaging update will drive unlabeled nodes toward the opposite class from their labeled neighbors. Always check the homophily ratio \( h = \frac{|\{(u,v) \in E : y_u = y_v\}|}{|E|} \) before applying LP. If \( h < 0.3 \), LP is likely to underperform a node-feature-only classifier.

**2. Forgetting to normalize edge weights in LTM.** The constraint \( \sum_{u \in N^-(v)} w_{uv} \leq 1 \) for each node \( v \) is load-bearing: without it, the threshold \( \theta_v \sim [0,1] \) has no meaningful relationship to the incoming weights, and any node with more than one active neighbor will activate regardless of its threshold. A common bug is to use symmetric weights summing over both directions.

**3. Conflating fixed-label LP with label spreading.** Running scikit-learn's `LabelPropagation` with a graph kernel automatically uses the fixed-boundary variant. If you are working with noisy labels, use `LabelSpreading` and tune `alpha`. The two APIs share similar names but implement fundamentally different regularization strategies.

**4. Using greedy influence maximization without enough Monte Carlo samples.** The marginal gain estimates for each candidate seed are noisy because each ICM simulation is a random draw. Using too few simulations (e.g., 100 instead of 10,000) can cause the greedy algorithm to make suboptimal choices early in the seed selection, producing a final set far worse than the theoretical guarantee. The Borgs et al. (2012) SKIM algorithm and Tang et al. (2014) TIM algorithm provide theoretically motivated sample-size bounds.

**5. Applying the mean-field SIR ODE to structured networks.** The ODE system assumes homogeneous mixing: every pair of nodes contacts each other at the same rate. On a real social network with degree heterogeneity, the ODE systematically underestimates early epidemic growth (high-degree hubs infect many neighbors quickly) and overestimates the epidemic peak (the high-degree nodes recover early, leaving the epidemic to spread more slowly through the low-degree periphery). Always use the network-structured SIR simulation for quantitative predictions.

---

!!! mascot-celebration "Label Propagation Across the Chapter"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating">
    You have just completed one of the most conceptually productive chapters in the textbook. Starting from the simple observation that connected nodes tend to share labels, you derived label propagation as a harmonic function problem, extended it to the regularized label spreading framework, and connected both to the GNN message-passing paradigm via APPNP and C&S. From there you shifted to stochastic spreading: the Linear Threshold Model captures complex contagion requiring social proof, the Independent Cascade Model captures probabilistic single-exposure spread, and the SIR epidemic model bridges graph dynamics to the classical epidemiology literature. These are not unrelated algorithms — they are the same diffusion intuition expressed in different probabilistic vocabularies. The next chapter introduces Graph Neural Networks, which learn to parameterize this diffusion process end-to-end. Your neighbors have insights too!

---

## 5.12 Exercises

### Remember

1. Define **label propagation** in one sentence and state the homophily assumption it relies on. Give one real-world domain where homophily is likely to hold and one where it is likely to fail.

2. List the three components of **collective classification** and briefly describe the role each plays in the iterative inference procedure.

### Understand

3. Label propagation converges to a **harmonic function** on the graph. Explain what the harmonic property means formally (the averaging condition at unlabeled nodes) and describe its probabilistic interpretation in terms of random walks.

4. The Linear Threshold Model and the Independent Cascade Model both simulate influence spread, but they make different assumptions about how activation occurs. Compare the two models: what is the key probabilistic difference, and what type of real-world contagion is each better suited to model?

### Apply

5. Consider a path graph \( P_5 \) with nodes \( \{1, 2, 3, 4, 5\} \) and edges \( \{(1,2), (2,3), (3,4), (4,5)\} \). Nodes 1 and 5 have labels \( y_1 = 1 \) and \( y_5 = -1 \). Initialize all unlabeled nodes at \( f_v^{(0)} = 0 \). Run 3 iterations of label propagation (averaging update, labeled nodes fixed) by hand and report \( f_2^{(3)}, f_3^{(3)}, f_4^{(3)} \). What does the pattern suggest about the harmonic solution?

6. In a small network with 5 nodes and edges \( \{(A,B), (A,C), (B,D), (C,D), (D,E)\} \), suppose seed set \( S = \{A\} \) and each edge has ICM propagation probability \( p = 0.5 \). Enumerate all possible cascades and compute \( \sigma(\{A\}) \), the expected number of eventually activated nodes (including the seed).

### Analyze

7. On a graph that contains many **short triangles** (3-cycles), loopy belief propagation can give overconfident marginal estimates. Construct a minimal graph example (4–6 nodes) and trace through two rounds of the LBP message update to show how information from a single seed node double-counts evidence when it travels through a triangle. What property of the graph's topology is responsible?

8. The label propagation update \( F_U \leftarrow D_{UU}^{-1} A_{UU} F_U + D_{UU}^{-1} A_{UL} Y_L \) is equivalent to a **biased random walk** on the graph. What does this equivalence imply about the predicted label for an unlabeled node that is far (graph-distance-wise) from all labeled nodes? Under what graph conditions would label propagation produce near-uniform (uninformative) predictions for such distant nodes?

### Evaluate

9. Compare **label spreading** (with tunable \( \alpha \)) to **label propagation** (fixed boundary conditions) from the perspective of the regularization objective each minimizes. Under what graph structure — and what label quality assumption — would you strongly prefer one over the other? Support your answer with reference to the specific terms in each method's energy function.

10. Evaluate the claim: "APPNP is simply label propagation applied on top of learned node features." Is this claim accurate? Identify at least one property of APPNP that standard label propagation does not possess, and explain what practical advantage that property provides in the low-label-rate regime.

### Create

11. Design a **directed, temporally decaying** variant of the Independent Cascade Model in which the propagation probability from \( u \) to \( v \) at time \( t \) decays as \( p_{uv}(t) = p_0 \cdot e^{-\lambda (t - t_u)} \), where \( t_u \) is the time \( u \) first became active and \( \lambda > 0 \) is a decay constant. Write out the formal activation rule, explain what real-world phenomenon this models, and describe how the expected cascade size \( \sigma(S) \) changes as \( \lambda \to 0 \) and \( \lambda \to \infty \).

12. Propose a **feature-aware belief propagation** algorithm that augments the standard BP edge potential \( \psi_{uv}(X_u, X_v) \) with node feature vectors \( \mathbf{x}_u, \mathbf{x}_v \). Specifically, define a modified edge potential \( \psi_{uv}^{\text{feat}}(X_u, X_v; \mathbf{x}_u, \mathbf{x}_v) \) that uses feature similarity to weight the class-compatibility term. Show that your formulation reduces to standard BP in the case where all node feature vectors are identical, and explain what qualitative behavior you expect when two adjacent nodes have very dissimilar features.

---

## 5.13 Further Reading

1. **Zhu, X., Ghahramani, Z., & Lafferty, J. (2003). Semi-supervised learning using Gaussian fields and harmonic functions.** *Proceedings of ICML 2003.* The foundational paper deriving label propagation as a harmonic function problem on graphs. Includes the random-walk interpretation and closed-form solution that this chapter follows.

2. **Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004). Learning with local and global consistency.** *Advances in Neural Information Processing Systems (NeurIPS) 17.* Introduces label spreading with the symmetric normalized Laplacian, the regularization objective, and the α trade-off. Required reading alongside the Zhu et al. paper for a complete picture of the two complementary approaches.

3. **Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the spread of influence through a social network.** *Proceedings of KDD 2003.* The landmark paper establishing the submodularity of influence functions under both the LTM and ICM, proving the \( (1-1/e) \) greedy guarantee, and introducing the influence maximization problem as a formal optimization problem.

4. **Yedidia, J. S., Freeman, W. T., & Weiss, Y. (2001). Generalized belief propagation.** *Advances in Neural Information Processing Systems (NeurIPS) 13.* Extends belief propagation to factor graphs with higher-order interactions and provides a rigorous analysis of when and why loopy BP converges. A technically demanding but essential reference for understanding BP's approximation behavior on graphs with cycles.

5. **Gasteiger, J., Bojchevski, A., & Günnemann, S. (2019). Predict then propagate: Graph neural networks meet personalized PageRank.** *International Conference on Learning Representations (ICLR 2019).* Introduces APPNP, making the connection between label spreading and GNN propagation explicit. The separation of feature learning and propagation is a key architectural insight for the chapters ahead.

6. **Huang, Q., He, H., Singh, A., Lim, S. N., & Benson, A. R. (2021). Combining label propagation and simple models out-performs graph neural networks.** *International Conference on Learning Representations (ICLR 2021).* Demonstrates that C&S post-processing — a two-step label propagation applied after a simple MLP — achieves state-of-the-art results on ogbn-arxiv and other OGB benchmarks with far fewer parameters than deep GNNs. A compelling argument that classical diffusion ideas are far from obsolete.

7. **Pastor-Satorras, R., & Vespignani, A. (2001). Epidemic spreading in scale-free networks.** *Physical Review Letters, 86(14), 3200.* Derives the epidemic threshold \( \beta_c = \gamma \langle k \rangle / \langle k^2 \rangle \) for heterogeneous networks, showing that scale-free networks have vanishing epidemic thresholds. The result fundamentally changed how epidemiologists and network scientists think about disease control on realistic contact networks.

8. **Iscen, A., Tolias, G., Avrithis, Y., & Chum, O. (2019). Label propagation for deep semi-supervised learning.** *Proceedings of CVPR 2019.* Shows that label propagation in the embedding space of a deep network (using cosine similarity as the graph kernel) outperforms standard semi-supervised approaches on visual recognition benchmarks — a modern demonstration that label propagation remains a competitive component of deep learning pipelines when labels are scarce.
