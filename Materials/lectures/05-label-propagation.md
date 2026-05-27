# Lecture 5: Label Propagation & Semi-Supervised Node Classification

## Overview

Given a partially labeled graph, how do we classify unlabeled nodes? Label propagation and belief propagation are classical semi-supervised methods that spread information through the graph structure.

**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/05-message.pdf

---

## Key Concepts

### Problem Setup
- **Semi-supervised node classification:** some nodes have labels Y_L, others are unlabeled Y_U
- Goal: predict labels for unlabeled nodes using graph structure + labeled nodes
- **Assumption:** connected nodes tend to have the same label (homophily)

### Label Propagation Algorithm
1. Initialize: Y_i = observed label for labeled nodes; Y_i = 0.5 (or uniform) for unlabeled
2. For each iteration, for each unlabeled node v:
   - Y_v = (1/|N(v)|) Σ_{u ∈ N(v)} Y_u (average neighbor labels)
3. Keep labeled nodes fixed (or use a combination)
4. Repeat until convergence

### Belief Propagation
- More general probabilistic framework (factor graphs)
- Each node sends a "message" to its neighbors: estimates of their variable states
- Exact on trees; approximate on graphs with cycles (loopy belief propagation)
- Useful for: fraud detection, spam detection, community detection

### Collective Classification
Three interconnected components:
1. **Local classifier:** initial label assignment based on node attributes
2. **Relational classifier:** label assignment based on neighbor labels
3. **Collective inference:** iterate the above simultaneously

### Modern Connection: GNNs as Learnable Label Propagation
- GNNs generalize label propagation by learning the aggregation function
- APPNP separates prediction from propagation
- C&S (Correct and Smooth) is a modern label propagation post-processing step for GNNs

## Textbook Chapter Notes

Maps to **Chapter 5: Label Propagation & Semi-Supervised Learning**.
- Code: label propagation with scikit-learn (`LabelPropagation`, `LabelSpreading`)
- MicroSim: visualize labels propagating through a graph step by step
