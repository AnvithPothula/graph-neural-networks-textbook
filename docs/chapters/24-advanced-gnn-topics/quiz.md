# Quiz: Advanced GNN Topics: In-Context Learning and Uncertainty

Test your understanding of self-supervised GNN learning, contrastive methods, in-context learning, and uncertainty quantification.

---

#### Question 1

Graph Contrastive Learning (GraphCL) creates two views of a graph via augmentation and maximizes agreement between their representations. Why must the two views be semantically consistent?

<div class="upper-alpha" markdown>
1. Inconsistent views prevent the model from computing gradients through the contrastive loss
2. Inconsistent views cause overfitting because the model memorizes both augmented versions of each graph
3. If augmentations destroy the label-relevant structure (e.g., removing functional groups in molecular graphs changes the property being predicted), the model learns to produce similar representations for graphs with different semantics — degrading downstream task performance
4. Semantic consistency is only required for node-level tasks, not graph-level tasks
</div>

??? question "Show Answer"
    The correct answer is **C**. Contrastive learning trains the encoder to produce similar representations for positive pairs (two augmentations of the same graph) and dissimilar representations for negative pairs (augmentations of different graphs). If an augmentation removes the substructure responsible for the label (e.g., removing the ring system in a drug molecule changes its pharmacological class), positive pairs no longer share the semantically relevant structure — the model learns a representation that ignores the label-relevant features. Domain knowledge is required to choose augmentations that preserve the label signal.

    **Concept Tested:** Graph Contrastive Learning, Graph Augmentation (SSL)

---

#### Question 2

The InfoNCE (NT-Xent) contrastive loss for graph learning has the form \(\mathcal{L} = -\log\!\left[\exp(\text{sim}(z, z^+)/\tau) / \sum_j \exp(\text{sim}(z, z_j)/\tau)\right]\). What role does the temperature \(\tau\) play?

<div class="upper-alpha" markdown>
1. \(\tau\) controls the sharpness of the similarity distribution: low \(\tau\) makes the loss focus intensely on hard negatives (highest-similarity wrong pairs); high \(\tau\) spreads learning signal across all negatives more uniformly
2. τ controls the number of negative samples to use in each mini-batch
3. \(\tau\) is the learning rate for the contrastive training objective
4. \(\tau\) normalizes the embeddings to unit length before computing cosine similarity
</div>

??? question "Show Answer"
    The correct answer is **A**. Temperature \(\tau\) in the softmax scales the logits. When \(\tau \to 0\), the softmax approaches a hard argmax — only the hardest negative (highest similarity) contributes gradient. When \(\tau \to \infty\), the softmax becomes uniform — all negatives contribute equally. In practice, small \(\tau\) (0.05–0.2) is used to concentrate learning on hard negatives, which forces the model to make finer discriminations. Too-small \(\tau\) causes instability (gradients dominated by near-duplicate negatives); too-large \(\tau\) provides weak gradients. \(\tau\) is typically tuned as a hyperparameter.

    **Concept Tested:** Contrastive Loss

---

#### Question 3

Deep Graph Infomax (DGI) maximizes mutual information between patch (node) representations and a global graph summary. Why is the global summary computed using a readout function rather than another GNN?

<div class="upper-alpha" markdown>
1. A second GNN would double the training time without improving the quality of the global summary
2. Using a GNN for the global summary would create circular dependencies in the computation graph
3. The global summary must be a fixed-dimensional permutation-invariant representation of the whole graph — a simple readout (mean/sum of node representations) achieves this without additional trainable parameters, keeping the pre-training objective tractable
4. Readout functions produce more informative summaries than GNNs for graph-level representations
</div>

??? question "Show Answer"
    The correct answer is **C**. DGI's training signal requires a global summary \(s = R(\{h_v\})\) that represents the whole graph while remaining a fixed-dimensional vector comparable to individual node representations. A simple mean or sum readout achieves this in \(O(N)\) operations without adding parameters. Using a second GNN for the summary would add complexity (more parameters, potentially conflicting gradients) for no clear benefit. The key is that the summary must capture some overall graph structure that is shared across all node representations — a mean pooling accomplishes this effectively.

    **Concept Tested:** Deep Graph Infomax (DGI)

---

#### Question 4

PRODIGY enables in-context learning (ICL) for GNNs: the model answers a query about a new graph by conditioning on a few labeled examples (support set) provided in context. How does this differ from few-shot fine-tuning?

<div class="upper-alpha" markdown>
1. ICL updates the GNN weights on the support set before making predictions; fine-tuning does not
2. ICL requires more labeled examples than fine-tuning for the same performance level
3. ICL makes predictions at inference time using only a forward pass conditioned on the support set — no gradient updates; fine-tuning updates model parameters on labeled examples before deployment
4. ICL can only be applied to node-level tasks; fine-tuning works for all task types
</div>

??? question "Show Answer"
    The correct answer is **C**. In-context learning borrows from language model ICL: at inference time, the support set (labeled examples) is provided as context and the model makes predictions in a single forward pass without any parameter updates. Fine-tuning instead performs gradient descent on the support set, updating weights before the query is answered. ICL is faster (no optimization loop) and avoids overfitting to tiny support sets — the model relies on pre-trained knowledge to generalize from the few examples, treating the support set as contextual information rather than training data.

    **Concept Tested:** In-Context Learning (Graphs)

---

#### Question 5

Conformal prediction is used to produce uncertainty-aware predictions for GNNs. What guarantee does a conformalized GNN provide?

<div class="upper-alpha" markdown>
1. The prediction set \(C(x)\) — a subset of possible labels — contains the true label with at least \((1-\alpha)\) probability, regardless of the GNN's architecture or training procedure, under exchangeability
2. The model's predicted class probabilities are calibrated (correctly reflect confidence percentages)
3. The model's loss on the test set is bounded by the calibration set loss plus ε
4. The model cannot make errors on test examples that are similar to calibration examples
</div>

??? question "Show Answer"
    The correct answer is **A**. Conformal prediction provides a distribution-free coverage guarantee: given a calibration set, the prediction set \(C(x)\) is constructed such that \(P(y \in C(x)) \geq 1 - \alpha\) for a user-specified error rate \(\alpha\). This holds under the exchangeability assumption (calibration and test data are exchangeable) without assuming the GNN is well-calibrated or making any distributional assumptions. For a GNN, this means: compute nonconformity scores on the calibration set, set a threshold at the \((1-\alpha)\) quantile, then include all labels whose nonconformity score is below the threshold in \(C(x)\).

    **Concept Tested:** Conformalized GNN

---

#### Question 6

Graph augmentation for self-supervised learning on molecular graphs requires domain-aware strategies. Why is edge dropping (randomly removing bonds) less semantically safe than node feature masking?

<div class="upper-alpha" markdown>
1. Edge dropping changes the graph's adjacency structure, requiring more expensive recomputation
2. Removing a bond changes the molecular structure (breaking a ring, disconnecting atoms) and can fundamentally alter the molecule's chemical properties; masking an atom's feature leaves the connectivity intact and creates a prediction task that resembles standard chemistry
3. Edge dropping is computationally more expensive than feature masking
4. Edge dropping only works for molecules with more than 10 atoms, while feature masking works for all sizes
</div>

??? question "Show Answer"
    The correct answer is **B**. Dropping bonds can disconnect the molecular graph or open rings — changes that dramatically alter chemical properties (a benzene ring is aromatic; break one bond and the ring opens, losing aromaticity). These structural changes may create chemically impossible or semantically different "views" of the molecule. Node feature masking leaves the bond topology intact and only hides the atom's type/charge, creating a fill-in-the-blank task analogous to BERT's masked language modeling — the model must predict the masked atom type from the surrounding molecular context.

    **Concept Tested:** Graph Augmentation (SSL), Graph Contrastive Learning

---

#### Question 7

The One-For-All (OFA) model uses a unified prompt-based framework for different graph tasks. How does it handle the different prediction heads required for node classification vs. link prediction?

<div class="upper-alpha" markdown>
1. OFA uses separate model weights for each task type, selected via a routing mechanism
2. OFA requires fine-tuning all parameters for each new task type, losing the foundation model advantage
3. OFA adds a task-specific linear layer that is the only component trained for each downstream task
4. OFA reformulates all tasks as subgraph classification: node classification becomes "classify the ego network of this node"; link prediction becomes "classify whether this 2-node subgraph has a connecting edge" — enabling a single graph classifier to handle all tasks
</div>

??? question "Show Answer"
    The correct answer is **D**. OFA's key insight is that node classification, link prediction, and graph classification can all be reformulated as "does this subgraph have property X?" Node classification: classify the ego subgraph centered on the target node. Link prediction: classify a 2-hop subgraph containing both endpoint nodes. Graph classification: classify the full graph. By reducing all tasks to a unified subgraph classification problem, OFA can use a single model with a single classification head, enabling true cross-task transfer.

    **Concept Tested:** One-For-All (OFA) Model

---

#### Question 8

Contrastive learning on graphs requires careful negative sampling. Why are "false negatives" a problem in graph contrastive learning?

<div class="upper-alpha" markdown>
1. False negatives cause the contrastive loss to become negative, preventing convergence
2. False negatives cannot occur in graph contrastive learning because the negative sampling is always random
3. False negatives only occur in molecular graphs where augmentations preserve the molecular formula
4. A false negative is a node or graph in the "negative" sample that is actually semantically similar to the anchor — treating it as a negative pushes away representations that should be close, corrupting the learned embedding space
</div>

??? question "Show Answer"
    The correct answer is **D**. In contrastive learning, negatives are typically random samples from the mini-batch. But some "random" negatives may be semantically identical to the anchor — e.g., two papers from the same research area that happen to be in the same batch. If both are treated as negatives, the model is penalized for producing similar representations for genuinely similar graphs. This corrupts the representation space. Solutions include supervised contrastive learning (using labels to filter negatives), hard negative mining with a similarity threshold, and queue-based large-batch approaches.

    **Concept Tested:** Contrastive Loss, Graph Contrastive Learning

---

#### Question 9

Conformal prediction for GNNs on non-i.i.d. graph data faces a specific challenge: graph nodes are not exchangeable because they are connected. How does conformal prediction for transductive node classification address this?

<div class="upper-alpha" markdown>
1. It applies conformal prediction independently to each connected component
2. It uses a modified transductive conformal predictor where calibration scores are computed on masked nodes within the same graph, exploiting the inductive structure of the GNN to maintain approximate exchangeability
3. Conformal prediction is not applicable to graph-structured data — a simulation-based alternative is used instead
4. It removes all edges during calibration to make nodes i.i.d., then adds edges back for inference
</div>

??? question "Show Answer"
    The correct answer is **B**. Standard conformal prediction requires exchangeability (calibration and test points are interchangeable). Graph nodes are not exchangeable — removing a node changes the representations of neighboring nodes. For transductive node classification, conformal prediction can still be applied by treating the GNN's output scores on labeled calibration nodes as nonconformity scores, computing the \((1-\alpha)\) quantile, and applying the same threshold to test nodes. The exchangeability assumption holds approximately when the GNN's aggregation smooths out individual node dependencies.

    **Concept Tested:** Conformalized GNN

---

#### Question 10

PRODIGY (in-context learning for graphs) builds a "prompt graph" that combines the query graph with support examples. What is the architectural mechanism that allows PRODIGY to condition predictions on the support set?

<div class="upper-alpha" markdown>
1. PRODIGY constructs a super-graph connecting the query node to support nodes via virtual edges, then runs GNN message passing — the query node aggregates labeled support node representations as part of the standard forward pass
2. PRODIGY fine-tunes the GNN weights on the support set before processing the query
3. PRODIGY uses cross-attention between the query embedding and support embeddings in a separate attention module
4. PRODIGY concatenates support embeddings to the query embedding as additional node features before classification
</div>

??? question "Show Answer"
    The correct answer is **A**. PRODIGY creates a "prompt graph" that contains the original query graph plus the support examples, connected by virtual edges from the query nodes to their nearest support nodes. Standard GNN message passing then allows the query node to aggregate information from labeled support nodes — effectively "reading" the labels from context as part of the forward pass. No separate attention module or fine-tuning is required: the support examples act as labeled "anchor" nodes whose representations propagate to the query node through the virtual edges.

    **Concept Tested:** In-Context Learning (Graphs)
