# Quiz: Knowledge Graph Foundation Models

Test your understanding of inductive KG reasoning, ULTRA, transfer learning on graphs, and GNN pre-training.

---

#### Question 1

What is the fundamental limitation of transductive KG embedding methods (like TransE and RotatE) that motivates inductive KG reasoning?

<div class="upper-alpha" markdown>
1. They can only handle KGs with fewer than 100,000 entities
2. They cannot model antisymmetric relations, which are the most common type in real KGs
3. They store per-entity embedding vectors, so they cannot produce representations for entities not seen during training — new entities require full retraining
4. They require all entities to have the same number of relations
</div>

??? question "Show Answer"
    The correct answer is **C**. TransE, RotatE, and related methods maintain an embedding matrix \(E \in \mathbb{R}^{N \times d}\) with one row per entity. When a new entity appears after training, it has no row in \(E\) — the model cannot score any triple involving it without retraining on the expanded graph. Inductive reasoning methods instead derive entity representations from their structural context (e.g., which relations connect them to known entities), allowing generalization to completely new entities and even completely new KGs.

    **Concept Tested:** Inductive KG Reasoning

---

#### Question 2

ULTRA achieves "universal" KG reasoning by pre-training a single model that transfers to any KG without fine-tuning. What is the core architectural insight that enables this?

<div class="upper-alpha" markdown>
1. ULTRA uses very large entity embeddings (1024 dimensions) that generalize across all KG schemas
2. ULTRA pre-trains on all publicly available KGs and memorizes a lookup table of 10 million entity embeddings
3. ULTRA represents each entity by its relative structural position via relation interactions — it builds a relation graph and treats relation-to-relation interactions as the fundamental learning signal, making the model invariant to entity identity
4. ULTRA uses a language model encoder to convert entity names to embeddings, generalizing via text
</div>

??? question "Show Answer"
    The correct answer is **C**. ULTRA's key insight is to reason about *relations* rather than entities. It constructs a "relation graph" where nodes are relation types and edges encode how relations interact (e.g., "r₁ followed by r₂ gives r₃"). Entity representations are then derived from their relational context — which relations connect them to which other entities. Because the relation graph's structure is common across KGs (subject-predicate-object patterns are universal), a model trained on the relation graph of one KG transfers its inductive biases to any other KG.

    **Concept Tested:** ULTRA, Inductive KG Reasoning

---

#### Question 3

InGram learns a KG reasoning model by inducing a "relation graph" alongside the entity graph. What does the relation graph encode?

<div class="upper-alpha" markdown>
1. The schema of the KG — which entity types each relation connects
2. The co-occurrence and compositional patterns between relation types: two relations are connected if they frequently appear in paths together
3. The frequency of each relation type, used to weight edge importance during message passing
4. The inverse relation for each forward relation, enabling bidirectional traversal
</div>

??? question "Show Answer"
    The correct answer is **B**. InGram builds a relation graph where nodes are relation types and edge weights encode how often two relation types appear in the same multi-hop path (e.g., "r₁ is often followed by r₂"). GNN message passing on this relation graph produces relation embeddings that capture compositional patterns. These relation embeddings then initialize entity-level reasoning, allowing the model to generalize to new entities by using the learned relation interaction structure. This is philosophically similar to ULTRA but uses a different construction of the relation graph.

    **Concept Tested:** InGram

---

#### Question 4

Transfer learning for GNNs involves pre-training on one graph and fine-tuning on another. What is the key challenge compared to NLP transfer learning?

<div class="upper-alpha" markdown>
1. GNNs are too small to benefit from pre-training because they have fewer parameters than language models
2. Fine-tuning a pre-trained GNN always leads to catastrophic forgetting of the pre-training objective
3. GNNs cannot be pre-trained because graph datasets do not have enough diversity
4. Graphs lack a shared "token vocabulary" — different graphs have completely different entity sets, making direct weight transfer impossible without structural alignment
</div>

??? question "Show Answer"
    The correct answer is **D**. In NLP, all text shares the same vocabulary, so pre-trained token embeddings transfer directly. For graphs, a node in a molecular graph (representing an atom type) has no correspondence to a node in a citation network (representing a paper). There is no shared entity vocabulary across graph domains. ULTRA's solution is to base representations on relation structure (which is domain-independent) rather than entity identity. Pre-training still works when the structural inductive biases are shared, even when entity sets differ.

    **Concept Tested:** Transfer Learning (Graphs), Pre-Training (GNN)

---

#### Question 5

ULTRA is evaluated under "zero-shot" and "fine-tuned" settings. What does zero-shot evaluation mean in the context of KG reasoning?

<div class="upper-alpha" markdown>
1. The pre-trained ULTRA model is applied directly to a new KG without any gradient updates — no training examples from the target KG are used
2. The evaluation uses zero negative samples, testing only on positive triples
3. The model is given zero relation type information and must infer relations from entity features alone
4. The model is evaluated on triples it has seen exactly zero times during training
</div>

??? question "Show Answer"
    The correct answer is **A**. In the zero-shot setting, ULTRA's weights are frozen after pre-training on a set of source KGs. At test time, ULTRA is applied to a completely new KG — one with different entities, different relation types, and different statistics — without any fine-tuning gradient steps. The model must rely entirely on its inductive structural reasoning ability. ULTRA's strong zero-shot performance across 57 KGs demonstrates that its learned relation-interaction representation generalizes beyond the specific graphs seen during training.

    **Concept Tested:** ULTRA, Inductive KG Reasoning

---

#### Question 6

GNN pre-training strategies for molecular graphs often use two objectives: context prediction and attribute masking. What does attribute masking pre-train the GNN to do?

<div class="upper-alpha" markdown>
1. Predict whether two molecules belong to the same chemical class based on their graph structures
2. Predict the 3D coordinates of atoms given only the bond connectivity graph
3. Generate new molecular graphs by sampling from the latent space of the pre-trained encoder
4. Reconstruct masked node (atom) or edge (bond) attributes from the surrounding graph structure, learning to predict chemical properties from connectivity
</div>

??? question "Show Answer"
    The correct answer is **D**. Attribute masking (Hu et al., 2020) randomly masks node or edge attributes (e.g., atom type, bond type) and trains the GNN to predict them from the surrounding graph context. This forces the model to learn that certain substructure configurations imply specific chemical properties — for example, a carbon atom in an aromatic ring has different expected properties than one in a chain. The pre-trained encoder captures transferable chemical knowledge that improves downstream performance on molecular property prediction tasks.

    **Concept Tested:** Pre-Training (GNN), Fine-Tuning (GNN)

---

#### Question 7

Fine-tuning a pre-trained GNN on a downstream task risks "negative transfer" — performance worse than training from scratch. When does negative transfer most commonly occur?

<div class="upper-alpha" markdown>
1. When the pre-training and fine-tuning tasks are on very different graph types (e.g., pre-training on social networks and fine-tuning on molecular graphs), causing the pre-trained features to be misaligned
2. When the fine-tuning dataset has too many labeled examples, overwhelming the pre-trained prior
3. When the learning rate during fine-tuning is too low, preventing the model from adapting
4. When the pre-training graph is larger than the fine-tuning graph
</div>

??? question "Show Answer"
    The correct answer is **A**. When the pre-training distribution is very different from the target task, the pre-trained representations may actually be worse than random initialization because they encode the wrong inductive biases. For GNNs, pre-training on citation graphs (where edges encode "cites" relationships) develops features tuned to citation-network structure — these may interfere with learning on molecular graphs where the graph topology encodes chemistry. Domain-matched pre-training (molecules → molecules) is far less susceptible to negative transfer.

    **Concept Tested:** Fine-Tuning (GNN), Transfer Learning (Graphs)

---

#### Question 8

ULTRA builds entity representations from relation-graph structure rather than entity IDs. Why does this make the model invariant to KG entity labeling?

<div class="upper-alpha" markdown>
1. ULTRA hashes entity names to fixed-size vectors that are identical across KGs
2. ULTRA normalizes all entity embeddings to unit norm, making labels irrelevant
3. ULTRA uses a universal entity vocabulary shared across all KGs it was trained on
4. Because entity names are not used as input — only the pattern of which relation types connect which other relation types is used, a purely structural feature that is the same regardless of what the entities are called
</div>

??? question "Show Answer"
    The correct answer is **D**. ULTRA never uses entity name or type information — it derives entity representations purely from structural features: "this entity is connected to relation r₁ as a head and relation r₂ as a tail." This structural fingerprint is meaningful regardless of whether the entity is called "Barack Obama" or "Protein XYZ". The same reasoning rules apply: if r₁∘r₂ usually implies r₃, that pattern holds regardless of which specific entities instantiate it. This invariance is what enables zero-shot transfer to new KGs.

    **Concept Tested:** ULTRA, Inductive KG Reasoning

---

#### Question 9

The graph foundation model paradigm for KGs draws inspiration from large language model pre-training. What is the analogous "next token prediction" objective for GNN pre-training on KGs?

<div class="upper-alpha" markdown>
1. Predicting the next node to visit in a random walk starting from a given entity
2. Predicting the missing tail entity in a masked triple (h, r, ?), analogous to masked token prediction in BERT
3. Predicting the number of edges that will be added to the KG in the next update
4. Predicting the community label of each node in an unsupervised clustering task
</div>

??? question "Show Answer"
    The correct answer is **B**. KG completion — predicting the missing entity in (h, r, ?) — is the natural analogue of masked language modeling. In BERT, the model learns to predict masked tokens from context; in KG foundation models, the model learns to predict missing entities from the relational context of the known entities. This self-supervised signal requires no external labels, uses the KG's own structure as supervision, and forces the model to learn meaningful entity and relation representations.

    **Concept Tested:** Pre-Training (GNN), Inductive KG Reasoning

---

#### Question 10

What distinguishes a "fully inductive" KG reasoning setting from a "semi-inductive" one?

<div class="upper-alpha" markdown>
1. Fully inductive means all training triples are used; semi-inductive means only 50% of training triples are used
2. In the fully inductive setting, both the test entities AND the test graph are completely new (unseen during training); semi-inductive means test edges involve new entities but the graph structure is partially observed
3. Fully inductive requires no pre-training; semi-inductive requires at least 1000 pre-training triples
4. Fully inductive reasoning uses attention; semi-inductive uses sum aggregation
</div>

??? question "Show Answer"
    The correct answer is **B**. Fully inductive evaluation tests the model on a completely disjoint KG — neither the entities nor the specific triples from this KG appeared during training. The model must generalize purely from structural reasoning. Semi-inductive is an intermediate setting where some entities are seen during training but new entities appear at test time in the same graph. ULTRA targets the fully inductive setting, making it the most general and the hardest benchmark — a model achieving strong fully inductive performance has truly learned transferable structural reasoning.

    **Concept Tested:** Inductive KG Reasoning, ULTRA
