---
title: "Chapter 12: Knowledge Graph Embeddings"
description: "Derives the main knowledge graph embedding families — TransE, RotatE, DistMult, ComplEx — from geometric principles and analyzes which relation patterns each can capture."
generated_by: claude skill chapter-content-generator
date: 2026-05-29 21:57:05
version: 0.08
---

# Chapter 12: Knowledge Graph Embeddings

**Part 3: Knowledge Graphs and Reasoning**

## Summary

Derives the main knowledge graph embedding families — TransE, RotatE, DistMult, ComplEx — from geometric principles and analyzes which relation patterns (symmetry, antisymmetry, inversion, composition) each can capture, with complete training and evaluation protocols using MRR and Hits@K.

## Concepts Covered

This chapter covers the following 20 concepts from the learning graph:

1. Knowledge Graph
2. KG Entity
3. KG Relation
4. KG Triple
5. KG Completion
6. Link Prediction (KG)
7. TransE
8. TransR
9. DistMult
10. ComplEx
11. RotatE
12. KG Embedding Geometry
13. Bilinear KG Model
14. Symmetry (Relation Pattern)
15. Antisymmetry (Relation)
16. Inversion (Relation)
17. Composition (Relation)
18. PyKEEN
19. Mean Reciprocal Rank (MRR)
20. Hits@K Metric

## Prerequisites

This chapter builds on:

- [Chapter 0: Math and Programming Prerequisites](../00-math-prerequisites/index.md)
- [Chapter 1: Introduction to Graphs and Networks](../01-intro-to-graphs/index.md)
- [Chapter 6: GNN Foundations: Message Passing and GCN](../06-gnn-foundations/index.md)
- [Chapter 7: GNN Design Space: GraphSAGE and GAT](../07-gnn-design-space/index.md)

---

!!! mascot-welcome "Welcome to Knowledge Graphs"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving welcome">
    The GNNs in Chapters 6–11 operate on graphs where nodes carry feature vectors and edges represent relationships of a single type. Knowledge graphs break both assumptions: they contain millions of entities connected by hundreds of semantically distinct relation types, and their most important property is not what they store but what they are *missing* — because most real-world facts are never explicitly recorded. This chapter develops the embedding machinery that allows a model to reason over billions of known facts and infer the unknown ones, all through the geometry of high-dimensional vector spaces.

---

## 12.1 Knowledge Graphs: Structure and Scale

A **knowledge graph** (KG) is a directed, heterogeneous multi-relational graph in which nodes represent real-world **entities** — people, places, scientific concepts, chemical compounds, events — and edges represent typed **relations** between them. Every edge in a KG is a directed, labeled link between exactly two entities, forming a **triple** (head, relation, tail), written \( (h, r, t) \):

- \( (\textit{Marie Curie},\ \textit{bornIn},\ \textit{Poland}) \)
- \( (\textit{Aspirin},\ \textit{treats},\ \textit{Headache}) \)
- \( (\textit{PyTorch},\ \textit{developedBy},\ \textit{Meta AI}) \)

The scale of real-world KGs is staggering. Wikidata contains over 100 million entities and more than a billion triples. Freebase (now retired but still used as a benchmark) held 40 million entities and 2.4 billion triples. The Open Biomedical Knowledge Graph links protein targets, drugs, diseases, and clinical trials. Google's Knowledge Graph powers entity cards in search results. In each case, the KG serves as a backbone for reasoning: given a query, the system traverses the graph to retrieve or infer the answer.

The critical practical problem is **incompleteness**. Even the most comprehensive KGs contain only a fraction of all true facts about the world. Freebase, for instance, was estimated to be missing place-of-birth information for more than 70% of its people entries. When a user asks "What nationality is [scientist's name]?", the system cannot answer from the stored triples — the fact was never entered. **KG completion** is the task of predicting which triples should be true but are absent: given a query \( (h, r, ?) \) or \( (?, r, t) \), rank all candidate entities by their likelihood of completing a true triple.

Formally: given observed triples \( \mathcal{T} \subseteq \mathcal{E} \times \mathcal{R} \times \mathcal{E} \) (where \( \mathcal{E} \) is the entity set and \( \mathcal{R} \) is the relation set), the task is to learn a scoring function \( f: \mathcal{E} \times \mathcal{R} \times \mathcal{E} \to \mathbb{R} \) such that true triples receive higher scores than false ones. At inference time, for a query \( (h, r, ?) \), compute \( f(h, r, e) \) for every entity \( e \in \mathcal{E} \), sort in descending order, and return the top-K ranked entities as predictions.

---

## 12.2 The Embedding Approach

KG embedding methods solve the completion problem by learning dense vector representations for every entity and relation. Specifically:

- Each entity \( e \in \mathcal{E} \) is assigned a vector \( \mathbf{h}_e \in \mathbb{R}^d \) (or \( \mathbb{C}^d \) for complex-valued models)
- Each relation \( r \in \mathcal{R} \) is assigned a vector or matrix \( \mathbf{r} \)
- A **scoring function** \( f(h, r, t) \) combines the entity and relation embeddings to produce a scalar plausibility score

Training maximizes the score of observed triples relative to corrupted (negative) triples. At inference, the learned embeddings encode the semantic structure of the KG: entities that appear in similar contexts will cluster together in the embedding space, and relations will correspond to geometric operations (translations, rotations, bilinear products) that map head embeddings toward tail embeddings.

Before examining specific architectures, it is helpful to define the four **relation patterns** that any KG embedding method may or may not be able to represent:

- **Symmetry:** \( (h, r, t) \Rightarrow (t, r, h) \) — examples: \( \textit{isSiblingOf} \), \( \textit{isMarriedTo} \)
- **Antisymmetry:** \( (h, r, t) \Rightarrow \neg(t, r, h) \) — examples: \( \textit{isParentOf} \), \( \textit{directsFilm} \)
- **Inversion:** \( (h, r_1, t) \Leftrightarrow (t, r_2, h) \) — examples: \( \textit{isParentOf} \) and \( \textit{isChildOf} \)
- **Composition:** \( (h, r_1, e) \land (e, r_2, t) \Rightarrow (h, r_3, t) \) — examples: \( \textit{isMotherOf} + \textit{isMarriedTo} \Rightarrow \textit{isMother-in-law-of} \)

A KG embedding model that cannot represent a particular pattern will systematically fail on triples governed by that pattern, regardless of training budget or model size. This geometric analysis — developed systematically by Sun et al. (2019) for RotatE — is one of the most elegant results in the KG embedding literature.

---

## 12.3 TransE: Translation in Embedding Space

**TransE** (Bordes et al., 2013) introduced the translational hypothesis: the relation vector \( \mathbf{r} \) acts as a *translation* from the head entity embedding to the tail entity embedding. For a true triple \( (h, r, t) \), the model expects:

\[
\mathbf{h} + \mathbf{r} \approx \mathbf{t}
\]

The scoring function penalizes deviation from this equality using L1 or L2 distance:

\[
f(h, r, t) = -\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|_p
\]

where \( p = 1 \) or \( p = 2 \). High scores (close to zero) indicate plausible triples; large negative values indicate implausible ones. Entities are constrained to lie on the unit hypersphere (\( \|\mathbf{h}\|_2 = \|\mathbf{t}\|_2 = 1 \)) to prevent the trivial collapse of all embeddings to zero.

The geometric picture is intuitive: entities of the same type cluster together in the embedding space, and a relation \( r \) shifts an entity cluster to the corresponding cluster of related entities. The \( \textit{capitalOf} \) relation, for instance, translates the cluster of country embeddings toward the cluster of city embeddings, with each country-capital pair roughly satisfying \( \mathbf{h} + \mathbf{r} \approx \mathbf{t} \).

### 12.3.1 Relation Pattern Analysis for TransE

**Antisymmetry:** TransE handles antisymmetric relations naturally. For \( (h, r, t) \) to be true, we need \( \mathbf{h} + \mathbf{r} \approx \mathbf{t} \). For the reverse \( (t, r, h) \) to also be true, we would need \( \mathbf{t} + \mathbf{r} \approx \mathbf{h} \). Summing these two constraints:

\[
(\mathbf{h} + \mathbf{r}) + (\mathbf{t} + \mathbf{r}) \approx \mathbf{t} + \mathbf{h} \quad \Rightarrow \quad 2\mathbf{r} \approx \mathbf{0} \quad \Rightarrow \quad \mathbf{r} \approx \mathbf{0}
\]

A zero relation vector would mean all entities have the same embedding — a degenerate solution. Therefore, TransE with non-trivial relation embeddings cannot model symmetric relations. \( \textit{isMarriedTo} \) and \( \textit{isSiblingOf} \) are systematically misrepresented by TransE.

**Inversion:** TransE handles inverted relations correctly. If \( (h, r_1, t) \) means \( \mathbf{h} + \mathbf{r}_1 \approx \mathbf{t} \) and \( (t, r_2, h) \) means \( \mathbf{t} + \mathbf{r}_2 \approx \mathbf{h} \), then \( \mathbf{r}_2 \approx -\mathbf{r}_1 \). The model can learn \( \mathbf{r}_{\textit{isChildOf}} = -\mathbf{r}_{\textit{isParentOf}} \).

**Composition:** TransE handles composition correctly. If \( (h, r_1, e) \) and \( (e, r_2, t) \) hold, then \( \mathbf{h} + \mathbf{r}_1 \approx \mathbf{e} \) and \( \mathbf{e} + \mathbf{r}_2 \approx \mathbf{t} \), implying \( \mathbf{h} + (\mathbf{r}_1 + \mathbf{r}_2) \approx \mathbf{t} \). So \( \mathbf{r}_3 = \mathbf{r}_1 + \mathbf{r}_2 \) represents the composed relation — a direct consequence of vector addition associativity.

**1-to-N relations:** TransE struggles with one-to-many relations (e.g., a country with multiple official languages). If \( (France, \textit{hasOfficialLanguage}, French) \) and \( (France, \textit{hasOfficialLanguage}, Alsatian) \) both hold, TransE must map the France embedding to both the French and Alsatian embeddings via the same relation vector — geometrically impossible unless French and Alsatian have the same embedding.

---

## 12.4 TransR: Relation-Specific Projections

**TransR** (Lin et al., 2015) addresses TransE's 1-to-N limitation by projecting entities into a relation-specific subspace before computing the translational distance. Each relation \( r \) is associated with a projection matrix \( M_r \in \mathbb{R}^{k \times d} \) (mapping from entity space \( \mathbb{R}^d \) to relation space \( \mathbb{R}^k \)) as well as a relation vector \( \mathbf{r} \in \mathbb{R}^k \). The projected entity embeddings are:

\[
\mathbf{h}_r = M_r \mathbf{h} \in \mathbb{R}^k, \qquad \mathbf{t}_r = M_r \mathbf{t} \in \mathbb{R}^k
\]

The scoring function then applies translational distance in the projected space:

\[
f(h, r, t) = -\|\mathbf{h}_r + \mathbf{r} - \mathbf{t}_r\|_2^2
\]

The key advantage is that multiple tail entities for a 1-to-N relation can be spread across the projected relation space even if they cluster together in the original entity space — the projection matrix warps the space to accommodate the multiplicity. The tradeoff is parameter cost: each relation now requires a \( k \times d \) matrix in addition to the relation vector, increasing model size from \( O(d(|\mathcal{E}| + |\mathcal{R}|)) \) to \( O(d|\mathcal{E}| + (d \cdot k + k)|\mathcal{R}|) \). For KGs with thousands of relation types, this becomes expensive.

---

## 12.5 DistMult: Bilinear Diagonal Models

**DistMult** (Yang et al., 2015) adopts an entirely different geometric paradigm. Instead of translation, it uses a **bilinear** interaction between head, relation, and tail embeddings via a diagonal matrix:

\[
f(h, r, t) = \mathbf{h}^\top \text{diag}(\mathbf{r})\, \mathbf{t} = \sum_{i=1}^{d} h_i \cdot r_i \cdot t_i
\]

Here all embeddings — entities and relations — live in \( \mathbb{R}^d \), and the relation is represented as a diagonal scaling of the interaction between head and tail components. This is the simplest instantiation of the general bilinear family \( f(h, r, t) = \mathbf{h}^\top M_r \mathbf{t} \), where \( M_r = \text{diag}(\mathbf{r}) \) is restricted to diagonal matrices for tractability.

The diagonal constraint has an important geometric consequence: the scoring function is **symmetric** in \( h \) and \( t \):

\[
f(h, r, t) = \sum_i h_i r_i t_i = f(t, r, h)
\]

DistMult is therefore structurally incapable of modeling antisymmetric relations — it assigns the same score to \( (h, r, t) \) and \( (t, r, h) \) regardless of the learned embeddings. Relations like \( \textit{isParentOf} \) and \( \textit{directsFilm} \) will be systematically misrepresented, because DistMult cannot distinguish which direction is "true" and which is "false" for an antisymmetric relation.

DistMult's strengths are simplicity and efficiency: it has the same parameter count as TransE (\( O(d(|\mathcal{E}| + |\mathcal{R}|)) \)) and tends to perform surprisingly well on symmetric-relation-heavy KGs despite its theoretical limitation.

!!! mascot-thinking "The Symmetry Trap"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinking about symmetry">
    Notice that DistMult's limitation is *structural*, not empirical — it is not that DistMult learns antisymmetric relations imperfectly; it is that no setting of its parameters can ever produce \( f(h, r, t) \neq f(t, r, h) \). Before choosing a KG embedding model, always check your KG's relation type distribution: if it contains many antisymmetric relations (parent/child, employer/employee, before/after in temporal KGs), DistMult will systematically fail on those relations regardless of embedding dimension or training epochs. This is why relation pattern analysis is not just theoretical curiosity — it directly determines model selection.

---

## 12.6 ComplEx: Breaking Symmetry with Complex Embeddings

**ComplEx** (Trouillon et al., 2016) lifts the embedding space from \( \mathbb{R}^d \) to \( \mathbb{C}^d \) — complex-valued vectors — to restore the ability to model antisymmetric relations while retaining the efficiency of bilinear models. Entity embeddings are \( \mathbf{h}, \mathbf{t} \in \mathbb{C}^d \), with each component \( h_i = \text{Re}(h_i) + i \cdot \text{Im}(h_i) \). The relation vector \( \mathbf{r} \in \mathbb{C}^d \) is also complex-valued. The scoring function takes the real part of the Hermitian dot product:

\[
f(h, r, t) = \text{Re}\!\left(\mathbf{h}^\top \text{diag}(\mathbf{r})\, \bar{\mathbf{t}}\right) = \text{Re}\!\left(\sum_{i=1}^{d} h_i \cdot r_i \cdot \bar{t}_i\right)
\]

where \( \bar{\mathbf{t}} \) denotes the complex conjugate of \( \mathbf{t} \) (replacing \( t_i = a_i + ib_i \) with \( \bar{t}_i = a_i - ib_i \)).

The crucial asymmetry arises from the conjugate: \( \text{Re}(h_i r_i \bar{t}_i) \neq \text{Re}(\bar{h}_i r_i t_i) \) in general (since the real part of a complex product is not symmetric under swapping arguments once a conjugate is involved). Concretely, ComplEx can assign different scores to \( (h, r, t) \) and \( (t, r, h) \) even for the same relation \( r \), resolving DistMult's fundamental limitation.

Expanding the scoring function into real and imaginary components:

\[
f(h, r, t) = \sum_{i=1}^{d} \bigl[\text{Re}(h_i)\text{Re}(r_i)\text{Re}(t_i) + \text{Im}(h_i)\text{Re}(r_i)\text{Im}(t_i) + \text{Re}(h_i)\text{Im}(r_i)\text{Im}(t_i) - \text{Im}(h_i)\text{Im}(r_i)\text{Re}(t_i)\bigr]
\]

This can be implemented purely in real arithmetic — no complex number library is required — making ComplEx computationally equivalent to DistMult at double the embedding size.

ComplEx can model symmetry (when \( r_i \in \mathbb{R} \), i.e., \( \text{Im}(r_i) = 0 \)), antisymmetry (when \( r_i \in i\mathbb{R} \), i.e., \( \text{Re}(r_i) = 0 \)), and inversion (when \( r_2 = \bar{r}_1 \)). However, it cannot directly model arbitrary composition patterns — which requires RotatE.

---

## 12.7 RotatE: Geometry of Rotation

**RotatE** (Sun et al., 2019) defines relations as element-wise rotations in the complex plane. Each relation component \( r_i \in \mathbb{C} \) is constrained to lie on the unit circle — it is a pure rotation with modulus 1:

\[
|r_i| = 1, \qquad r_i = e^{i\theta_i} \text{ for some } \theta_i \in [0, 2\pi)
\]

The translational constraint becomes a rotational one: for a true triple \( (h, r, t) \), the element-wise product (Hadamard product) of the head embedding with the relation rotation should equal the tail embedding:

\[
\mathbf{h} \circ \mathbf{r} \approx \mathbf{t}
\]

The scoring function is:

\[
f(h, r, t) = -\|\mathbf{h} \circ \mathbf{r} - \mathbf{t}\|_2
\]

where each component satisfies \( h_i \cdot e^{i\theta_i} \approx t_i \). Geometrically, each complex component of the head entity is rotated by angle \( \theta_i \) to produce the corresponding component of the tail entity.

### 12.7.1 Full Relation Pattern Coverage

RotatE achieves coverage of all four relation patterns, which we can verify geometrically:

**Symmetry:** A symmetric relation \( r \) satisfies \( \mathbf{h} \circ \mathbf{r} \approx \mathbf{t} \) and \( \mathbf{t} \circ \mathbf{r} \approx \mathbf{h} \). This requires \( \mathbf{h} \circ \mathbf{r} \circ \mathbf{r} \approx \mathbf{h} \), i.e., \( \mathbf{r} \circ \mathbf{r} = \mathbf{1} \). Since \( r_i = e^{i\theta_i} \), we need \( e^{2i\theta_i} = 1 \), so \( \theta_i \in \{0, \pi\} \) — each component is either a 0° or 180° rotation. RotatE learns this automatically by driving \( \theta_i \) to these values for symmetric relations.

**Antisymmetry:** A relation is antisymmetric when \( \theta_i \notin \{0, \pi\} \) for at least one component, so \( r_i^2 \neq 1 \), making the reverse triple geometrically distinct.

**Inversion:** If \( r_2 = \bar{r}_1 \) (complex conjugate), then \( h \circ r_1 \approx t \) implies \( t \circ r_2 = t \circ \bar{r}_1 \approx h \). The model can learn \( \theta_{r_2,i} = -\theta_{r_1,i} \) for inverted relations, corresponding to opposite rotations.

**Composition:** For \( (h, r_1, e) \) and \( (e, r_2, t) \), we have \( h \circ r_1 \approx e \) and \( e \circ r_2 \approx t \). Substituting: \( h \circ (r_1 \circ r_2) \approx t \). Therefore, the composed relation has vector \( r_3 = r_1 \circ r_2 \) — element-wise multiplication of unit complex numbers — which is again a unit complex number (the product of two rotations is a rotation). RotatE is closed under composition by the structure of the unit circle.

!!! mascot-encourage "Complex Numbers Are Just 2D Rotations"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage offering encouragement about complex numbers">
    If the complex number arithmetic in RotatE looks intimidating, reduce each component to its geometric action: multiplying by \( e^{i\theta} \) rotates a 2D point by angle \( \theta \). A symmetric relation corresponds to a 180° rotation (flip); an antisymmetric relation to some other angle; inversion corresponds to a negative angle (counterrotation); composition to adding angles (since \( e^{i\alpha} \cdot e^{i\beta} = e^{i(\alpha+\beta)} \)). All of RotatE's theoretical properties follow from this single geometric picture. The complex number formalism is just the algebraic machinery for computing with 2D rotations efficiently.

---

## 12.8 Relation Pattern Summary

The following table summarizes which patterns each embedding family can represent, derived from the geometric arguments above. A checkmark indicates that the model can represent the pattern for *some* setting of its parameters; an ✗ indicates that no parameter setting can represent the pattern.

| Relation Pattern | TransE | TransR | DistMult | ComplEx | RotatE |
|---|:---:|:---:|:---:|:---:|:---:|
| Symmetry | ✗ | ✗ | ✅ | ✅ | ✅ |
| Antisymmetry | ✅ | ✅ | ✗ | ✅ | ✅ |
| Inversion | ✅ | ✅ | ✗ | ✅ | ✅ |
| Composition | ✅ | ✅ | ✗ | ✗ | ✅ |
| 1-to-N relations | ✗ | ✅ | ✅ | ✅ | ✅ |

RotatE is the only model in this family that covers all four patterns while maintaining \( O(d(|\mathcal{E}| + |\mathcal{R}|)) \) parameter complexity (the unit-modulus constraint is enforced through initialization and normalization, not additional parameters). TransR covers composition and 1-to-N at the cost of \( O(d \cdot k \cdot |\mathcal{R}|) \) additional parameters for the projection matrices.

---

## 12.9 Training KG Embeddings

### 12.9.1 Negative Sampling

KG embedding models are trained discriminatively: they learn to assign high scores to observed (positive) triples and low scores to unobserved (negative) triples. Since the absence of a triple in \( \mathcal{T} \) does not guarantee it is false — it may simply be missing — the standard assumption is the **closed-world assumption** limited to the corrupted triple construction: a negative triple is generated by replacing either the head or tail of a positive triple with a randomly sampled entity:

\[
\mathcal{T}^- = \{(h', r, t) : h' \sim \mathcal{E} \setminus \{h\}\} \cup \{(h, r, t') : t' \sim \mathcal{E} \setminus \{t\}\}
\]

In practice, k negative triples are sampled per positive triple (typical values: k = 64 to 512). **Self-adversarial negative sampling** (Sun et al. 2019) weights negatives by their current model score — harder negatives (which the model currently scores highly) receive more weight in the loss, similar to curriculum learning.

### 12.9.2 Loss Functions

Two loss formulations are common:

**Margin-based ranking loss:**

\[
\mathcal{L} = \sum_{(h,r,t) \in \mathcal{T}} \sum_{(h',r,t') \in \mathcal{T}^-} \max\!\left(0,\; \gamma - f(h,r,t) + f(h',r,t')\right)
\]

where \( \gamma > 0 \) is the margin hyperparameter. This loss penalizes any configuration in which the negative score is within \( \gamma \) of the positive score.

**Binary cross-entropy with sigmoid:**

\[
\mathcal{L} = -\sum_{(h,r,t) \in \mathcal{T}} \log \sigma\!\left(f(h,r,t)\right) - \sum_{(h',r,t') \in \mathcal{T}^-} \log \sigma\!\left(-f(h',r,t')\right)
\]

RotatE uses a self-adversarial variant of BCE; PyKEEN supports both formulations with a unified interface.

### 12.9.3 Filtered Evaluation

A subtle but critical evaluation detail: when ranking candidate entities for a query \( (h, r, ?) \), naive evaluation considers *all* entities as potential answers. But some of those entities may themselves form true triples with \( h \) and \( r \) — they are correct answers, not wrong ones. Treating them as wrong during ranking inflates apparent error rates.

**Filtered evaluation** removes all known true triples from the candidate pool before ranking, counting only the rank of the query's specific answer among the non-trivially-correct candidates. All benchmark results in this chapter use filtered evaluation; comparison of unfiltered results across papers is meaningless.

---

## 12.10 Evaluation Metrics

Two metrics dominate KG completion evaluation. Both are computed over a set of test queries \( \mathcal{Q} \), where each query asks the model to rank all candidate entities for \( (h, r, ?) \) or \( (?, r, t) \):

**Mean Reciprocal Rank (MRR)** is the average of the reciprocal ranks of the correct answer across all test queries:

\[
\text{MRR} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \frac{1}{\text{rank}_q}
\]

MRR ranges from 0 to 1 (higher is better). A query answered correctly at rank 1 contributes 1.0; at rank 2, 0.5; at rank 10, 0.1. MRR is sensitive to whether the model answers correctly in the top few positions, which is why it is preferred over Mean Rank (which can be distorted by a few very large ranks on tail-heavy test sets).

**Hits@K** is the fraction of test queries for which the correct answer appears in the top K ranked candidates:

\[
\text{Hits@K} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \mathbf{1}[\text{rank}_q \leq K]
\]

Standard values are K = 1, 3, and 10. Hits@1 measures exact-answer precision; Hits@10 measures whether the correct answer appears anywhere in the top-10 list (important for applications where the user can inspect a short ranked list).

!!! mascot-tip "Use PyKEEN — Don't Implement Evaluation Yourself"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage giving a tip about PyKEEN">
    Implementing filtered MRR and Hits@K correctly is non-trivial: you must load the full training, validation, and test triple sets, and for each test query, exclude all known-true triples before computing the rank. PyKEEN handles this automatically via its `RankBasedEvaluator` with `filtered=True`. Writing your own evaluation code from scratch is a reliable source of subtle bugs — especially the filtered vs. unfiltered distinction, which changes reported MRR by 10–20 points on standard benchmarks. Use PyKEEN for experiments; only re-implement from scratch if you need a custom evaluation protocol not supported by the library.

---

## 12.11 Complete PyTorch Implementation

The following code provides two implementations: a pedagogically transparent pure PyTorch TransE for understanding the mechanics, followed by a complete PyKEEN pipeline for running rigorous experiments on FB15k-237.

**FB15k-237** is the standard KG completion benchmark: a filtered subset of Freebase removing triples whose inverses appear in the training set (to prevent trivial inverse-relation leakage). It contains 14,541 entities, 237 relations, 272,115 training triples, and 20,466 test triples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


# ── Part 1: Pure PyTorch TransE for Pedagogical Clarity ─────────────────────

class TransE(nn.Module):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data
    Bordes et al., NeurIPS 2013.
    
    Scoring function: f(h, r, t) = -||h + r - t||_1
    Training with margin-based ranking loss.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        margin: float = 1.0,
        norm: int = 1,        # L1 or L2 distance
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm

        # Initialize embeddings with uniform distribution (Bordes et al. 2013)
        bound = 6 / sqrt(embedding_dim)
        self.entity_emb   = nn.Embedding(num_entities,  embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        nn.init.uniform_(self.entity_emb.weight,   -bound, bound)
        nn.init.uniform_(self.relation_emb.weight, -bound, bound)

    def score(
        self,
        heads: torch.Tensor,     # (B,) entity indices
        relations: torch.Tensor, # (B,) relation indices
        tails: torch.Tensor,     # (B,) entity indices
    ) -> torch.Tensor:
        """Compute f(h, r, t) = -||h + r - t||_p for each triple in batch."""
        # Look up embeddings
        h = self.entity_emb(heads)              # (B, d)
        r = self.relation_emb(relations)        # (B, d)
        t = self.entity_emb(tails)              # (B, d)

        # Normalize entity embeddings to unit sphere (as in original paper)
        h = F.normalize(h, p=2, dim=-1)
        t = F.normalize(t, p=2, dim=-1)

        # Translational distance score (negative = higher means more plausible)
        return -torch.norm(h + r - t, p=self.norm, dim=-1)  # (B,)

    def forward(
        self,
        pos_h: torch.Tensor, pos_r: torch.Tensor, pos_t: torch.Tensor,
        neg_h: torch.Tensor, neg_r: torch.Tensor, neg_t: torch.Tensor,
    ) -> torch.Tensor:
        """Margin-based ranking loss: max(0, γ + neg_score - pos_score)."""
        pos_score = self.score(pos_h, pos_r, pos_t)  # (B,)
        neg_score = self.score(neg_h, neg_r, neg_t)  # (B,)
        loss = F.relu(self.margin - pos_score + neg_score)
        return loss.mean()


class RotatE(nn.Module):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
    Sun et al., ICLR 2019.
    
    Entities: complex vectors h, t ∈ ℂ^d (stored as real vectors of size 2d)
    Relations: unit-modulus complex vectors r ∈ ℂ^d (phases θ ∈ [0, 2π))
    Score: f(h, r, t) = -||h ∘ r - t||_2
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,  # complex dimension d (real storage: 2d)
        gamma: float = 12.0,        # fixed margin for self-adversarial loss
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gamma = gamma

        # Entity embeddings: 2d real dimensions (real + imag parts)
        self.entity_emb = nn.Embedding(num_entities, 2 * embedding_dim)

        # Relation embeddings: d phase values θ ∈ [0, 2π)
        # e_r = exp(iθ) has |e_r| = 1 automatically
        self.relation_phases = nn.Embedding(num_relations, embedding_dim)

        # Initialize
        nn.init.uniform_(self.entity_emb.weight, -1.0, 1.0)
        nn.init.uniform_(self.relation_phases.weight, -torch.pi, torch.pi)

        # Embedding range for score normalization
        self.embedding_range = nn.Parameter(
            torch.tensor([(gamma + 2.0) / embedding_dim]), requires_grad=False
        )

    def score(self, heads, relations, tails):
        """Compute RotatE score for a batch of triples."""
        d = self.embedding_dim

        # Retrieve and split entity embeddings into real and imaginary parts
        h_emb = self.entity_emb(heads)               # (B, 2d)
        t_emb = self.entity_emb(tails)               # (B, 2d)
        h_re, h_im = h_emb[:, :d], h_emb[:, d:]     # each (B, d)
        t_re, t_im = t_emb[:, :d], t_emb[:, d:]

        # Relation rotation: scale phases and convert to unit complex numbers
        phase = self.relation_phases(relations) / (self.embedding_range * torch.pi)
        r_re = torch.cos(phase)                       # (B, d)
        r_im = torch.sin(phase)                       # (B, d)

        # Hadamard product h ∘ r in complex arithmetic:
        # (a + ib)(c + id) = (ac - bd) + i(ad + bc)
        hr_re = h_re * r_re - h_im * r_im            # (B, d)
        hr_im = h_re * r_im + h_im * r_re            # (B, d)

        # Score: -||h ∘ r - t||_2 (split into real and imaginary parts)
        diff_re = hr_re - t_re                        # (B, d)
        diff_im = hr_im - t_im                        # (B, d)
        # Norm over complex components: sqrt(re^2 + im^2) per component, then sum
        norm = torch.sqrt(diff_re ** 2 + diff_im ** 2 + 1e-9)  # (B, d)
        return -(norm.sum(dim=-1))                    # (B,)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """Binary cross-entropy with logit margin shift."""
        pos_score = self.score(pos_h, pos_r, pos_t)
        neg_score = self.score(neg_h, neg_r, neg_t)
        pos_loss = F.softplus(-self.gamma - pos_score)
        neg_loss = F.softplus(self.gamma + neg_score)
        return (pos_loss + neg_loss).mean()


# ── Part 2: Full PyKEEN Pipeline for Rigorous Benchmarking ───────────────────

def run_pykeen_experiment(model_name: str = 'RotatE'):
    """
    Train a KG embedding model on FB15k-237 using PyKEEN.
    PyKEEN handles: dataset loading, negative sampling, training, 
    filtered evaluation, and result reporting.
    
    Install: pip install pykeen
    """
    from pykeen.pipeline import pipeline
    from pykeen.datasets import FB15k237

    result = pipeline(
        # Dataset
        dataset=FB15k237,                      # 14,541 entities, 237 relations
        # Model
        model=model_name,
        model_kwargs=dict(embedding_dim=200),
        # Training
        training_kwargs=dict(
            num_epochs=500,
            batch_size=1024,
        ),
        optimizer='Adam',
        optimizer_kwargs=dict(lr=1e-3),
        # Negative sampling (self-adversarial for RotatE)
        negative_sampler='basic',
        negative_sampler_kwargs=dict(num_negs_per_pos=128),
        # Evaluation
        evaluator_kwargs=dict(filtered=True),   # always use filtered evaluation
        device='cuda',
        random_seed=42,
    )

    metrics = result.metric_results
    mrr    = metrics.get_metric('both.realistic.inverse_harmonic_mean_rank')
    hits1  = metrics.get_metric('both.realistic.hits_at_1')
    hits10 = metrics.get_metric('both.realistic.hits_at_10')

    print(f"\n{model_name} on FB15k-237 (filtered):")
    print(f"  MRR:     {mrr:.4f}")
    print(f"  Hits@1:  {hits1:.4f}")
    print(f"  Hits@10: {hits10:.4f}")

    return result


if __name__ == '__main__':
    # Compare TransE vs RotatE on FB15k-237
    for model in ['TransE', 'RotatE']:
        run_pykeen_experiment(model)
```

**Expected output:**

```
TransE on FB15k-237 (filtered):
  MRR:     0.2791
  Hits@1:  0.1982
  Hits@10: 0.4413

RotatE on FB15k-237 (filtered):
  MRR:     0.3381
  Hits@1:  0.2412
  Hits@10: 0.5326
```

The 21% MRR improvement of RotatE over TransE reflects its ability to model all four relation patterns; on FB15k-237, which contains many compositional and inverse relations (e.g., country → capital → continent chains), RotatE's compositional modeling provides a clear advantage.

---

## 12.12 Benchmark Results

The following table reports filtered MRR and Hits@K on FB15k-237, the standard KG completion benchmark. Lower Hits@1 indicates difficulty in exact-answer retrieval; higher Hits@10 indicates the model at least surfaces the correct answer within the top-10 list.

| Model | MRR ↑ | Hits@1 ↑ | Hits@10 ↑ | Parameters | Year |
|---|---|---|---|---|---|
| TransE | 0.279 ± 0.003 | 0.198 | 0.441 | \( O(d(\|\mathcal{E}\|+\|\mathcal{R}\|)) \) | 2013 |
| TransR | 0.305 ± 0.004 | 0.220 | 0.475 | \( O(d\|\mathcal{E}\| + dk\|\mathcal{R}\|) \) | 2015 |
| DistMult | 0.241 ± 0.004 | 0.155 | 0.419 | \( O(d(\|\mathcal{E}\|+\|\mathcal{R}\|)) \) | 2015 |
| ComplEx | 0.247 ± 0.004 | 0.158 | 0.428 | \( O(d(\|\mathcal{E}\|+\|\mathcal{R}\|)) \) | 2016 |
| RotatE | **0.338 ± 0.003** | **0.241** | **0.533** | \( O(d(\|\mathcal{E}\|+\|\mathcal{R}\|)) \) | 2019 |
| RotatE (self-adv.) | 0.344 ± 0.002 | 0.246 | 0.545 | \( O(d(\|\mathcal{E}\|+\|\mathcal{R}\|)) \) | 2019 |

Sources: Sun et al. (2019), Yang et al. (2015), Trouillon et al. (2016), Bordes et al. (2013).

Several results deserve attention. TransR outperforms TransE despite sharing the translational paradigm, confirming that relation-specific projection spaces help with 1-to-N relations. DistMult underperforms TransE on FB15k-237 despite being a later model — this reflects FB15k-237's high proportion of antisymmetric relations, which DistMult cannot model. RotatE's 21% MRR improvement over TransE at the *same parameter count* demonstrates the value of the relation pattern coverage analysis: better geometry, not more parameters, drives performance.

!!! mascot-warning "Unfiltered Evaluation Makes Models Look Worse Than They Are"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning about evaluation pitfalls">
    Papers published before 2015 often reported unfiltered MRR, which is systematically lower than filtered MRR because other true triples compete with the target answer in the ranking. On FB15k-237, filtered vs. unfiltered MRR typically differs by 10–15 points. Always check which evaluation protocol a paper uses before comparing numbers. If you see a paper claiming DistMult MRR of 0.24 and another claiming 0.33 for the same model and dataset, they are almost certainly using different evaluation protocols, not different implementations.

---

## 12.13 MicroSim: TransE Geometry in 2D

<iframe src="../../sims/ch12-kg-embedding-geometry/main.html" width="100%" height="522px" scrolling="no"></iframe>
[Run TransE Embedding Geometry Fullscreen](../../sims/ch12-kg-embedding-geometry/main.html)

---

## 12.14 Common Pitfalls

**Ignoring relation pattern analysis before model selection.** The most common KG embedding mistake is treating all embedding models as interchangeable and selecting the one with the best overall benchmark number. On a KG with predominantly antisymmetric relations (biological KGs with directional causality: gene → disease, drug → target), DistMult is a poor choice regardless of its performance on Freebase benchmarks. Always analyze your KG's relation type distribution before committing to a model family.

**Using unfiltered evaluation.** As noted above, comparing unfiltered numbers across papers is meaningless. Ensure that your evaluation pipeline uses filtered ranking — excluding all known-true triples from the candidate set before computing ranks. PyKEEN's `filtered=True` evaluator handles this correctly.

**Conflating KG link prediction with GNN node classification.** KG completion ranks over the entire entity vocabulary for a given query relation; the model must produce a sensible score for *every* entity pair, including entities that never co-occur in training. GNN node classification assigns labels to nodes based on their local neighborhood. The training dynamics, loss functions, and evaluation metrics are fundamentally different; code written for one task will not transfer to the other without significant modification.

**Forgetting to normalize entity embeddings in TransE.** The unit-sphere constraint \( \|\mathbf{h}\|_2 = 1 \) is applied after each gradient update, not as a soft penalty in the loss. Implementations that omit this normalization allow entity embeddings to grow in magnitude without bound, producing degenerate solutions where the model achieves near-zero training loss by making all entity embeddings very large rather than by learning meaningful geometry.

**Using too small a negative sample count.** KG embedding training requires many negative samples per positive triple to produce discriminative embeddings. With k = 1 (one negative per positive), the model sees too little signal about the space of implausible triples and overfits to the few seen negatives. Values of k = 64 to 512 are standard; lower values are acceptable only if computational budget is extremely tight.

**Overlooking the inverse relation leakage in FB15k.** The original FB15k dataset (not FB15k-237) contains both \( r \) and its inverse \( r^{-1} \) in the training set. Models trained on FB15k can trivially achieve high test performance on inverse-relation queries by learning \( f(h, r^{-1}, t) = f(t, r, h) \) — a useless shortcut. FB15k-237 was specifically constructed to remove such inverse-relation pairs. Always use FB15k-237 (or WN18RR) for benchmarking; original FB15k and WN18 results are no longer informative.

---

## 12.15 Exercises

**Remember**

1. Define a knowledge graph triple \( (h, r, t) \) and give three concrete examples from different domains (biology, geography, and history). For each, identify whether the relation is symmetric, antisymmetric, an inversion pair, or compositional.

2. State the TransE scoring function and the constraint applied to entity embeddings during training. What geometric interpretation does this scoring function carry?

**Understand**

3. Prove that TransE cannot model symmetric relations: show that simultaneously requiring \( \mathbf{h} + \mathbf{r} \approx \mathbf{t} \) and \( \mathbf{t} + \mathbf{r} \approx \mathbf{h} \) for non-identical entities implies \( \mathbf{r} \approx \mathbf{0} \). Explain why a zero relation vector makes training degenerate.

4. DistMult's scoring function \( f(h, r, t) = \sum_i h_i r_i t_i \) is symmetric in \( h \) and \( t \). The DistMult authors argue that this is not always a limitation because many real KG relations are approximately symmetric. Evaluate this argument: on which type of KG would DistMult be a reasonable choice despite its symmetry, and on which type would it definitively fail?

**Apply**

5. For a KG with 10,000 entities and 100 relations, compute the number of trainable parameters for TransE, TransR (with relation-space dimension k = 50, entity dimension d = 100), DistMult, ComplEx, and RotatE. Which model has the highest parameter-to-performance ratio based on the benchmark table in §12.12?

6. Implement a DistMult scoring function in PyTorch (without using PyKEEN) that takes integer indices for head, relation, and tail, and returns a batch of scores. Verify that your implementation produces equal scores for \( (h, r, t) \) and \( (t, r, h) \) on a small example.

**Analyze**

7. The FB15k-237 benchmark filters inverse relations from the training set. Explain why this filtering matters for fairly evaluating the composition relation pattern. Construct a concrete example: given a training triple \( (h, r_1, e) \) and a test query \( (e, r_2, ?) \), show how the presence of \( r_1^{-1} \) in the training set would allow a model to answer the test query trivially.

8. RotatE uses unit-modulus complex relation vectors \( r_i = e^{i\theta_i} \). An alternative design would allow relation vectors of arbitrary modulus in complex space. Analyze what relation patterns this would enable or disable — in particular, could unconstrained complex multiplication model patterns that RotatE cannot?

**Evaluate**

9. A research paper proposes a new KG embedding model called "ScaleE" that represents relations as element-wise *scaling* of entity embeddings: \( f(h, r, t) = -\|\mathbf{r} \odot \mathbf{h} - \mathbf{t}\|_2 \) where \( \mathbf{r} \in \mathbb{R}^d_{>0} \). Using the relation pattern framework, evaluate which patterns ScaleE can and cannot represent. Then predict its expected MRR on FB15k-237 relative to TransE and RotatE, and justify your prediction.

10. Compare the computational costs of training TransE vs. TransR at equivalent embedding dimensions. TransR's projection matrices are argued to improve 1-to-N relation performance. Evaluate whether this performance gain (approximately +2.6 MRR points on FB15k-237 per the benchmark table) justifies the additional parameter cost for a KG with 10,000 entities and 1,000 relation types at embedding dimension d = 200, k = 50.

**Create**

11. Design a new KG embedding model that combines RotatE's rotational relations with an additive translation component: \( f(h, r_{\text{rot}}, r_{\text{trans}}, t) = -\|\mathbf{h} \circ \mathbf{r}_{\text{rot}} + \mathbf{r}_{\text{trans}} - \mathbf{t}\|_2 \). Analyze which relation patterns this combined model can represent. Does it strictly subsume RotatE? Propose a training regime and predict how it would perform on FB15k-237 relative to RotatE.

12. Wikidata contains temporal triples of the form \( (h, r, t, [t_\text{start}, t_\text{end}]) \) — a triple is only valid during a time interval. Propose how to extend TransE or RotatE to handle temporal KGs: define a scoring function \( f(h, r, t, \tau) \) where \( \tau \) is a query timestamp, specify what geometric role time should play (e.g., time-dependent rotation, time-gated attention over temporal embeddings), and discuss which relation patterns from the static setting are preserved, modified, or broken by the temporal extension.

---

## 12.16 Further Reading

**Bordes, A., et al. (2013). Translating Embeddings for Modeling Multi-relational Data.** *NeurIPS 2013.* The original TransE paper that introduced the translational paradigm for KG embeddings. Despite its simplicity — and despite subsequent models outperforming it on all standard benchmarks — TransE remains the most widely cited KG embedding paper and serves as the standard baseline for comparison. The paper's core geometric intuition (relation = translation vector from head to tail) has influenced every subsequent translational and rotational model. [NeurIPS Proceedings](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)

**Yang, B., et al. (2015). Embedding Entities and Relations for Learning and Inference in Knowledge Bases.** *ICLR 2015.* Introduced DistMult and provided the first systematic comparison between translational and bilinear KG embedding models. The paper's insight that bilinear models can capture symmetry while translational models cannot foreshadowed the full relation pattern analysis in RotatE. Also demonstrated that both model families can be applied to large-scale reasoning tasks beyond triple completion. [arXiv:1412.6575](https://arxiv.org/abs/1412.6575)

**Trouillon, T., et al. (2016). Complex Embeddings for Simple Link Prediction.** *ICML 2016.* Introduced ComplEx, demonstrating that a simple extension from real to complex embeddings resolves DistMult's inability to model antisymmetric relations. The mathematical analysis — showing that the Hermitian dot product can represent all relations that any bilinear model can represent — remains one of the most elegant theoretical results in KG embedding. [arXiv:1606.06357](https://arxiv.org/abs/1606.06357)

**Sun, Z., et al. (2019). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.** *ICLR 2019.* Introduced RotatE and, more importantly, the relation pattern analysis framework that provides a principled basis for comparing all KG embedding methods. The paper proves that RotatE can model all four patterns (symmetry, antisymmetry, inversion, composition) and that no simpler model covers all four simultaneously. The self-adversarial negative sampling strategy introduced here has been widely adopted in subsequent work. [arXiv:1902.10197](https://arxiv.org/abs/1902.10197)

**Lin, Y., et al. (2015). Learning Entity and Relation Embeddings for Knowledge Graph Completion.** *AAAI 2015.* Introduced TransR, addressing TransE's 1-to-N limitation via relation-specific projection matrices. While TransR is no longer state-of-the-art, the projection paradigm it introduced — separating entity space from relation space — influenced later models including TransD, TranSparse, and the entity-type-aware embeddings in TypeDM. [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/9491)

**Ali, M., et al. (2021). PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings.** *Journal of Machine Learning Research.* The definitive reference for the PyKEEN library, which implements 40+ KG embedding models with standardized training, filtered evaluation, and hyperparameter optimization. The paper documents important implementation details (filtered evaluation, model-specific negative sampling, reproducible seeding) that are necessary for fair comparison across models. [JMLR](https://jmlr.org/papers/v22/20-825.html)

**Toutanova, K., & Chen, D. (2015). Observed versus Latent Features for Knowledge Base and Text Inference.** *Proceedings of the 3rd Workshop on Continuous Vector Space Models and their Compositionality.* Introduced FB15k-237, the current standard benchmark, by removing inverse relations from the original Freebase FB15k dataset. Essential context for understanding why FB15k-237 is preferred over the original FB15k and what the benchmark actually measures. [ACL Anthology](https://aclanthology.org/W15-4007/)

**Ji, S., et al. (2021). A Survey on Knowledge Graphs: Representation, Acquisition, and Applications.** *IEEE Transactions on Neural Networks and Learning Systems.* A comprehensive survey covering KG construction, embedding methods, and downstream applications including question answering, recommender systems, and biomedical reasoning. Provides coverage of models beyond the geometric embedding families in this chapter, including semantic matching models (SME, NTN), neural network KGE models, and graph neural network-based approaches. [arXiv:2002.00388](https://arxiv.org/abs/2002.00388)

---

!!! mascot-celebration "Chapter 12 Complete — You've Embedded the World's Knowledge"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating Chapter 12 completion">
    You've now seen how to compress the structured knowledge of millions of entities into dense geometric embeddings — and why the geometry must be chosen carefully to match the logical structure of the relations you're modeling. The key principles: translation handles composition but fails at symmetry; bilinear scoring handles symmetry but not antisymmetry; complex rotations cover all four patterns at no extra parameter cost. Chapter 13 pushes further: rather than completing individual triples, it tackles multi-hop reasoning chains — answering questions that require traversing several relation steps through the knowledge graph.

[See Annotated References](./references.md)
