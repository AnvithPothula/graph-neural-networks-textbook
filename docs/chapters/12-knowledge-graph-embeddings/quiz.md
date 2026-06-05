# Quiz: Knowledge Graph Embeddings

Test your understanding of KG embedding models, geometric intuitions, relation patterns, and evaluation metrics.

---

#### Question 1

TransE models a relation \(r\) between entities \(h\) and \(t\) with the scoring function \(f(h, r, t) = -\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|\). What geometric interpretation does this encode?

<div class="upper-alpha" markdown>
1. Entity embeddings are matrices and relations are matrix multiplications
2. The score is the dot product of \(\mathbf{h}\) and \(\mathbf{t}\), with \(\mathbf{r}\) as a learnable bias term
3. The relation \(r\) is a rotation in the complex plane applied to \(h\)
4. The relation vector \(\mathbf{r}\) is a translation: a valid triple \((h, r, t)\) should satisfy \(\mathbf{h} + \mathbf{r} \approx \mathbf{t}\) in the embedding space
</div>

??? question "Show Answer"
    The correct answer is **D**. TransE interprets each relation as a translation in the embedding space: if \((h, r, t)\) is a valid triple, then the embedding of head entity \(h\) plus the relation vector \(\mathbf{r}\) should be close to the embedding of tail entity \(t\). Training pushes \(\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|\) toward 0 for true triples and toward a margin \(\gamma\) for corrupted triples. This geometric simplicity makes TransE fast and effective for antisymmetric relations but unable to model symmetric or one-to-many relations.

    **Concept Tested:** TransE, KG Embedding Geometry

---

#### Question 2

RotatE represents each relation as an element-wise rotation in the complex plane. How does this allow RotatE to model symmetric relations, which TransE cannot?

<div class="upper-alpha" markdown>
1. A rotation by 0 or \(\pi\) satisfies \(\mathbf{r} \circ (\mathbf{r} \circ \mathbf{h}) = \mathbf{h}\), enabling RotatE to learn symmetric relations as 180° rotations; TransE's translation model has no such fixed point
2. RotatE uses a larger embedding dimension, giving it more capacity for symmetric relations
3. RotatE applies a symmetric normalization to entity embeddings before scoring
4. RotatE models symmetry by using the same embedding for both \(r\) and \(r^{-1}\)
</div>

??? question "Show Answer"
    The correct answer is **A**. RotatE's scoring is \(f(h,r,t) = -\|\mathbf{h} \circ \mathbf{r} - \mathbf{t}\|\), where \(\circ\) is element-wise complex multiplication (rotation by angle \(\theta_r\)). For a symmetric relation \(r\), we need \((h,r,t)\) and \((t,r,h)\) to both hold — i.e., \(\mathbf{h} \circ \mathbf{r} = \mathbf{t}\) and \(\mathbf{t} \circ \mathbf{r} = \mathbf{h}\). This is satisfied when \(r\) is a 180° rotation (\(|r_i| = 1\), \(\theta_r = \pi\)): rotating \(\mathbf{h}\) by \(\pi\) gives \(\mathbf{t}\), and rotating \(\mathbf{t}\) by \(\pi\) gives back \(\mathbf{h}\). TransE has no such mechanism: \(\mathbf{h} + \mathbf{r} = \mathbf{t}\) and \(\mathbf{t} + \mathbf{r} = \mathbf{h}\) requires \(\mathbf{r} = \mathbf{0}\), making \(\mathbf{h} = \mathbf{t}\) — a degenerate solution.

    **Concept Tested:** RotatE, Symmetry (Relation Pattern)

---

#### Question 3

Mean Reciprocal Rank (MRR) is computed as the mean over all test triples of \(1/\text{rank}(\text{correct entity})\). What is the advantage of MRR over Hits@K for evaluating KG completion?

<div class="upper-alpha" markdown>
1. MRR can be computed without ranking candidates; Hits@K requires sorting all entities
2. MRR penalizes every rank position, not just those above a threshold K; rank 2 scores 0.5 while rank 100 scores 0.01, continuously rewarding better rankings
3. MRR is normalized between 0 and 100, making it easier to compare across datasets
4. MRR does not require negative sampling, while Hits@K does
</div>

??? question "Show Answer"
    The correct answer is **B**. Hits@K is binary: a prediction contributes 1 if the correct entity ranks in the top K and 0 otherwise. It ignores the difference between rank 1 and rank K−1 (both count equally) and between rank K+1 and rank 10K (both score 0). MRR assigns 1/rank to each test triple, rewarding better ranks continuously. A rank-1 answer contributes 1.0; rank 2 contributes 0.5; rank 10 contributes 0.1. This makes MRR sensitive to the quality of the top predictions rather than just a pass/fail threshold.

    **Concept Tested:** Mean Reciprocal Rank (MRR), Hits@K Metric

---

#### Question 4

DistMult scores triples as \(f(h, r, t) = \sum_i r_i h_i t_i\) (bilinear, diagonal). Why can DistMult not model antisymmetric relations?

<div class="upper-alpha" markdown>
1. DistMult uses integer embeddings that cannot represent negative scores
2. DistMult requires all relation vectors to have unit norm, preventing sign changes
3. DistMult's scoring function is symmetric in \(h\) and \(t\): \(f(h,r,t) = f(t,r,h)\), so it assigns the same score to a triple and its reverse — antisymmetric relations require \(f(h,r,t) \neq f(t,r,h)\) when \(t \neq h\)
4. DistMult cannot represent transitive relations, which are required for antisymmetry
</div>

??? question "Show Answer"
    The correct answer is **C**. The DistMult score \(\sum_i r_i h_i t_i\) is symmetric: swapping \(h\) and \(t\) gives the same value. An antisymmetric relation (e.g., "is_parent_of": if Alice is_parent_of Bob, then Bob is NOT is_parent_of Alice) requires the model to assign high score to (Alice, is_parent_of, Bob) and low score to (Bob, is_parent_of, Alice). DistMult cannot achieve this because both triples receive the same score. ComplEx and RotatE overcome this with asymmetric scoring functions.

    **Concept Tested:** DistMult, Antisymmetry (Relation)

---

#### Question 5

ComplEx extends DistMult to complex embeddings. How does working in complex space allow ComplEx to model antisymmetric relations?

<div class="upper-alpha" markdown>
1. Complex numbers have imaginary parts that automatically encode directionality
2. ComplEx computes \(f(h,r,t) = \text{Re}\!\left(\sum_i r_i h_i \bar{t}_i\right)\) using the conjugate of \(t\), which breaks the \(h \leftrightarrow t\) symmetry since \(\bar{t} \neq t\) in general
3. Complex embeddings allow negative scores, which are required to penalize antisymmetric triples
4. ComplEx uses matrix products rather than element-wise products, enabling full bilinear capacity
</div>

??? question "Show Answer"
    The correct answer is **B**. ComplEx's score \(\text{Re}\!\left(\sum_i r_i h_i \bar{t}_i\right)\) uses the complex conjugate of \(t\) rather than \(t\) itself. Since conjugation flips the imaginary part (\(\bar{a} = a_{\text{re}} - i\,a_{\text{im}} \neq a\) unless \(a\) is real), swapping \(h\) and \(t\) changes the score: \(\text{Re}\!\left(\sum_i r_i t_i \bar{h}_i\right) \neq \text{Re}\!\left(\sum_i r_i h_i \bar{t}_i\right)\) in general. This asymmetry lets ComplEx capture both symmetric (where imaginary parts cancel) and antisymmetric relations (where they don't), while remaining computationally efficient.

    **Concept Tested:** ComplEx, Antisymmetry (Relation)

---

#### Question 6

What is the KG completion task, and how is it evaluated in practice?

<div class="upper-alpha" markdown>
1. Predicting missing node features in a knowledge graph, evaluated by mean squared error
2. Predicting missing relations (edges) in a KG given entity embeddings, evaluated by ranking the correct missing entity among all N entities and computing MRR and Hits@K
3. Constructing a knowledge graph from raw text, evaluated by precision-recall of extracted triples
4. Partitioning the KG into communities, evaluated by modularity score
</div>

??? question "Show Answer"
    The correct answer is **B**. KG completion (also called link prediction in KGs) predicts which triples are missing from an incomplete KG. Given a test triple (h, r, ?) or (?, r, t), the model scores all N possible entity completions and ranks the correct answer. MRR and Hits@K are computed over all test triples. The filtered setting (used in standard evaluations) removes other true triples from the ranking to avoid penalizing correct predictions that were not in the test set.

    **Concept Tested:** KG Completion, Link Prediction (KG)

---

#### Question 7

TransR extends TransE by mapping entities into a relation-specific subspace before translation. What limitation of TransE does this address?

<div class="upper-alpha" markdown>
1. TransE cannot handle graphs with more than one relation type
2. TransE requires directed graphs; TransR extends the model to undirected graphs
3. TransE cannot handle one-to-many relations because translations are deterministic; TransR uses probabilistic projections
4. TransE uses the same embedding space for all relations, making it hard to model relations where the relevant entity features differ across relation types; TransR uses a projection matrix M_r to map entities into a relation-specific space before applying the translation
</div>

??? question "Show Answer"
    The correct answer is **D**. TransE uses a single \(d\)-dimensional embedding space for all entities and all relations. For different relations, the semantically relevant dimensions of an entity may differ: for "is_located_in," the geographic coordinates of an entity matter; for "was_founded_by," the founder's identity matters. TransR projects entity embeddings into a relation-specific \(d_r\)-dimensional subspace via a learned matrix \(M_r \in \mathbb{R}^{d_r \times d}\), then applies the translation in that subspace, giving each relation its own "lens" on the entity embeddings.

    **Concept Tested:** TransR, KG Embedding Geometry

---

#### Question 8

Which of the following relation patterns can RotatE model but TransE cannot?

<div class="upper-alpha" markdown>
1. Composition (r₁ ∘ r₂ = r₃)
2. Antisymmetry
3. Symmetry (r(h,t) ⟹ r(t,h))
4. Inversion (r₁(h,t) ⟹ r₂(t,h))
</div>

??? question "Show Answer"
    The correct answer is **C**. TransE models relations as translations: for symmetry, \(\mathbf{h} + \mathbf{r} = \mathbf{t}\) and \(\mathbf{t} + \mathbf{r} = \mathbf{h}\) together require \(\mathbf{r} = \mathbf{0}\) and \(\mathbf{h} = \mathbf{t}\) — degenerate. RotatE models relations as rotations by angle \(\theta_r\): symmetry requires \(\theta_r = \pi\) (180°), giving \(\mathbf{r} \circ (\mathbf{r} \circ \mathbf{h}) = \mathbf{h}\). TransE can model antisymmetry (its default behavior for non-zero \(\mathbf{r}\)), inversion (using \(-\mathbf{r}\) as the inverse relation), and approximately composition (\(\mathbf{r}_1 + \mathbf{r}_2 \approx \mathbf{r}_3\)). Symmetry is the one pattern where RotatE has a non-degenerate solution and TransE does not.

    **Concept Tested:** RotatE, Symmetry (Relation Pattern), Inversion (Relation)

---

#### Question 9

Negative sampling for KG embedding training generates corrupted triples to contrast against positive ones. What is the "self-adversarial" negative sampling strategy?

<div class="upper-alpha" markdown>
1. Negatives are sampled proportionally to the current model's score for the corrupted triple — harder negatives (higher-scored corruptions) are sampled more often to focus training
2. The model generates negatives by corrupting both head and tail entities simultaneously
3. Negatives are generated by randomly shuffling relation types rather than entity substitutions
4. The model uses its own predictions as pseudo-positive triples to train against
</div>

??? question "Show Answer"
    The correct answer is **A**. Self-adversarial negative sampling (Sun et al., RotatE paper) weights each negative triple \((h', r, t)\) by its current score under the model: \(p(h' \mid h,r,t) \propto \exp(\alpha\, f(h',r,t))\). High-scoring corruptions (ones the model currently confuses for positives) are sampled more frequently, focusing training on the hardest negatives. This is analogous to hard negative mining in metric learning and significantly improves training efficiency compared to uniform negative sampling.

    **Concept Tested:** KG Completion

---

#### Question 10

The KG Embedding evaluation protocol uses "filtered" MRR. What does the filtering step remove, and why is it important?

<div class="upper-alpha" markdown>
1. It removes from the ranking all other known true triples (from train/valid/test) so the model is not penalized for ranking a correct but non-target triple above the test answer
2. It removes test triples that appear in the training set, preventing data leakage
3. It removes entities with fewer than 10 appearances to focus evaluation on common entities
4. It filters out relation types that do not appear in the test set to avoid zero-shot evaluation
</div>

??? question "Show Answer"
    The correct answer is **A**. When scoring (h, r, ?), all N entities are ranked. But other entities t' where (h, r, t') is a known true triple should not count as wrong — they are correct answers, just not the specific test triple being evaluated. Filtered MRR removes all known true triples from the candidate ranking before computing the rank of the test answer. Without filtering, the model is penalized for correctly identifying other valid tails, producing artificially pessimistic metrics.

    **Concept Tested:** Mean Reciprocal Rank (MRR), KG Completion
