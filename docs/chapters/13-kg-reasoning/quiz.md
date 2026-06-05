# Quiz: Reasoning over Knowledge Graphs

Test your understanding of multi-hop query answering, query embeddings, box embeddings, and neural reasoning over KGs.

---

#### Question 1

A multi-hop query over a knowledge graph asks: "Which entities satisfy a chain of relations from a given anchor?" Why can't standard KG embedding methods (TransE, RotatE) directly answer multi-hop queries?

<div class="upper-alpha" markdown>
1. Standard KG embeddings score individual triples but cannot compose multiple relation steps or handle conjunctions — traversing r₁ then r₂ from an anchor is not a single triple lookup
2. Multi-hop queries require the graph to be undirected, but TransE only works on directed graphs
3. TransE and RotatE can only process triples with exactly one relation, not chains of relations
4. TransE and RotatE do not scale to the large KGs where multi-hop queries typically arise
</div>

??? question "Show Answer"
    The correct answer is **A**. TransE scores (h, r, ?) by computing h + r and finding nearby entities. A two-hop query "h -r₁→ ? -r₂→ ?" requires chaining two relation steps, and the intermediate node may not exist in an incomplete KG. Answering it requires reasoning about the set of all possible intermediate nodes and all their r₂-successors — a query over a distribution of entities, not a single triple lookup. Query embedding methods generalize KG embeddings to handle this.

    **Concept Tested:** Multi-Hop Query, Query Embedding

---

#### Question 2

Query2Box represents conjunctive queries as axis-aligned hyper-rectangles (boxes) in embedding space. What is the geometric interpretation of the intersection operator ∩ in box space?

<div class="upper-alpha" markdown>
1. The intersection of two boxes is their geometric union, taking the bounding box of both
2. The intersection of two boxes is computed as the element-wise minimum of center coordinates, contracting the box
3. A learned operator produces a new box inside the intersection of the input boxes, modeling the AND operator in conjunctive queries
4. The intersection is undefined when boxes do not overlap — non-overlapping boxes signal an empty answer set
</div>

??? question "Show Answer"
    The correct answer is **C**. In Query2Box, boxes represent sets of entities: an entity answers the query if its embedding falls inside the box. The intersection operator ∩(B₁, B₂) must produce a new box that is a subset of both input boxes (representing the AND of two query constraints). Query2Box implements this with a learned DeepSets-like network that takes the two box representations and outputs a smaller box inside both. The size of the output box encodes uncertainty: a large box means many possible answers, a small box means few.

    **Concept Tested:** Box Embedding (Query2Box), Conjunctive Query

---

#### Question 3

In EPFO queries (Existential Positive First-Order queries with negation excluded), what query types are included and what is the key limitation of box embeddings for the full EPFO class?

<div class="upper-alpha" markdown>
1. EPFO includes only 1-hop and 2-hop chain queries; box embeddings cannot handle 3+ hop chains
2. EPFO requires all query variables to be bound, while box embeddings only handle free variables
3. EPFO includes negation (¬) which box embeddings model as the complement of a box region
4. EPFO includes conjunction (∧), existential quantification (∃), and disjunction (∨); box embeddings handle ∧ and ∃ naturally but struggle with ∨ because the union of two boxes is not convex
</div>

??? question "Show Answer"
    The correct answer is **D**. EPFO queries allow conjunction ∧ (AND), disjunction ∨ (OR), and existential quantification ∃ — but not negation ¬. Box embeddings naturally represent ∧ (intersection of boxes) and ∃ (relation projection of a box). However, ∨ (OR) is problematic: the union of two convex boxes is not convex and cannot be represented by a single box. This is a fundamental geometric limitation: disjunctions require more expressive representations such as a mixture of boxes or a different embedding geometry.

    **Concept Tested:** Box Embedding (Query2Box), Conjunctive Query

---

#### Question 4

The Neural Bellman-Ford Network (NBFNet) answers path queries by running a generalized Bellman-Ford algorithm on a KG. What is the key insight that makes this approach interpretable?

<div class="upper-alpha" markdown>
1. Each node maintains a "path representation" that aggregates all paths leading to it from the query source — the final source-to-target path representation directly scores the query answer, making the reasoning traceable
2. NBFNet decomposes the KG into a tree before running Bellman-Ford, removing all cycles
3. NBFNet uses breadth-first search to enumerate all paths, making the reasoning process explicit
4. NBFNet scores queries by comparing them against stored path patterns from the training set
</div>

??? question "Show Answer"
    The correct answer is **A**. NBFNet propagates "path representations" \(\mathbf{h}_v^{(l)}\) from the source entity along relation edges: at each step, each node receives messages from its predecessors and combines them to summarize all incoming paths. The final representation \(\mathbf{h}_{\text{target}}^{(L)}\) encodes information about all paths from source to target in the KG. The scoring function then maps this path representation to a score. The process is interpretable because you can trace which paths contributed most to the final score using gradient attribution.

    **Concept Tested:** Neural Bellman-Ford Net

---

#### Question 5

Why is multi-hop KG reasoning harder on incomplete KGs than on complete ones?

<div class="upper-alpha" markdown>
1. Incomplete KGs have irregular degree distributions that make Bellman-Ford slower to converge
2. On an incomplete KG, the true intermediate entities along a reasoning path may be missing — a direct graph traversal would fail, so reasoning must implicitly complete the missing edges during query answering
3. Incomplete KGs cannot be stored in a standard adjacency matrix format
4. Multi-hop reasoning requires all intermediate entities to have been seen during training
</div>

??? question "Show Answer"
    The correct answer is **B**. Real-world KGs like Freebase or Wikidata are severely incomplete — it has been estimated that over 70% of entities have missing information. A two-hop chain query h -r₁→ e -r₂→ t might fail to find entity e because the triple (h, r₁, e) was never observed, even if both triples are true. Query embedding methods embed both the query and potential answers in the same space, allowing them to "hallucinate" plausible intermediate entities by reasoning in embedding space rather than traversing explicit graph edges.

    **Concept Tested:** Multi-Hop Query, KG Completion

---

#### Question 6

Box embeddings represent query answers as a region of embedding space. What makes the "center + offset" parameterization of a box particularly suited for relation projection?

<div class="upper-alpha" markdown>
1. The offset ensures the box always contains the source entity, providing a useful inductive bias
2. The center identifies the query entity and the offset encodes the relation type
3. Relation projection \(P_r(B)\) can be implemented as a learned affine transformation of the center and offset, intuitively shifting and resizing the box to represent the set of r-related entities
4. The center + offset parameterization prevents boxes from overlapping, ensuring distinct query answers
</div>

??? question "Show Answer"
    The correct answer is **C**. In Query2Box, a box is parameterized by center \(\mathbf{c} \in \mathbb{R}^d\) and non-negative offset \(\mathbf{o} \in \mathbb{R}^d\) (half-widths). Applying relation \(r\) to box \(B\) projects it to a new region: \(\mathbf{c}' = f_r(\mathbf{c})\) (a learned function shifting the center) and \(\mathbf{o}' = g_r(\mathbf{o})\) (a learned function changing the size). This captures the intuition that different relation types "shift" the entity set in different directions and with different spreads — a specific entity maps to a precise point, while a vague query maps to a large box.

    **Concept Tested:** Box Embedding (Query2Box)

---

#### Question 7

Query embedding methods can answer queries over incomplete KGs by reasoning in embedding space. What is the training objective for Query2Box?

<div class="upper-alpha" markdown>
1. Maximize the area of query boxes containing correct answer entities
2. Minimize the distance from correct answer entity embeddings to the center of the query box, while maximizing the distance from negative samples to the nearest box boundary
3. Reconstruct the original KG adjacency from query box representations
4. Predict the query type (1-hop vs. 2-hop vs. 3-hop) from the query box dimensions
</div>

??? question "Show Answer"
    The correct answer is **B**. Query2Box defines \(d(q, v) = \text{outside}(q, v) + \alpha \cdot \text{inside}(q, v)\) as the distance from entity \(v\) to query box \(q\), where outside measures how far \(v\) is outside the box and inside is the soft measure of \(v\)'s position within the box (penalized at a discounted rate \(\alpha < 1\)). Training minimizes this distance for correct answers and maximizes it (via margin) for negative samples. This trains both the query embedding operator and entity embeddings jointly.

    **Concept Tested:** Query Embedding, Box Embedding (Query2Box)

---

#### Question 8

The Neural Bellman-Ford Network is related to classical Bellman-Ford shortest path algorithms. In what way does NBFNet generalize classical Bellman-Ford?

<div class="upper-alpha" markdown>
1. NBFNet runs backward from target to source while Bellman-Ford runs forward from source
2. NBFNet uses floating-point embeddings instead of integer distances, reducing numerical precision
3. Classical Bellman-Ford propagates scalar distances using min/+ semiring; NBFNet propagates vector representations using learned message and aggregation functions, generalizing the semiring operations to neural networks
4. NBFNet terminates in exactly N iterations while Bellman-Ford may take up to N−1 iterations
</div>

??? question "Show Answer"
    The correct answer is **C**. Classical Bellman-Ford: \(d(v) \leftarrow \min_{u \to v}(d(u) + w_{uv})\). NBFNet replaces scalar distance with a \(d\)-dimensional path representation vector, \(\min\) with a neural aggregation function, and scalar addition with a learned message function: \(\mathbf{h}_v^{(l+1)} \leftarrow \text{AGG}(\{\text{MSG}(\mathbf{h}_u^{(l)}, r_{uv}) : u \to v\})\). The key insight is that Bellman-Ford is essentially message passing on a directed graph — NBFNet makes this connection explicit by treating path representations as "generalized distances" that encode not just path length but rich semantic reasoning.

    **Concept Tested:** Neural Bellman-Ford Net

---

#### Question 9

A conjunctive query (2i query) has the form: ∃e₁, e₂ . r₁(a₁, e₁) ∧ r₂(a₂, e₁) — find all e₁ satisfying two constraints from different anchors. What additional geometric challenge does this pose compared to a chain query?

<div class="upper-alpha" markdown>
1. The query requires two separate models, one per anchor, making the computation twice as expensive
2. Projection from two different anchors produces two separate boxes; these must be intersected, but the two boxes may lie in very different regions of embedding space, making their intersection empty even for valid answers
3. Conjunctive queries require directed graphs while chain queries work on undirected graphs
4. Intersection queries cannot be answered with a continuous embedding because the answer set is discrete
</div>

??? question "Show Answer"
    The correct answer is **B**. A chain query (1p/2p/3p) projects a single query box through a sequence of relation operators. An intersection query (2i/3i) starts from multiple anchor entities, projects each along separate relation chains, and then intersects the resulting boxes. The challenge is that embedding learning must ensure the two query boxes overlap in the region of correct answer entities — a much harder training objective since the two relation chains start from different anchors. Query2Box's intersection operator must find the right overlap even when boxes are not perfectly aligned.

    **Concept Tested:** Conjunctive Query, Box Embedding (Query2Box)

---

#### Question 10

What makes NBFNet able to generalize to relation types or entity pairs unseen during training (inductive KG reasoning)?

<div class="upper-alpha" markdown>
1. NBFNet stores all training paths in a lookup table and retrieves the closest match at inference
2. NBFNet requires all test entities to appear in the training graph at least twice
3. NBFNet is pre-trained on a large external KG and fine-tuned on the target KG, like a language model
4. NBFNet uses only structural features (relation types and graph topology) rather than entity-specific embeddings, allowing path reasoning rules to transfer to new entities based on their structural context
</div>

??? question "Show Answer"
    The correct answer is **D**. Shallow embedding methods (TransE, RotatE) store a vector per entity — new entities have no embedding. NBFNet computes representations dynamically using relation-type parameters and graph structure. A new entity's path representation is computed from scratch by running Bellman-Ford from the query source through whatever edges exist in the graph. Since the relation-type message functions are shared across all entities, they transfer directly to unseen entities with new edges. This is the key to inductive reasoning, formalized further in Chapter 14 with ULTRA.

    **Concept Tested:** Neural Bellman-Ford Net, Multi-Hop Query
