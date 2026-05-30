---
title: "Chapter 14: Knowledge Graph Foundation Models"
description: "Examines how inductive reasoning and transfer learning extend KG methods beyond fixed entity sets, covering ULTRA's universal pre-training approach and InGram's relation graph induction."
generated_by: claude skill chapter-content-generator
date: 2026-05-29 23:30:00
version: 0.08
---

# Chapter 14: Knowledge Graph Foundation Models

**Part 3: Knowledge Graphs and Reasoning**

## Summary

Examines how inductive reasoning and transfer learning extend KG methods beyond fixed entity sets, covering ULTRA's universal pre-training approach and InGram's relation graph induction.

## Concepts Covered

This chapter covers the following 6 concepts from the learning graph:

1. Inductive KG Reasoning
2. ULTRA
3. InGram
4. Transfer Learning (Graphs)
5. Pre-Training (GNN)
6. Fine-Tuning (GNN)

## Prerequisites

This chapter builds on:

- [Chapter 13: Reasoning over Knowledge Graphs](../13-kg-reasoning/index.md)

---

!!! mascot-welcome "From Memorizing Entities to Understanding Structure"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving welcome">
    Chapters 12 and 13 developed powerful machinery for KG link prediction and multi-hop query answering — but that machinery carried a hidden limitation. Every model assumed the entity set was fixed at training time: TransE assigns a dedicated embedding vector to each entity ID, RotatE rotates a per-entity complex number, and Query2Box anchors a box at each entity's position. When a new entity appears — one that was never in the training vocabulary — there is no embedding to look up, and the model silently fails. This chapter asks: what would it take to build a KG reasoning system that works on any knowledge graph, even one it has never seen? The answer reveals a surprisingly deep connection between equivariance, structural induction, and the emerging paradigm of graph foundation models.

---

## 14.1 Motivating Example: Reasoning Under Epidemiological Uncertainty

When SARS-CoV-2 emerged in late 2019, biomedical researchers immediately began constructing knowledge graphs to organize what was rapidly being discovered: viral protein structures, host-pathogen interaction mechanisms, potential drug targets, approved small molecules. Within months, databases such as the Drug Repurposing Knowledge Graph (DRKG) had expanded to include thousands of new entities — viral gene products, host receptor sites, and novel compound-target interactions — that were entirely absent from pre-existing biomedical KGs.

Researchers attempting to use link prediction models trained on prior KGs (DRKG v1, hetionet, or PrimeKG) faced an immediate problem: every new viral protein was an out-of-vocabulary entity. TransE had no embedding for NSP3, RotatE had no complex rotation vector for the Spike receptor-binding domain, and ComplEx had never seen ACE2 in its training data. Retraining from scratch on the new combined graph would have taken weeks on academic GPU clusters — weeks that were unavailable during an active pandemic.

What was needed was a system capable of **zero-shot generalization**: given a new KG it had never seen, immediately produce sensible link predictions without any parameter updates. This requires the model to have learned not *facts* — not "which entities are connected" — but *patterns* — "what kinds of structural relationships tend to imply which other relationships, regardless of which specific entities are involved." A model that has internalized relational patterns from diverse training KGs can apply those patterns instantly to any new graph that exhibits similar structural regularities.

This chapter develops the theoretical and algorithmic framework for building such systems, culminating in ULTRA (Universal Learning of Trirelational data Analysis), which achieves competitive link prediction on 43 diverse KGs in zero-shot mode, trained on only 3.

---

## 14.2 The Transductive Bottleneck

Every model studied in Chapters 12 and 13 — TransE, DistMult, ComplEx, RotatE, Query2Box, and NBFNet — is ultimately **transductive**: it builds a fixed lookup table that maps entity IDs to embedding vectors. More precisely, each model learns a parameter matrix \( \mathbf{E} \in \mathbb{R}^{|\mathcal{E}| \times d} \), where row \( i \) is the embedding of entity \( e_i \). At inference time, scoring a triple \( (h, r, t) \) means looking up \( \mathbf{E}[h] \), \( \mathbf{W}_r \), and \( \mathbf{E}[t] \) and applying the scoring function.

This architecture enforces a hard boundary: only entities whose IDs were indexed during training can ever be scored. Formally, let \( \mathcal{E}_{train} \) be the entity set observed at training time and \( \mathcal{E}_{test} \supset \mathcal{E}_{train} \) be the entity set at test time. A transductive model learns

\[
f_\theta : \mathcal{E}_{train} \times \mathcal{R} \times \mathcal{E}_{train} \to \mathbb{R},
\]

but cannot evaluate \( f_\theta(h, r, t) \) when \( h \notin \mathcal{E}_{train} \) or \( t \notin \mathcal{E}_{train} \), because \( \mathbf{E}[h] \) and \( \mathbf{E}[t] \) simply do not exist. The parameters \( \theta \) encode entity-specific information that is non-transferable to new entities, and even more so to new knowledge graphs with entirely different entity vocabularies.

This limitation has a second, subtler consequence: the embedding matrix itself grows with the size of the entity set. Freebase has 40 million entities; storing a 200-dimensional complex embedding for each requires \( 40\text{M} \times 200 \times 8 \approx 64\text{ GB} \) of memory. Scaling to Wikidata (90M+ entities) is prohibitive.

The root cause is the reliance on entity IDs as the input signal. Entity IDs are arbitrary integers — they carry no information about what an entity *is* or how it *connects* to other entities. A more principled architecture would represent entities by their **structural context** — their patterns of connectivity to other entities and relations — which is portable across graphs and does not require indexing.

---

## 14.3 Inductive KG Reasoning: A Formal Taxonomy

Inductive reasoning in knowledge graphs refers to any setting where inference must be performed on entities, relations, or triples that were absent from the training distribution. The field has distinguished several increasingly demanding variants:

**Semi-inductive (new entities, same relations):** The training relation vocabulary \( \mathcal{R} \) is fixed, but the test entity set \( \mathcal{E}_{test} \) includes entities not seen during training. This is the setting explored by GraIL (Teru et al. 2020) and NodePiece (Galkin et al. 2021). The model must represent new entities without ID lookups, typically by aggregating information from their known relational neighborhoods.

**Fully inductive (new entities and new relations):** Both the entity set and the relation vocabulary may contain unseen elements at test time. InGram operates in this setting by representing relations through structural co-occurrence patterns rather than ID lookups, enabling generalization to relation types that appeared in zero training triples.

**Cross-KG transfer (new graph):** The most demanding setting: training occurs on source KGs \( G_1, G_2, \ldots, G_k \), and inference is required on a target KG \( G_{target} \) whose entity and relation vocabularies are entirely disjoint from all training graphs. A model capable of cross-KG transfer has learned domain-agnostic relational patterns rather than graph-specific facts. ULTRA achieves this zero-shot.

The key challenge distinguishing these settings from standard supervised machine learning is the absence of any shared vocabulary: there is no analogue of ImageNet's fixed 1,000-class label space or BERT's 30,000-word vocabulary. Every KG is a self-contained semantic universe, and a foundation model must bridge these universes without a common reference system.

!!! mascot-thinking "Structural Role vs. Entity Identity"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinking carefully">
    The key insight underlying all inductive KG methods is that entity *identity* — the specific integer ID assigned to an entity — carries no semantic information useful for transfer. What carries information is structural *role*: does this entity tend to appear at the head of functional relations? Does it appear as both head and tail in the same relation type (symmetric)? Is it a high-degree hub or a leaf node? A representation built from structural role can generalize because structural roles recur across KGs. The entity "Albert Einstein" in Wikidata and the entity "Leonhard Euler" in MathKG occupy similar structural roles (person → birthplace, person → field-of-study, person → influenced) even though they are entirely different entities in entirely different KGs.

---

## 14.4 Transfer Learning and Pre-Training on Graphs

Pre-training followed by fine-tuning is the dominant paradigm in modern deep learning. In natural language processing, pre-training a large transformer on a massive text corpus (language modeling) yields representations that transfer to downstream tasks with minimal fine-tuning. In computer vision, ImageNet pre-training produces visual features that transfer to medical imaging, satellite imagery, and industrial inspection. The question for KG reasoning is whether an analogous workflow is achievable — and if so, what the pre-training objective should be.

For GNNs applied to molecule property prediction, pre-training via graph-level contrastive learning or attribute masking (Hu et al. 2020) has demonstrated significant few-shot generalization. For KG reasoning specifically, pre-training is harder for three reasons.

First, **vocabulary mismatch**: Unlike text tokens (which appear in all corpora) or image pixels (which always span the same 0–255 intensity range), entity and relation IDs are arbitrary across KGs. There is no shared symbol that means "is-a" across FB15k-237, WN18RR, and a biomedical KG simultaneously.

Second, **structural heterogeneity**: Different KGs have radically different statistics. FB15k-237 has a power-law degree distribution dominated by a few high-degree hub entities (countries, famous people, major films); WN18RR has a more hierarchical structure (hypernym/hyponym chains); biomedical KGs are dense in certain biological pathway subgraphs and sparse elsewhere. A single representation space may need to accommodate all of these.

Third, **task diversity**: Link prediction, multi-hop query answering, and triple classification require different inductive biases even when applied to the same KG. A pre-trained foundation model ideally supports all three without task-specific architectural changes.

The solution pursued by ULTRA is to make the model **entity-agnostic** and **relation-agnostic** by construction — that is, to never use entity IDs or relation IDs as input features. Instead, both entities and relations are represented by their structural context in the graph, computed fresh for each new KG at inference time. Pre-training then learns the GNN parameters that extract these structural contexts, not a fixed vocabulary.

Fine-tuning on a target KG proceeds by optionally running a small number of gradient steps on observed target triples, adapting the structural context extraction to the target domain. Even a few hundred training triples on the target KG can provide meaningful signal, because the model is not learning entity embeddings from scratch — only refining its structural context extractor.

---

## 14.5 InGram: Inductive Reasoning via Relation Graphs

InGram (Lee et al. 2023) introduced the **relation graph** as the fundamental data structure for inductive KG representation. Rather than representing each relation by a lookup vector, InGram characterizes each relation by its co-occurrence pattern with other relations — a structure that is purely topological and therefore portable across KGs.

### 14.5.1 Constructing the Relation Graph

Given a KG \( G = (\mathcal{E}, \mathcal{R}, \mathcal{T}) \), the relation graph \( G_R = (\mathcal{R}, E_R, \Phi) \) is a directed graph over relation nodes with four types of edges, denoted by \( \Phi \in \{hh, ht, th, tt\} \):

- **Head-head (hh):** An edge of type hh connects \( r_1 \) to \( r_2 \) if there exists an entity \( e \) that appears as the head of a triple with relation \( r_1 \) *and* as the head of a triple with relation \( r_2 \): \( \exists e, t_1, t_2 : (e, r_1, t_1) \wedge (e, r_2, t_2) \in \mathcal{T} \).

- **Head-tail (ht):** Connects \( r_1 \) to \( r_2 \) if an entity appears as the head of an \( r_1 \)-triple and as the tail of an \( r_2 \)-triple: \( \exists e, h_2, t_1 : (e, r_1, t_1) \wedge (h_2, r_2, e) \in \mathcal{T} \).

- **Tail-head (th):** Connects \( r_1 \) to \( r_2 \) if an entity appears as the tail of an \( r_1 \)-triple and as the head of an \( r_2 \)-triple: \( \exists e, h_1, t_2 : (h_1, r_1, e) \wedge (e, r_2, t_2) \in \mathcal{T} \).

- **Tail-tail (tt):** Connects \( r_1 \) to \( r_2 \) if a shared entity appears as the tail of both relations.

These four edge types encode compositional and co-occurrence statistics without naming specific entities. Consider a biomedical KG: if "encodes" and "expressed-in" frequently share head entities (the same gene both encodes a protein and is expressed in a tissue), then there will be many hh-edges between those two relation nodes. This structural pattern recurs in any gene-centric KG, making it portable.

A **relational GNN** then operates on \( G_R \), computing a representation \( \mathbf{r}_i \in \mathbb{R}^d \) for each relation node by aggregating messages from neighboring relation nodes according to the four edge types:

\[
\mathbf{r}_i^{(l+1)} = \sigma\!\left(\mathbf{W}^{(l)}_\text{self}\, \mathbf{r}_i^{(l)} + \sum_{\phi \in \{hh,ht,th,tt\}} \sum_{j : (i \to j) \in E_R^\phi} \mathbf{W}^{(l)}_\phi\, \mathbf{r}_j^{(l)}\right)
\]

where the four weight matrices \( \mathbf{W}^{(l)}_\phi \in \mathbb{R}^{d \times d} \) are shared across all relation nodes (making the model relation-agnostic) and across all KGs (making it graph-agnostic). After \( L \) layers, each relation is represented by a \( d \)-dimensional vector that encodes its structural role in the relation co-occurrence graph.

### 14.5.2 Entity Representations in InGram

With relation embeddings in hand, InGram computes entity representations by aggregating over their incident triples. For entity \( e \), its embedding is the weighted combination of the relation embeddings of triples in which \( e \) participates:

\[
\mathbf{e}_i = \text{Aggregate}\!\left(\left\{\mathbf{r}_j : (e_i, r_j, *) \in \mathcal{T}\right\} \cup \left\{-\mathbf{r}_j : (*, r_j, e_i) \in \mathcal{T}\right\}\right)
\]

where the sign flip for tail participation follows RotatE's convention (tail participation is the inverse of head participation). This representation is computed purely from the entity's structural neighborhood — its pattern of relational connections — not from any ID lookup.

### 14.5.3 Triple Scoring

Given entity and relation embeddings produced inductively, InGram applies a RotatE-style scoring function:

\[
f(h, r, t) = -\|\mathbf{e}_h \circ e^{i\, \mathbf{r}} - \mathbf{e}_t\|_2^2
\]

where \( e^{i\, \mathbf{r}} \) denotes the element-wise complex rotation by angles in \( \mathbf{r} \). Because the entity and relation embeddings are computed inductively from the KG structure, this scoring function generalizes to any new KG without retraining — simply reconstruct the relation graph, run the relational GNN, aggregate entity embeddings from their neighborhoods, and score.

---

## 14.6 ULTRA: Universal Transfer for KG Reasoning

ULTRA (Galkin et al. 2023) scales the InGram idea into a full **foundation model** for KG reasoning by pre-training across multiple heterogeneous KGs simultaneously and integrating the NBFNet backbone (Chapter 13) for entity-level path reasoning.

### 14.6.1 Architecture Overview

ULTRA decomposes KG reasoning into two coupled components:

**Relation encoder (GNN on relation graph).** The same four-type relation graph as InGram is constructed for each KG in the training mixture. A shared GNN with parameters \( \theta_R \) processes this relation graph to produce relation representations \( \{\mathbf{r}_i\}_{r_i \in \mathcal{R}} \). Crucially, \( \theta_R \) is shared across all KGs — it is not specialized to any particular relation vocabulary.

**Entity encoder (NBFNet path propagation).** Given a query \( (e_s, r_q, ?) \), the entity encoder initializes node states according to the NBFNet rule (Chapter 13):

\[
\mathbf{h}_v^{(0)} = \begin{cases} \mathbf{r}_q & \text{if } v = e_s \\ \mathbf{0} & \text{otherwise} \end{cases}
\]

where \( \mathbf{r}_q \) is the query relation embedding produced by the relation encoder — not a fixed lookup vector. The Bellman-Ford propagation then proceeds for \( T \) steps, using the learned relation representations as message operators:

\[
\mathbf{h}_v^{(t+1)} = \sigma\!\left(\sum_{(u, r', v) \in \mathcal{N}(v)} f_\theta(\mathbf{h}_u^{(t)},\, \mathbf{r}_{r'}) + \mathbf{h}_v^{(t)}\right)
\]

where \( f_\theta(\mathbf{h}_u, \mathbf{r}_{r'}) = \mathbf{h}_u \odot (\mathbf{W}_{r'}^\text{rel}\, \mathbf{r}_{r'}) \) is the query-conditioned message function. Because \( \mathbf{r}_{r'} \) is produced by the relation encoder rather than looked up by ID, this propagation is fully inductive.

**Scoring.** After \( T \) steps, entity \( v \)'s score as the answer to query \( (e_s, r_q, ?) \) is:

\[
\text{score}(v) = \text{MLP}\!\left(\mathbf{h}_v^{(T)}\right)
\]

### 14.6.2 Multi-KG Pre-Training

ULTRA's training set comprises three source KGs chosen for structural diversity:

| Source KG | Entities | Relations | Triples | Domain |
|---|---|---|---|---|
| FB15k-237 | 14,541 | 237 | 272,115 | General (Freebase) |
| WN18RR | 40,943 | 11 | 86,835 | Lexical (WordNet) |
| Codex-M | 17,050 | 51 | 185,584 | Encyclopedic (Wikidata) |

At each training step, a minibatch of queries is sampled uniformly from the union of all three training KGs. For each query, the appropriate KG's relation graph is constructed and passed to the relation encoder, and NBFNet propagation is run on that KG's triple set. Gradient descent updates \( \theta_R \) and \( \theta_\text{NBFNet} \) jointly across KGs.

The model never sees entity IDs as inputs — only the topology of the relation graph and the topology of the entity-level KG. This constraint forces the model to learn transferable structural patterns.

### 14.6.3 Zero-Shot Inference on a New KG

Given a new KG \( G_{target} \) with completely novel entities and relations:

1. **Extract relation graph** \( G_R^{target} \) from \( \mathcal{T}_{target} \) using the four co-occurrence edge types.
2. **Run relation encoder** \( \theta_R \) on \( G_R^{target} \) to obtain relation embeddings \( \{\mathbf{r}_i^{target}\} \).
3. **For each query** \( (h, r, ?) \), initialize NBFNet with \( \mathbf{h}_h^{(0)} = \mathbf{r}_q^{target} \) and run \( T \) steps of Bellman-Ford propagation.
4. **Score all candidate entities** and return the top-\( k \).

No gradient updates. No entity vocabulary. No relation vocabulary. The entire inference is driven by structural patterns encoded in \( \theta_R \) and \( \theta_\text{NBFNet} \), which were learned from source KGs but generalize to target.

!!! mascot-tip "When to Fine-Tune ULTRA"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage giving practical advice">
    Zero-shot ULTRA is most powerful when the target KG has structural patterns similar to the training mixture (general-purpose encyclopedic or lexical graphs). For highly specialized KGs — deep biomedical pathway networks, legal citation graphs, manufacturing process ontologies — the structural patterns may differ substantially from anything in the training mix. In these cases, fine-tuning ULTRA for a few dozen steps on even 500–1,000 observed target triples typically yields 5–10 MRR points of improvement. The key advantage over training from scratch is that fine-tuning adapts structural *context extraction*, not individual entity embeddings, so it remains inductive: fine-tuned ULTRA still generalizes to new entities within the target KG without re-running any optimization.

---

## 14.7 Double Equivariance: The Theoretical Foundation

Guo, Namata, and Faloutsos (2023) provide a formal theoretical characterization of what any inductive KG model must satisfy to generalize across entity and relation vocabularies. Their central concept is **double equivariance** — equivariance with respect to simultaneous permutations of both entity and relation IDs.

Let \( \pi_\mathcal{E} \) denote a permutation of entity IDs and \( \pi_\mathcal{R} \) denote a permutation of relation IDs. Applying \( \pi_\mathcal{E} \) to a KG \( G \) relabels every entity while preserving all triple relationships; applying \( \pi_\mathcal{R} \) relabels every relation type.

A link prediction model \( f_\theta : G \times \mathcal{Q} \to \mathbb{R}^{|\mathcal{E}|} \) is **doubly equivariant** if

\[
f_\theta\!\left(\pi_\mathcal{E}(\pi_\mathcal{R}(G)),\ \pi_\mathcal{E}(\pi_\mathcal{R}(q))\right) = \pi_\mathcal{E}\!\left(f_\theta(G, q)\right)
\]

for all permutations \( \pi_\mathcal{E} \) and \( \pi_\mathcal{R} \) and all queries \( q \). Informally: if you arbitrarily rename every entity and every relation type in the graph and the query, the model's answer (reordered by the same entity permutation) should be identical.

This condition immediately rules out transductive models. A TransE model assigns entity \( e_i \) the embedding \( \mathbf{E}[i] \), which depends on the integer index \( i \). Permuting entity IDs — renaming entity 3 to entity 7 and vice versa — changes the embedding without changing the graph structure, violating entity equivariance.

It also rules out relation-ID-dependent models. A model that looks up \( \mathbf{W}_r \) by relation integer index will produce different scores if relation IDs are permuted, even though the triple relationships are unchanged.

The constructive significance of double equivariance is that it forces models to represent entities and relations by their **structural fingerprints** — functions of the graph topology that are invariant to ID assignment. Both InGram's relation graph GNN and ULTRA's full architecture satisfy double equivariance by construction: they use no ID features anywhere in the computation graph.

!!! mascot-warning "Common Equivariance Pitfalls"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning about pitfalls">
    Two errors frequently appear in inductive KG research. First, **partial inductiveness claims**: a model that learns inductive entity representations but uses fixed relation ID embeddings (a common approximation) is entity-equivariant but not doubly equivariant. It will fail on new KGs where the relation vocabulary differs from training. Second, **evaluation leakage**: standard filtered evaluation (from Chapter 12) filters out all *training* triples when ranking. In inductive settings, you must filter out all triples in the *inference graph*, not just training triples — failing to do so inflates MRR by incorrectly counting known true answers as "not filtered." Many early inductive baselines underreported difficulty for exactly this reason.

---

## 14.8 Complete Code: Building a Relation Graph and Running Inductive Scoring

The following code demonstrates three key ideas: constructing an inductive split where training and test entities are disjoint, building the relation co-occurrence graph, and running a simplified ULTRA-style inductive scoring pass. The code is self-contained and designed for transparency over efficiency; a production implementation would use sparse adjacency matrices and batched GNN passes.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from collections import defaultdict
from typing import List, Tuple, Dict

# ── Data types ──────────────────────────────────────────────────────────────
Triple = Tuple[int, int, int]   # (head_id, relation_id, tail_id)

# ── 1. Inductive Split Construction ─────────────────────────────────────────

def make_inductive_split(
    triples: List[Triple],
    num_relations: int,
    train_frac: float = 0.7,
    seed: int = 42,
) -> Tuple[List[Triple], List[Triple]]:
    """
    Partition triples so that the test graph's entity set is
    completely disjoint from the training graph's entity set.

    Strategy: assign entities to train/test pools randomly, then
    keep only triples where both endpoints belong to the same pool.
    Returns (train_triples, test_triples).
    """
    torch.manual_seed(seed)
    all_entities = list({e for h, r, t in triples for e in (h, t)})
    torch.manual_seed(seed)
    perm = torch.randperm(len(all_entities))
    split_idx = int(len(all_entities) * train_frac)
    train_entities = set(all_entities[i] for i in perm[:split_idx])
    test_entities  = set(all_entities[i] for i in perm[split_idx:])

    train_triples = [(h, r, t) for h, r, t in triples
                     if h in train_entities and t in train_entities]
    test_triples  = [(h, r, t) for h, r, t in triples
                     if h in test_entities  and t in test_entities]

    print(f"Train: {len(train_entities)} entities, {len(train_triples)} triples")
    print(f"Test:  {len(test_entities)} entities, {len(test_triples)} triples")
    print(f"Entity overlap: {train_entities & test_entities}")  # should be empty set
    return train_triples, test_triples


# ── 2. Relation Graph Construction ──────────────────────────────────────────

def build_relation_graph(
    triples: List[Triple],
    num_relations: int,
) -> Data:
    """
    Build the 4-type relation co-occurrence graph used by InGram / ULTRA.

    Edge types (following Lee et al. 2023):
      0: head-head  (r1, r2 share a head entity)
      1: head-tail  (r1's head == r2's tail)
      2: tail-head  (r1's tail == r2's head)
      3: tail-tail  (r1, r2 share a tail entity)

    Returns a PyG Data object where nodes = relations, edges = co-occurrences.
    """
    # Index triples by entity role
    head_to_rels: Dict[int, List[int]] = defaultdict(list)  # entity -> list of rels where entity is head
    tail_to_rels: Dict[int, List[int]] = defaultdict(list)  # entity -> list of rels where entity is tail

    for h, r, t in triples:
        head_to_rels[h].append(r)
        tail_to_rels[t].append(r)

    edges = []
    edge_types = []

    def add_edges(rels_a: List[int], rels_b: List[int], etype: int):
        for r1 in rels_a:
            for r2 in rels_b:
                if r1 != r2:
                    edges.append((r1, r2))
                    edge_types.append(etype)

    for entity in head_to_rels:
        rels = head_to_rels[entity]
        # head-head: entity is head of both
        add_edges(rels, rels, etype=0)
        # head-tail: entity is head of r1 and tail of r2
        if entity in tail_to_rels:
            add_edges(rels, tail_to_rels[entity], etype=1)

    for entity in tail_to_rels:
        rels = tail_to_rels[entity]
        # tail-tail: entity is tail of both
        add_edges(rels, rels, etype=3)
        # tail-head: entity is tail of r1 and head of r2
        if entity in head_to_rels:
            add_edges(rels, head_to_rels[entity], etype=2)

    if not edges:
        # Degenerate: return empty graph
        return Data(num_nodes=num_relations,
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_type=torch.zeros(0, dtype=torch.long))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type  = torch.tensor(edge_types, dtype=torch.long)
    return Data(num_nodes=num_relations, edge_index=edge_index, edge_type=edge_type)


# ── 3. Relation Encoder (GNN on Relation Graph) ──────────────────────────────

class RelationEncoder(nn.Module):
    """
    GNN that operates on the relation co-occurrence graph to produce
    relation representations. Shared parameters across all KGs — this
    is what makes ULTRA's transfer possible.

    Uses R-GCN (Schlichtkrull et al. 2018) with 4 relation types (hh/ht/th/tt).
    In ULTRA's full implementation this is a more expressive GNN; R-GCN
    suffices for this pedagogical demonstration.

    Parameters
    ----------
    hidden_dim : int
        Dimension of relation representations.
    num_layers : int
        Number of GNN message-passing rounds over the relation graph.
    num_edge_types : int
        Always 4 for the standard InGram / ULTRA relation graph.
    """
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2,
                 num_edge_types: int = 4):
        super().__init__()
        # Initial relation features: learned per relation-graph edge type
        # but the *initial node features* are set to all-ones (no ID information)
        self.layers = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations=num_edge_types)
            for _ in range(num_layers)
        ])
        self.hidden_dim = hidden_dim

    def forward(self, rel_graph: Data) -> torch.Tensor:
        """
        Parameters
        ----------
        rel_graph : Data
            The relation graph produced by build_relation_graph().

        Returns
        -------
        rel_emb : Tensor of shape (num_relations, hidden_dim)
            Inductively computed relation embeddings. No entity IDs used.
        """
        # Initial features: uniform (no relation-ID information)
        x = torch.ones(rel_graph.num_nodes, self.hidden_dim)

        for conv in self.layers:
            x = conv(x, rel_graph.edge_index, rel_graph.edge_type)
            x = F.relu(x)

        return x  # shape: (num_relations, hidden_dim)


# ── 4. Simplified Inductive Scoring (ULTRA-style) ────────────────────────────

class ULTRAScorer(nn.Module):
    """
    Simplified ULTRA: combines the RelationEncoder with a 1-step
    Bellman-Ford propagation over the entity graph.

    For a query (head, rel, ?):
      1. Encode relations inductively via RelationEncoder.
      2. Initialize source entity with the query relation embedding.
      3. Propagate one hop over entity graph using relation embeddings.
      4. Score all entities.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.rel_encoder = RelationEncoder(hidden_dim)
        self.msg_lin = nn.Linear(hidden_dim, hidden_dim)
        self.score_lin = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        rel_graph: Data,
        entity_edge_index: torch.Tensor,   # (2, num_triples) — edges in entity graph
        entity_edge_rel: torch.Tensor,     # (num_triples,) — relation type for each edge
        query_head: int,
        query_rel: int,
        num_entities: int,
    ) -> torch.Tensor:
        """Returns a score for each entity in [0, num_entities)."""
        # Step 1: encode relations
        rel_emb = self.rel_encoder(rel_graph)  # (num_relations, d)

        # Step 2: initialize entity states
        d = rel_emb.size(1)
        h = torch.zeros(num_entities, d)
        h[query_head] = rel_emb[query_rel]    # query-conditioned source initialization

        # Step 3: 1-hop propagation
        src, dst = entity_edge_index          # both shape (num_triples,)
        msgs = self.msg_lin(h[src] * rel_emb[entity_edge_rel])  # element-wise modulation
        h_new = h.clone()
        h_new.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)
        h_new = F.relu(h_new)

        # Step 4: score each entity
        scores = self.score_lin(h_new).squeeze(-1)  # (num_entities,)
        return scores


# ── 5. Quick demonstration on a toy KG ────────────────────────────────────────

def demo_inductive_scoring():
    """
    Demonstrates zero-shot inductive scoring on a tiny toy KG
    with 10 entities and 3 relation types.
    """
    # Toy training triples
    train_triples = [
        (0, 0, 1), (1, 0, 2), (2, 1, 3), (3, 1, 4),
        (4, 2, 5), (0, 2, 3), (1, 1, 5), (2, 0, 4),
    ]
    # Toy test triples (disjoint entities: 6,7,8,9)
    test_triples = [
        (6, 0, 7), (7, 1, 8), (8, 2, 9), (6, 2, 8),
    ]

    num_relations = 3
    num_train_entities = 6
    num_test_entities = 4   # entities 6,7,8,9 — re-indexed as 0,1,2,3

    # Build relation graph from training triples
    rel_graph = build_relation_graph(train_triples, num_relations)
    print(f"\nRelation graph: {rel_graph.num_nodes} relation nodes, "
          f"{rel_graph.edge_index.size(1)} co-occurrence edges")

    # Build entity graph for test triples (re-indexed: entity 6→0, 7→1, etc.)
    reindex = {6: 0, 7: 1, 8: 2, 9: 3}
    test_src = torch.tensor([reindex[h] for h, r, t in test_triples])
    test_dst = torch.tensor([reindex[t] for h, r, t in test_triples])
    test_rel = torch.tensor([r for h, r, t in test_triples])
    entity_edge_index = torch.stack([test_src, test_dst])

    model = ULTRAScorer(hidden_dim=32)
    scores = model(
        rel_graph=rel_graph,
        entity_edge_index=entity_edge_index,
        entity_edge_rel=test_rel,
        query_head=reindex[6],
        query_rel=0,          # query relation type 0
        num_entities=num_test_entities,
    )
    print(f"Scores for query (entity 6, rel 0, ?): {scores.detach().numpy()}")
    print(f"Top-1 predicted answer (test entity index): {scores.argmax().item()}")

if __name__ == "__main__":
    demo_inductive_scoring()
```

The code highlights the three load-bearing ideas: the inductive split ensures entity sets are disjoint (line 43), the relation graph is built purely from triple co-occurrence topology with no entity-ID features (lines 60–107), and the scoring pass uses relation embeddings computed fresh for each graph rather than a pre-stored lookup table (lines 150–165).

---

## 14.9 Benchmark Results: Inductive Link Prediction

The standard inductive evaluation protocol, introduced by Teru et al. (2020), constructs four versions (v1–v4) of inductive FB15k-237 where training and test entity sets are disjoint. We report Hits@10 and MRR on v1, the most widely reported split, to allow comparison across methods.

| Model | Setting | Hits@10 | MRR | Year |
|---|---|---|---|---|
| Rule Mining (AMIE3) | Inductive (transductive rules) | 38.7 | 0.218 | 2020 |
| GraIL | Inductive supervised | 64.2 | 0.359 | 2020 |
| DRUM | Inductive (soft rules) | 61.2 | 0.337 | 2019 |
| NodePiece | Semi-inductive | 70.4 | 0.401 | 2021 |
| NBFNet (adapted) | Inductive (adapted) | 78.3 | 0.452 | 2021 |
| InGram | Fully inductive | 75.1 | 0.427 | 2023 |
| ULTRA (zero-shot) | Cross-KG transfer | 81.3 | 0.487 | 2023 |
| ULTRA (fine-tuned) | Fine-tuned 3 epochs | 86.2 | 0.531 | 2023 |

Several patterns are worth noting. First, rule-mining methods (AMIE3) perform poorly because inductive rules mined on the training graph frequently do not generalize when entity sets change — structural rules tend to overfit to entity-specific regularities. Second, GraIL, which uses local subgraph structure to represent entities, was the first method to demonstrate that GNNs could generalize inductively at all; its competitive performance in 2020 established the paradigm. Third, ULTRA's zero-shot performance (81.3 Hits@10) exceeds *every supervised baseline* except the adapted NBFNet — meaning ULTRA achieves this without seeing a single test-graph triple during optimization. Fine-tuning with as few as 3 epochs on the test graph's training split pushes ULTRA to state-of-the-art.

!!! mascot-encourage "Abstract Patterns, Concrete Power"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage encouraging the reader">
    The ULTRA benchmark numbers can feel abstract — 86.2 Hits@10 on an inductive split of a Facebook dataset. Bring it back to the motivating example. An ULTRA model pre-trained on encyclopedic KGs (Freebase, WordNet, Codex) achieves competitive link prediction on a new biomedical KG for which it has seen zero triples. The model has not memorized which gene encodes which protein — it has internalized structural patterns like "if two entities share many co-relations of type X, they are likely to share a direct edge of type Y." These patterns, extracted from general-purpose KGs, generalize because biological knowledge graphs and encyclopedic knowledge graphs obey similar structural laws. The abstraction is the power.

---

## 14.10 MicroSim: KG Embedding Space Visualizer

#### Diagram: Cross-KG Structure Transfer

<details markdown="1">
<summary>Interactive 2D embedding space showing how ULTRA represents structural patterns across two different KGs</summary>
Type: MicroSim
**sim-id:** ch14-kg-embedding-space<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Apply and Analyze — students observe how structurally equivalent entities in two different KGs (drawn side-by-side) obtain similar embedding positions despite having completely different entity IDs, illustrating that ULTRA represents structural role rather than identity. Bloom's: Applying (recognizing structural equivalence), Analyzing (comparing embedding clusters across KGs).

**Canvas:** 900×500px total, split into two 420×440 panels (left: training KG, right: test KG) with a 60px divider. Responsive to window resize.

**Left panel — training KG:**
- Display 12 nodes arranged in a mixed topology: 4 high-degree hub nodes (large circles, degree ≥ 4), 4 mid-degree nodes (medium circles, degree 2–3), 4 leaf nodes (small circles, degree 1).
- Edge colors encode relation types: 3 types, drawn as colored lines (teal, orange, purple).
- Node fill color: a 2D embedding computed by a fixed random projection of the structural fingerprint (node degree + relation-type distribution). This projection is identical for both panels.
- On hover, a tooltip shows the node's structural role ("Hub: 5 outgoing edges, types [0,2]" etc.) and 2D embedding coordinates.

**Right panel — test KG:**
- Display 12 entirely different entities (different IDs, completely new KG), but arranged in a structurally identical topology (same adjacency pattern, same relation types).
- Use the same structural fingerprint → 2D projection. Hub nodes in the test KG should cluster near hub nodes in the training KG panel (shown via a faint connecting line when "Show Transfer" is toggled on).
- "Show Transfer" toggle button (below center divider): when ON, draws faint arcs from each test node to its nearest structural neighbor in the training panel, with line opacity proportional to structural similarity score.

**Controls:**
- "Shuffle Entity IDs" button: re-labels all entity integers randomly. The node positions and colors do NOT change (demonstrating that the representation is ID-invariant).
- "Show Transfer" toggle: reveals cross-KG structural similarity arcs.
- "Perturb Structure" button: randomly rewires 2 edges in the test KG, showing that the embedding of affected nodes drifts while structurally stable nodes remain stationary.
- Relation type selector (3 checkboxes): toggle which relation types are shown as edges.

**Behavior:**
- Initial state: both KGs visible, no transfer arcs, entity IDs shown as small integers on nodes.
- After "Shuffle Entity IDs": IDs change, embeddings stay the same — reinforcing entity equivariance.
- After "Show Transfer" toggle: arcs appear between structurally similar nodes across the two panels. The user should observe that hub↔hub, leaf↔leaf connections dominate.
- After "Perturb Structure": affected node updates its color in real time (interpolated over 0.5 seconds), while unaffected nodes stay fixed.

**Color scheme:** dark background (#1a1a2e), teal edges for relation type 0 (#4ecdc4), orange for type 1 (#ff6b6b), purple for type 2 (#c77dff), node fill from viridis scale based on 2D embedding x-coordinate.

Implementation: p5.js with hand-coded force-directed layout (fixed positions stored as arrays, not dynamic). Structural fingerprint computation done once on canvas setup and stored. Cross-KG arcs drawn with bezier curves between the two panels.
</details>

---

## 14.11 Common Pitfalls

**Conflating levels of inductiveness.** "Inductive KG reasoning" is not a single problem but a spectrum from semi-inductive (new entities, same relations) to cross-KG transfer (new entities, new relations, new graph). A model that handles the first setting may fail catastrophically on the third. Always specify which inductiveness level an experiment evaluates.

**Forgetting filtered evaluation in inductive settings.** Standard filtered evaluation (Chapter 12) removes known true triples from the ranking. In inductive settings, the "known true triples" are those of the *inference graph*, not the training graph. Using training-graph triples for filtering conflates the two entity sets and produces systematically optimistic MRR estimates, because training-graph entities are never the answer entities in a truly inductive split.

**Assuming all relation vocabularies transfer.** ULTRA pre-trains on Freebase, WordNet, and Codex. If your target KG has extremely unusual relation patterns absent from all three — for instance, a KG encoding temporal dependencies between manufacturing operations — zero-shot transfer may perform worse than a simple heuristic. Structural fingerprints generalize when structural patterns generalize; if the target domain has unique topological regularities, fine-tuning is essential.

**Underestimating fine-tuning efficiency.** A common misconception is that ULTRA requires thousands of fine-tuning triples to improve. In practice, 100–500 target-KG triples and 3–5 training epochs typically recover most of the gap between zero-shot and fully supervised performance. This is because ULTRA is not learning entity embeddings during fine-tuning — it is adapting its structural context extractor, which converges quickly when the structural patterns are broadly similar to the training distribution.

**Confusing InGram with GraIL.** GraIL (Teru et al. 2020) represents each entity by the subgraph structure of its local enclosing neighborhood — a fixed-radius enclosing subgraph that is extracted and fed to a GNN. This is a different abstraction from InGram's relation graph. GraIL is entity-equivariant but not relation-equivariant: it still requires a fixed relation vocabulary because the relational GNN over the enclosing subgraph uses relation IDs as features. InGram's relation graph explicitly breaks relation-ID dependence.

---

## 14.12 Exercises

The following 12 exercises span all six levels of Bloom's taxonomy.

**Remember**

1. Define "inductive KG reasoning" and explain how it differs from transductive link prediction in one sentence each.
2. List the four edge types used in InGram's relation graph and describe what structural co-occurrence each one captures.

**Understand**

3. Explain why TransE is not entity-equivariant. Use the formal definition of equivariance and show which step in the TransE computation breaks equivariance when entity IDs are permuted.
4. Describe in plain language why pre-training on FB15k-237 and WN18RR would likely transfer better to a general encyclopedic KG than to a domain-specific legal citation network.

**Apply**

5. Given the following 6 triples over 4 entities and 2 relation types, manually construct the relation graph nodes and at least 3 edges with their correct edge types:
   \( (A, r_1, B), (A, r_1, C), (B, r_2, D), (C, r_2, D), (D, r_1, A), (B, r_2, C) \).
6. Write pseudocode (not full Python) for a function `inductive_mrr(model, test_triples, train_triples)` that evaluates MRR on a fully inductive split, correctly applying filtered evaluation using only inference-graph triples.

**Analyze**

7. The GraIL model uses local subgraph structure to represent entities and achieves Hits@10 = 64.2 on inductive FB15k-237 v1. ULTRA achieves 81.3 zero-shot. Analyze two specific structural reasons why ULTRA outperforms GraIL, drawing on the double equivariance framework.
8. Compare InGram and ULTRA on three dimensions: (a) training data requirements, (b) the level of inductiveness they achieve, and (c) computational cost at inference time. Which would you prefer for a streaming KG that receives 1,000 new entities per hour? Justify your choice.

**Evaluate**

9. A colleague argues: "Since ULTRA achieves 86.2 Hits@10 when fine-tuned, and supervised models on the transductive split achieve ~88 Hits@10, the gap between transductive and inductive performance is negligible." Critically evaluate this claim. What additional information would you need to assess it fairly?
10. Evaluate the following experimental design: a researcher trains ULTRA on FB15k-237 + WN18RR, then tests zero-shot on Codex-M. They compare against a TransE baseline trained *transductively* on Codex-M. Identify all confounds in this comparison and propose a fairer evaluation protocol.

**Create**

11. Design a **relation graph augmentation strategy** for KGs that have very few triples (fewer than 500), where the standard co-occurrence graph is too sparse to support meaningful GNN propagation. Propose at least two augmentation techniques, explain the structural justification for each, and describe how you would validate whether the augmentation improves inductive generalization.
12. Propose a training curriculum for ULTRA that dynamically adjusts which source KGs contribute to each training batch based on their structural similarity to a target KG. Describe the similarity metric you would use, how the curriculum would be implemented, and which hyperparameters require tuning.

---

## 14.13 Further Reading

The following papers are recommended for readers who wish to deepen their understanding of inductive KG reasoning and graph foundation models.

**ULTRA: Towards Foundation Models for KG Reasoning** — Galkin, Yuan, Mostafa, Tang, Hamilton (2023). The primary paper for this chapter. ULTRA establishes the multi-KG pre-training + double equivariance + NBFNet backbone combination that enables zero-shot transfer to 43 KGs. The paper includes extensive ablations showing the contribution of each design choice, including the number of source KGs, the number of relation graph edge types, and the number of NBFNet propagation steps. Appendix A provides a clean theoretical derivation of the double equivariance requirement.

**InGram: Inductive KG Embedding via Relation Graphs** — Lee, Lee, Seo (2023). The foundational paper on relation graph construction for inductive KG embedding. Particularly recommended for its empirical analysis of which edge types in the relation graph carry the most useful structural information (Section 5.2), which provides intuition for when the four-type construction can be simplified without major performance loss.

**Double Equivariance for Inductive KG Link Prediction** — Guo, Namata, Faloutsos (2023). The theoretical companion to ULTRA. Provides formal proofs that entity equivariance and relation equivariance are necessary conditions for cross-KG generalization, and shows that existing inductive methods satisfy varying subsets of these conditions. Recommended for readers interested in the mathematical foundations.

**Inductive Relation Prediction by Subgraph Reasoning (GraIL)** — Teru, Denis, Hamilton (2020). The paper that introduced the inductive benchmark splits used throughout this chapter and established GNN-based subgraph reasoning as the dominant approach. Reading GraIL alongside ULTRA clarifies how much the field has progressed in three years: from 64 Hits@10 (GraIL supervised) to 81 Hits@10 (ULTRA zero-shot).

**NodePiece: Compositional and Parameter-Efficient Representations of Large Knowledge Graphs** — Galkin, Denis, Pautasso, Shu, Siekiera (2021). NodePiece addresses the entity-scalability problem by representing each entity as a set of anchor tokens — a small subset of entities that serve as structural landmarks. This tokenized representation is inductive for new entities (new entities can be represented by their distances to anchors) but still requires a fixed relation vocabulary. Useful background for understanding the spectrum between purely transductive and fully inductive approaches.

**PRODIGY: Enabling In-Context Learning Over Graphs** — Huang, Cao, Goldsmith, Velickovic, Grohe, Leskovec, Jegelka (2023). Extends the foundation model concept by framing graph tasks as in-context learning: the model receives a few labeled examples as "context" alongside the query graph and performs the task without gradient updates. Demonstrates that the structural inductive biases learned during pre-training generalize to few-shot settings in a manner analogous to LLM in-context learning.

**Towards Foundation Models for KG Reasoning (Workshop Version)** — Galkin, Yuan, Mostafa, Tang, Hamilton (2023). An earlier, shorter presentation of the ULTRA ideas, useful for getting the core intuitions without the full technical apparatus. Recommended as a first read before the main ULTRA paper.

---

!!! mascot-celebration "You've Crossed the Transductive Boundary"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating">
    Part 3 of this textbook — knowledge graph embeddings, multi-hop reasoning, and now foundation models — has taken you from memorizing individual facts (TransE) to reasoning over chains of unknown facts (Query2Box, NBFNet) to generalizing across entirely unknown knowledge graphs (ULTRA). The unifying thread is a shift from *identity-based* representations (entity ID → embedding) toward *structure-based* representations (entity neighborhood → embedding). This shift mirrors what happened in NLP when fixed word vectors gave way to contextual representations: the meaning of a word is not intrinsic to the word, but emerges from its context. For knowledge graphs, the "context" is the relational neighborhood. In the next part of the textbook, we'll extend this structural perspective to heterogeneous graphs, recommender systems, and beyond.
