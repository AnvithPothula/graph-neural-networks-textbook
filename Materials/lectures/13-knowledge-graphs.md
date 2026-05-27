# Lecture 13: Knowledge Graph Embeddings

## Overview

Knowledge graphs store world knowledge as (head, relation, tail) triples. KG embedding methods learn vector representations for entities and relations that enable reasoning, completion, and question answering.

**Slides (2025):** http://web.stanford.edu/class/cs224w/slides/10-kg.pdf
**Slides (2021):** https://snap.stanford.edu/class/cs224w-2021/slides/10-kg.pdf

---

## Key Concepts

### Knowledge Graphs
- **Triple:** (h, r, t) = (head entity, relation, tail entity)
  - Example: (Freebase, createdBy, Google), (Einstein, bornIn, Germany)
- **KG completion:** predict missing triples — which (h, r, ?) or (?, r, t) are true?
- **Scale:** Freebase (40M entities), Wikidata (90M+ entities), NELL, ConceptNet, Yago

### Embedding-Based Methods
All methods learn embeddings **h, r, t ∈ ℝ^d** and define a scoring function f(h, r, t).
High score → triple is likely true. Train with:
- Positive triples from KG
- Negative triples by corrupting (h, r, t) → (h', r, t) or (h, r, t')

### TransE (Bordes et al., 2013) — Translational Model
- **Idea:** relation r is a translation from h to t in embedding space
- **Score:** f(h, r, t) = -||h + r - t||
- Simple and efficient; captures 1-to-1 relations well
- **Fails:** symmetric relations (r = r⁻¹), 1-to-N relations, N-to-1 relations

### TransR (Lin et al., 2015)
- Projects entities into a relation-specific space: h_r = M_r · h, t_r = M_r · t
- Handles 1-to-N relations better than TransE

### DistMult (Yang et al., 2015) — Bilinear Model
- **Score:** f(h, r, t) = h^T · diag(r) · t = Σ_i h_i · r_i · t_i
- Efficient; works well for symmetric relations
- Cannot model antisymmetric relations

### ComplEx (Trouillon et al., 2016) — Complex Embeddings
- Uses complex-valued embeddings: h, r, t ∈ ℂ^d
- **Score:** Re(h^T · diag(r) · t̄) where t̄ is complex conjugate
- Handles asymmetric, antisymmetric, and symmetric relations
- Equivalent to a bilinear model over complex space

### RotatE (Sun et al., 2019) — Rotational Model
- Relations are rotations in complex space: t = h ∘ r where |r_i| = 1
- **Score:** -||h ∘ r - t||
- Can model: symmetry (r ∘ r = I), antisymmetry, inversion (r⁻¹), composition

### Training KG Embeddings
- **Loss:** margin-based or binary cross-entropy with negative sampling
- **Negative sampling:** corrupt head or tail randomly (or filtered)
- **Filtered evaluation:** rank all entities, ignore true triples during evaluation

### Relation Patterns Comparison

| Pattern | TransE | DistMult | ComplEx | RotatE |
|---|---|---|---|---|
| Symmetry | ❌ | ✅ | ✅ | ✅ |
| Antisymmetry | ✅ | ❌ | ✅ | ✅ |
| Inversion | ✅ | ❌ | ✅ | ✅ |
| Composition | ✅ | ❌ | ❌ | ✅ |

## Key Papers
- [TransE (Bordes et al., 2013)](https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
- [TransR (Lin et al., 2015)](https://linyankai.github.io/publications/aaai2015_transr.pdf)
- [DistMult (Yang et al., 2015)](https://arxiv.org/pdf/1412.6575.pdf)
- [ComplEx (Trouillon et al., 2016)](https://arxiv.org/pdf/1606.06357.pdf)
- [RotatE (Sun et al., 2019)](https://arxiv.org/pdf/1902.10197.pdf)

## Textbook Chapter Notes

Maps to **Chapter 13: Knowledge Graph Embeddings**.
- Geometric intuition diagrams for each method
- MicroSim: 2D visualization of TransE learning to separate valid from invalid triples
- Code: train TransE on FB15k-237 with PyKEEN library
