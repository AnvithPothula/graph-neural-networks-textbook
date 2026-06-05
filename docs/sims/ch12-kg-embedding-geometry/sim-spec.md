<!--
Original MicroSim specification for ch12-kg-embedding-geometry.
Excluded from the built site via exclude_docs (**/sim-spec.md).
-->

# Spec: ch12-kg-embedding-geometry

<details markdown="1">
<summary>MicroSim: Knowledge Graph TransE Embedding Space</summary>

#### MicroSim: ch12-kg-embedding-geometry

**Concept:** Visualize TransE's translational geometry in 2D, showing how training drives valid triples to satisfy h + r ≈ t.

**Controls:**
- Entity selector: choose head entity (e.g., France, Germany, Japan)
- Relation selector: choose relation (e.g., hasCapital, hasLanguage, borderedBy)
- Step button: advance one training epoch and watch entity/relation vectors update
- "Show invalid" toggle: highlight triples where ||h + r - t|| exceeds margin

**p5.js implementation notes:**
- Initialize 6 entities and 3 relations as 2D random vectors
- Each frame, compute margin loss and update vectors via gradient step
- Draw entities as colored circles; relations as colored arrows
- For each true triple, draw a dotted arrow from h → h+r, solid arrow from h → t; close arrows = low loss
- Show loss curve in corner panel

**Pedagogical goal:** Students observe that training rotates and scales the vectors until the translational constraint is approximately satisfied for all positive triples, and that symmetric-relation triples cannot be simultaneously satisfied (h + r ≈ t AND t + r ≈ h → 2r ≈ 0).

```html
<iframe
  src="../../sims/ch12-kg-embedding-geometry/main.html"
  width="100%"
  height="520px"
  style="border:none;">
</iframe>
```

</details>
