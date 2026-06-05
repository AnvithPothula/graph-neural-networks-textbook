<!--
Original MicroSim specification for ch13-kg-query-traversal.
Excluded from the built site via exclude_docs (**/sim-spec.md).
-->

# Spec: ch13-kg-query-traversal

<details markdown="1">
<summary>MicroSim: Interactive KG Multi-Hop Query Traversal</summary>

#### MicroSim: ch13-kg-query-traversal

**Concept:** Visualize how Query2Box grows and intersects boxes as it resolves a multi-hop query over a toy KG.

**Controls:**
- Query type selector: 1p / 2p / 2i
- Anchor entity selector (click a node in the KG graph)
- Relation dropdowns for each hop
- Step button: advance one projection step and animate the box update

**p5.js implementation notes:**
- Render a toy KG with 8 entities as labeled circles connected by colored directed edges
- Below the graph, show a 2D slice of the embedding space as a scatter plot with entity points
- On each "Step" click: animate the current box (blue rectangle), shift its center and widen its offset, draw the new box (green rectangle)
- For 2i: show two chains projecting independently to two boxes, then animate the intersection (gold region = overlap)
- Entities inside the final answer box are highlighted as candidates in both views

**Pedagogical goal:** Students observe that (1) each projection shifts and widens the box, (2) intersection narrows the feasible region to the overlap of two query chains, and (3) entities inside the final box are exactly those satisfying all path conditions simultaneously.

```html
<iframe
  src="../../sims/ch13-kg-query-traversal/main.html"
  width="100%"
  height="520px"
  style="border:none;">
</iframe>
```

</details>
