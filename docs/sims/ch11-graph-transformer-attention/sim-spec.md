<!--
Original MicroSim specification for ch11-graph-transformer-attention.
Excluded from the built site via exclude_docs (**/sim-spec.md).
-->

# Spec: ch11-graph-transformer-attention

<details markdown="1">
<summary>MicroSim: Graph Transformer Attention Heatmap</summary>

#### MicroSim: ch11-graph-transformer-attention

**Concept:** Visualize how GPS attention weights change with graph distance and layer depth.

**Controls:**
- Graph selector: benzene ring, steroid backbone, random Erdős–Rényi graph
- Layer slider: 1 → 5 GPS layers
- Attention type toggle: MPNN local vs. global MHA
- Node hover: display query vector magnitude and top-5 attended nodes

**p5.js implementation notes:**
- Initialize a small molecular graph (12–20 nodes) from a SMILES-like adjacency list
- At each layer, sample synthetic attention weights from a softmax over pairwise distances + random noise (to mimic trained behavior without a real model)
- Render attention as edge opacity; highlight the selected node's attention distribution in a side panel
- When "global MHA" is selected, show attention edges even between non-adjacent nodes (rendered as curved arcs)
- When "local MPNN" is selected, only show attention along existing graph edges

**Pedagogical goal:** Students should observe that (1) MPNN attention is strictly local (only adjacent nodes), while (2) global MHA connects distant nodes directly, and (3) deeper layers use progressively more distributed attention patterns across the entire graph.

```html
<iframe
  src="../../sims/ch11-graph-transformer-attention/main.html"
  width="100%"
  height="520px"
  style="border:none;">
</iframe>
```

</details>
