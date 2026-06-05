<!--
Original MicroSim specification for ch10-gin-gcn-expressiveness (GIN vs. GCN Expressiveness MicroSim).
Excluded from the built site via exclude_docs (**/sim-spec.md).
-->

# Spec: GIN vs. GCN Expressiveness MicroSim

- sim-id: ch10-gin-gcn-expressiveness
- chapter: 10-powerful-gnns
- bloom: (see spec body)
- library: p5.js

## Original specification block

<details markdown="1">
<summary>ch10-gin-gcn-expressiveness — Interactive Expressiveness Demonstrator</summary>

**sim-id:** ch10-gin-gcn-expressiveness  
**Library:** p5.js  
**Status:** scaffold

**Purpose:** Show the expressive difference between GCN (mean aggregation) and GIN (sum aggregation) on two classic failure cases: the K₃,₃ / prism graph pair (both regular, WL-indistinguishable) and a pair of graphs that differ only in a triangle.

**Layout:**

- **Left panel**: Graph A with node embeddings shown as colored circles; color encodes the current embedding's first principal component (updated in real time as epochs advance).
- **Right panel**: Graph B, same display.
- **Bottom panel**: scatter plot of the final node embeddings in 2D (PCA projection). Distinct clusters = model can distinguish; overlapping clusters = model cannot.

**Controls:**

- **Model selector**: GCN (mean) vs. GIN (sum). Switches architecture without reloading the graphs.
- **Graph pair selector**: K₃,₃ vs. prism | triangle vs. no-triangle | regular vs. irregular.
- **Epoch slider**: 0–200 epochs. Drag to see how embeddings evolve during training.
- **Run animation**: animates embedding evolution from epoch 0 to 200.

**Interaction:**

- Click any node → highlight which nodes in the other graph receive the same embedding (if any).
- At convergence, a banner appears: "DISTINGUISHABLE" (green) or "INDISTINGUISHABLE" (red) based on whether any node embedding pair differs.

</details>
