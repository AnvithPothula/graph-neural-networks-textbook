<!--
Original MicroSim specification for ch06-gcn-message-passing (GCN Message Passing Visualizer).
Excluded from the built site via exclude_docs (**/sim-spec.md).
-->

# Spec: GCN Message Passing Visualizer

- sim-id: ch06-gcn-message-passing
- chapter: 06-gnn-foundations
- bloom: Understand
- library: p5.js

## Original specification block

<details markdown="1">
<summary>Interactive GCN message passing animation on the Karate Club graph</summary>

**sim-id:** ch06-gcn-message-passing<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Understanding (Bloom's Level 2) — Students observe how a GCN layer aggregates neighbor representations into a node's new embedding, developing intuition for the receptive field expansion across layers.

**Canvas:** 760×540px, responsive to window resize. Force-directed layout of the 34-node Karate Club graph with fixed seed positions.

**Initial state:**
- All nodes rendered as gray circles (radius proportional to degree, range 10–20px)
- Each node labeled with its current scalar feature value \( h_v^{(0)} \) (initialize to degree value normalized to [0,1])
- A layer counter: "Layer: 0 / K"

**Controls (below canvas):**
- **K slider** (1–3, integer): number of GNN layers to animate
- **Select node** button: click a node to highlight it as the "focal node" — messages from its neighbors will be animated
- **Step** button: animate one layer of message passing for the focal node
- **Reset** button: return to initial state

**Animation per step:**
1. **Message phase (1.2s):** animated arrows appear along all edges incident to the focal node, flowing from neighbors toward the focal node (color: gradient from gray to gold). Neighbor nodes briefly highlight (brighter outline).
2. **Aggregation phase (0.8s):** the arrows converge to a glowing circle at the focal node. A small text overlay shows "AGGREGATE: mean of [N] neighbors".
3. **Update phase (0.8s):** the focal node's color transitions smoothly from its current shade to its updated shade (new \( h_v \) value). The layer counter increments.

**Receptive field visualization:**
- When K > 1 and the user clicks Step multiple times, concentric rings highlight the 1-hop, 2-hop, 3-hop neighborhoods of the focal node in decreasing opacity (dark blue, medium blue, light blue).
- A sidebar panel lists: "Focal node: [id] | Degree: [d] | h_v after Layer [k]: [value]"

**Hover:** hovering any node shows "Node [id] | h_v: [value] | Neighbors: [list of IDs]"

**Implementation notes:** Pre-compute Karate Club adjacency as a hardcoded edge list. Use a simple numerical mean aggregation (\( h_v^{(k)} \leftarrow \text{mean}_{u \in N(v) \cup \{v\}} h_u^{(k-1)} \)) for the animation — no learned weights needed. Ensure arrows are curved slightly for visual clarity on bidirectional edges.

</details>
