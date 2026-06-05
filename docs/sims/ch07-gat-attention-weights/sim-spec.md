<!--
Original MicroSim specification for ch07-gat-attention-weights (GAT Attention Weight Visualizer).
Excluded from the built site via exclude_docs (**/sim-spec.md).
-->

# Spec: GAT Attention Weight Visualizer

- sim-id: ch07-gat-attention-weights
- chapter: 07-gnn-design-space
- bloom: Apply
- library: p5.js

## Original specification block

<details markdown="1">
<summary>Interactive GAT attention weight visualization on a small graph</summary>

**sim-id:** ch07-gat-attention-weights<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Applying (Bloom's Level 3) — Students apply their understanding of attention coefficients by manipulating node features and observing how attention weights change, connecting the mathematical definition to visual intuition.

**Canvas:** 720×500px, responsive to window resize. A 12-node graph (Karate Club subgraph: nodes 0–11) drawn with fixed force-layout positions.

**Initial state:**
- Nodes rendered as circles (radius 14px), colored by their class label (2 classes: blue/red for the two Karate Club factions)
- Edges rendered as thin gray lines
- No focal node selected; instruction text: "Click a node to inspect its attention pattern"

**Interaction — click focal node:**
When the user clicks a node \( i \):
1. Highlight node \( i \) with a white border ring
2. For each neighbor \( j \in \mathcal{N}(i) \):
   - Compute a synthetic attention score: \( \alpha_{ij} = \text{softmax}(\cos(\mathbf{h}_i, \mathbf{h}_j)) \) where \( \mathbf{h}_v \) is the node's 2D position vector (proxy for features)
   - Render the edge \( (i,j) \) with width proportional to \( \alpha_{ij} \) (range: 1px–12px)
   - Color the edge on a cold-to-warm gradient: low attention = cool blue (#93c5fd), high attention = warm orange (#f97316)
   - Display \( \alpha_{ij} \) as a label floating midway along the edge (2 decimal places)
3. Render non-neighbor edges in light gray (opacity 0.2)
4. Show a sidebar panel:
   - "Focal node: [id]" 
   - Table of neighbors sorted by \( \alpha_{ij} \) descending: columns = Neighbor ID, Attention Weight

**Head selector:** a row of 4 toggle buttons labeled "Head 1", "Head 2", "Head 3", "Head 4" below the canvas. Each head uses a different synthetic attention function (e.g., different random projections of the position vectors) to simulate independent attention patterns. Selecting a head re-renders edge widths and colors for that head.

**Reset button:** clears focal node selection.

**Hover tooltip on any edge:** "Edge (\( i \), \( j \)) | \( \alpha_{ij} \) = [value] | Head: [k]"

**Implementation notes:** Initialize 4 random 2×2 projection matrices at load time (fixed seed for reproducibility). For each head \( k \), attention score \( e_{ij}^{(k)} = \tanh(\mathbf{p}_k^T [\mathbf{pos}_i \| \mathbf{pos}_j]) \) where \( \mathbf{p}_k \) is the projection vector and \( \mathbf{pos}_v \) is node \( v \)'s 2D canvas position. Apply softmax over each focal node's neighborhood. Ensure canvas is readable at 480px width (collapse sidebar below canvas on narrow screens).

</details>
