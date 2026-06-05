<!--
Original MicroSim specification for ch08-gnn-training-dynamics (GNN Training Dynamics MicroSim).
Excluded from the built site via exclude_docs (**/sim-spec.md).
-->

# Spec: GNN Training Dynamics MicroSim

- sim-id: ch08-gnn-training-dynamics
- chapter: 08-gnn-training
- bloom: (see spec body)
- library: p5.js

## Original specification block

<details markdown="1">
<summary>ch08-gnn-training-dynamics — Interactive Training Curve Visualizer</summary>

**sim-id:** ch08-gnn-training-dynamics  
**Library:** p5.js  
**Status:** scaffold

**Purpose:** Visualize how training loss and validation accuracy diverge over epochs for GCNs of different depths, making the B₁ over-smoothing feedback loop tangible.

**Layout:**

- **Left panel**: dual y-axis chart. Left axis: training loss (red curve). Right axis: validation accuracy (blue curve). X-axis: epochs 0–300. A vertical dashed line marks the epoch of peak validation accuracy (best checkpoint).
- **Right panel**: schematic graph with 20 nodes colored by predicted class. Colors update every 10 epochs as the model trains.

**Controls:**

- **Depth slider** (1–16 layers): changes GCN depth; re-runs the simulation automatically.
- **Skip connections toggle**: switches between plain GCN and GCN + residual; both curves are overlaid in different line styles.
- **Dropout slider** (0.0–0.8): adjust regularization strength and watch how it affects the accuracy peak.
- **Run / Pause / Reset buttons**: control the animation.

**Interaction:**

- Hover anywhere on the chart → tooltip shows epoch number, training loss, and validation accuracy at that point.
- When early stopping fires, the chart displays a "STOP (patience=50)" annotation at the termination epoch.
- Click a node in the graph panel → highlight which training nodes influenced its embedding via K-hop neighborhood display.

</details>
