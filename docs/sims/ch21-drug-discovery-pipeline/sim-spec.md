<!--
Original MicroSim specification for ch21-drug-discovery-pipeline.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Drug Discovery GNN Pipeline

- sim-id: ch21-drug-discovery-pipeline
- chapter: 21-generative-models
- bloom: Understand
- library: p5.js

## Original specification

<details markdown="1">
<summary>Three-Stage Drug Discovery Pipeline — Interactive Workflow</summary>
Type: workflow-diagram
**sim-id:** ch21-drug-discovery-pipeline
**Library:** p5.js
**Status:** Specified

Interactive three-stage pipeline diagram showing how GNNs contribute to drug discovery. Canvas: 800×480px, responsive to window resize events.

**Layout:** Three vertically stacked panels, each representing one pipeline stage, connected by downward arrows. Left edge: stage labels (Stage 1, Stage 2, Stage 3). Each panel: 780×120px with rounded corners.

**Panel 1 — Target Identification (blue tint):**
- PPI network visualization: small graph of ~10 nodes (colored circles) with several edges; 3 nodes highlighted in red (disease-pathway proteins)
- Label: "GNN Node Classification on PPI Network"
- Caption: "Identifies high-centrality disease target proteins"
- Hovering a highlighted node shows tooltip: "EGFR: 94% disease-pathway probability; betweenness centrality: 0.72"

**Panel 2 — Hit Generation (green tint):**
- Molecule sketch: a simple 6-node ring (benzene-like) with atom type labels
- Arrow from a "Generative Model" box (GCPN/DiGress) pointing to molecule
- Label: "GCPN / DiGress — Conditional Molecule Generation"
- Caption: "Generates candidate molecules maximizing predicted binding affinity"
- Hovering the molecule shows tooltip: "QED: 0.84, SA: 2.1, Predicted IC50: 23nM"

**Panel 3 — Safety Screening (yellow tint):**
- DDI network: small graph of ~8 drug nodes; the new candidate drug shown as a flashing orange node; edges to existing drugs shown in red (predicted interactions)
- Label: "GNN Link Prediction on DDI Network"
- Caption: "Flags dangerous drug-drug interactions before clinical testing"
- Hovering a red edge shows tooltip: "Predicted interaction: candidate + warfarin → increased bleeding risk (confidence: 0.88)"

**Arrows between panels:** Each downward arrow is clickable; clicking opens a tooltip explaining what information flows between stages (e.g., "Target protein identity and binding pocket features pass from Stage 1 to Stage 2 as conditioning signal").

**Controls:** "Step Through Pipeline" button animates through stages sequentially, highlighting the active stage.

**Learning objective (Understanding — Bloom's Taxonomy):** Students can trace how graph structure and GNN predictions flow from target identification to molecule generation to safety screening, reinforcing the connection between discriminative and generative GNN roles.

Implementation: p5.js with DOM button and tooltip divs. Mouse-over callbacks on all interactive elements. Responsive via `windowResized()`.
</details>
