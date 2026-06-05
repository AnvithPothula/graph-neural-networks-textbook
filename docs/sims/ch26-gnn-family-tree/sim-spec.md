<!--
Original MicroSim specification for ch26-gnn-family-tree.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: GNN Architecture Family Tree

- sim-id: ch26-gnn-family-tree
- chapter: 26-conclusion
- bloom: Create
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive taxonomy of GNN architectures as an explorable directed graph</summary>
Type: MicroSim
**sim-id:** ch26-gnn-family-tree<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Students can locate any major GNN architecture within the design space taxonomy, trace its intellectual lineage through predecessor models, and identify which textbook chapter covers it in depth (Bloom's: Analyzing — situating architectures within a broader classification system).

**Canvas:** 960 × 600 px, responsive. Background: #f8f8fc with a subtle grid. Force-directed layout with spring forces between connected nodes and repulsion between all nodes.

**Graph content:** Approximately 25 nodes representing major architectures:
- GCN (2017), GraphSAGE (2017), GAT (2018), GIN (2019), SGC (2019), APPNP (2019)
- PNA (2020), NGNN (2021), GraphSAINT (2020), Cluster-GCN (2019)
- Graphormer (2021), SAN (2021), GPS (2022)
- TransE (2013), RotatE (2019), Query2Box (2020), NBFNet (2021), ULTRA (2023)
- HAN (2019), HGT (2020), RGCN (2018)
- GraphVAE (2018), GraphRNN (2018), GDSS (2022)
- TGN (2020), TGAT (2020)
- DGI (2019), GraphCL (2020), PRODIGY (2023), OFA (2024)

**Edges:** Directed arrows with labels: "extends", "inspired by", "special case of", "uses". Example: GIN → GCN (label: "extends"), GPS → Graphormer (label: "inspired by").

**Node color coding by design dimension (Dimension 9 from §26.2):**
- Blue: Homogeneous GNNs (GCN, GAT, GIN, …)
- Orange: Knowledge graph models (TransE, RotatE, …)
- Purple: Heterogeneous graph models (HAN, HGT, …)
- Green: Temporal graph models (TGN, TGAT)
- Red: Generative models (GraphVAE, GraphRNN)
- Teal: Self-supervised / foundation models (DGI, GraphCL, PRODIGY, OFA)

**On node click:** Pop-up card (200 × 180 px) showing:
- Model name and year
- Key innovation (one sentence)
- Design dimensions it touches (from §26.2.1, listed as colored chips)
- Textbook chapter link (e.g., "Chapter 6" as a highlighted badge)
- "Ancestors" and "Descendants" counts

**Filters (controls bar at top):**
- "Color by" toggle: dimension / year / part-of-textbook
- "Filter by dimension" dropdown: shows only models touching a selected design dimension
- Year range slider: 2013–2024
- "Highlight lineage" toggle: when a node is selected, dims all non-ancestor/descendant nodes
- "Reset" button: clears all filters and re-centers graph

**Responsiveness:** below 600px width, shows a simplified list view with the same click-to-expand cards; force layout hides on small screens.
</details>
