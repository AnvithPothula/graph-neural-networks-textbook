<!--
Original MicroSim specification for ch24-ssl-concept-map.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: DGI vs. Contrastive Learning — Concept Map

- sim-id: ch24-ssl-concept-map
- chapter: 24-advanced-gnn-topics
- bloom: Analyze
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive concept map comparing self-supervised learning paradigms</summary>
Type: MicroSim
**sim-id:** ch24-ssl-concept-map<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Students compare and contrast DGI and graph contrastive learning by interacting with a concept map that reveals how each method satisfies the three core SSL properties (Bloom's: Analyzing — comparing structural relationships between methods).

**Canvas:** 800 × 480 px, responsive to window resize. White background with a subtle radial gradient.

**Layout:** A concept map with three tiers:

Top tier — "Three Core SSL Properties" (three rounded rectangles):
- "Invariance to Augmentation" (light blue)
- "Discrimination Across Nodes" (light green)  
- "Structural Context" (light orange)

Middle tier — "SSL Methods" (two oval nodes):
- "DGI" (purple)
- "Graph Contrastive Learning" (teal)

Bottom tier — "Mechanisms" (five small boxes):
- "Graph Corruption" → DGI
- "Mutual Information (JSD)" → DGI
- "Edge Dropout / Feature Masking" → GCL
- "NT-Xent Loss" → GCL
- "Two-View Encoder" → GCL

Edges connect methods to the properties they satisfy:
- DGI → Discrimination ✓, Structural Context ✓, Invariance to Augmentation (partial — via corruption)
- GCL → Invariance ✓, Discrimination ✓, Structural Context ✓

**Interactions:**
- Clicking a method node (DGI or GCL) highlights all connected property and mechanism nodes, fades others
- Clicking a property node highlights which methods satisfy it (fully or partially)
- Clicking a mechanism node shows a tooltip with a one-paragraph explanation of that mechanism
- Clicking a property node shows a tooltip with a one-paragraph definition
- "Reset" button restores default state

**Visual style:** color-coded by tier, click to highlight sub-graph, animated edge pulses when a connection is highlighted

**Responsiveness:** nodes reposition to fill available canvas width via proportional scaling
</details>
