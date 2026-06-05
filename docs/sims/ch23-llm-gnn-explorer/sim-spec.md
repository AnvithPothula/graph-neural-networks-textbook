<!--
Original MicroSim specification for ch23-llm-gnn-explorer.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: LLM+GNN Pipeline Explorer (Full Version)

- sim-id: ch23-llm-gnn-explorer
- chapter: 23-llm-gnn
- bloom: Evaluate
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive step-through of text → GNN → prediction</summary>
Type: MicroSim
**sim-id:** ch23-llm-gnn-explorer<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Students build intuition for how text-attributed node features flow through the combined LLM+GNN pipeline, and what each stage contributes to the final prediction (Bloom's: Applying — tracing data through a multi-component system).

**Canvas:** 960 × 540 px, responsive. Background: #f5f5f5 with white panel cards.

**Modes (toggle at top):** Three modes selectable via tab buttons: "Encoder Mode", "Reasoner Mode", "Joint Mode". Switching modes changes which pipeline is visualized.

**Encoder Mode layout (five stages, horizontal):**
- Stage 1: "Node Text" — blue card with scrollable abstract text
- Stage 2: "LLM Encode" — shows embedding vector as a color-coded heat strip (positive = warm, negative = cool, each cell = one of 32 shown dimensions out of 384)
- Stage 3: "Neighbor Embeddings" — three smaller heat strips for three neighbor nodes
- Stage 4: "GNN Aggregate" — animated arrows from neighbors into center, shows mean of all four vectors as a new heat strip
- Stage 5: "Classification" — vertical bar chart with 5 top predicted categories

**Reasoner Mode layout:**
- Shows a text prompt being assembled: "Node abstract: [text]. Neighbors: [list]. Predict category."
- A "Send to LLM" button triggers a mock animation showing the prompt being evaluated
- Output: a mock LLM response in a speech bubble with reasoning text and predicted label

**Joint Mode layout:**
- Shows a bidirectional pipeline: text flows into LLM, LLM embeddings flow into GNN, GNN output is compared to a label, and a gradient arrow flows back through both components
- Clicking "Backprop" animates gradient colors (red = high gradient) flowing backwards through the GNN and into the LLM layers

**Controls across all modes:**
- "Select node" dropdown: pick one of five preloaded paper abstracts (from different arxiv categories)
- "Show neighbors": toggles display of neighbor nodes in the graph panel
- Step-by-step "Next →" and "Back ←" buttons

**Responsiveness:** below 700px width, display single stage per screen with swipe/click to advance
</details>
