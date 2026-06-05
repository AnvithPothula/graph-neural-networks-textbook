<!--
Original MicroSim specification for ch23-llm-gnn-pipeline.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: LLM+GNN Pipeline — Text-to-Prediction

- sim-id: ch23-llm-gnn-pipeline
- chapter: 23-llm-gnn
- bloom: Understand
- library: p5.js

## Original specification

<details markdown="1">
<summary>LLM+GNN step-by-step pipeline MicroSim</summary>
Type: MicroSim
**sim-id:** ch23-llm-gnn-pipeline<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Students trace how a text-described node travels through the LLM+GNN pipeline from raw text to final classification (Bloom's: Understanding — explaining how multi-step pipelines compose).

**Canvas:** 900 × 480 px, responsive to window resize. White background with a subtle grid.

**Layout:** Five vertical stages arranged left-to-right with connecting animated arrows:
1. **Stage 1 — Raw Text** (leftmost): A blue rounded card showing a short paper abstract (50 words, truncated with ellipsis). Label: "Node Text (Abstract)"
2. **Stage 2 — LLM Tokenizer**: Shows the abstract being split into colored token chips (each word becomes a colored pill). Label: "Tokenize"
3. **Stage 3 — LLM Encoder**: An animated transformer block (stacked orange rectangles representing attention heads). Output: a single green vector bar labeled "e_v ∈ ℝ³⁸⁴". Label: "LLM Encoder"
4. **Stage 4 — GNN Aggregation**: Shows the query node (green circle) surrounded by 3 neighbor circles (blue), with animated arrows flowing inward labeled "+neighbor". The aggregated vector grows wider. Label: "GNN Aggregation"
5. **Stage 5 — Output**: A bar chart showing predicted class probabilities (top-3 categories). The highest bar is highlighted in gold. Label: "Predicted Label"

**Controls (bottom panel):**
- "Step →" button: advances one stage at a time, animating the transition (tokens animate from text, vector animates from tokenizer, etc.)
- "Auto Play" button: runs all stages with 1.5s delay between each
- "Reset" button: returns to Stage 1
- Dropdown: select one of three abstract examples (Computer Vision, NLP, Graph ML) — switching changes the text in Stage 1 and the final predicted probabilities in Stage 5

**Interactions:**
- Clicking any stage card shows a tooltip explaining what that stage computes and what its output represents
- Hovering a token chip (Stage 2) shows its token ID
- Hovering the vector bar (Stage 3) shows its dimensionality and the model used
- Clicking a bar in Stage 5 shows the class name and confidence score

**Colors:** blue for raw text, orange for LLM components, green for graph components, gold for final prediction

**Responsiveness:** all five stage cards scale proportionally; at widths below 600px, collapse to a 2×3 grid with Stage 5 spanning the full width
</details>
