<!--
Original MicroSim specification for ch25-multi-hop-reasoning.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Multi-Hop KG Reasoning Agent

- sim-id: ch25-multi-hop-reasoning
- chapter: 25-agents-and-graphs
- bloom: Apply
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive multi-hop reasoning agent traversing a knowledge graph</summary>
Type: MicroSim
**sim-id:** ch25-multi-hop-reasoning<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Students watch an agent execute the query-retrieve-reason-refine loop step by step on a small knowledge graph, building intuition for how each hop narrows toward the answer (Bloom's: Applying — tracing an agent's reasoning path through a structured knowledge base).

**Canvas:** 960 × 540 px, responsive. Dark background (#0f0f1a). Two panels.

**Left panel (600px wide) — Knowledge Graph:**
Force-directed graph with 12 nodes and 18 edges. Nodes are colored by entity type:
- Person nodes: blue circles (Bahdanau, VaswaniEtAl, etc.)
- Institution nodes: green circles (UniversityMontreal, GoogleBrain, Stanford)
- Model/concept nodes: orange circles (Transformer, GAT, attention, GCN, GraphSAGE, GIN)

Edges are labeled with relationship types in small gray text. All edges are visible but unactivated at start.

**Animation per hop:**
- The seed node(s) for the current query glow gold
- Edges traversed in this hop animate with a traveling dot (gold, 0.5s)
- Newly discovered nodes light up white
- The "answer path" nodes and edges remain highlighted (green) after each hop

**Right panel (340px wide) — Agent Reasoning Log:**
A scrolling text panel showing:
- Current Query (Q_t): displayed in blue italic
- Retrieved Subgraph: list of (subject, relation, object) triples found this hop
- Reasoning: two-sentence LLM-style reasoning text
- Refined Query: next query in green
- Confidence bar: grows across hops (0 → 50% → 90% → terminates)
- Hop counter: "Hop 1 of 3 max"

**Controls:**
- "Next Hop →" button: advances one reasoning step
- "Auto Run" button: runs all hops with 2s delay between each
- "Reset" button: clears highlights and resets to hop 0
- Question dropdown: 3 preloaded multi-hop questions of increasing complexity (1-hop, 2-hop, 3-hop)

**Interactions:**
- Clicking any node shows its full list of outgoing relations in a tooltip
- Clicking any edge shows the (subject, relation, object) triple and confidence weight
- Hovering the confidence bar shows a tooltip explaining how confidence is estimated

**Responsiveness:** panels resize proportionally; at < 700px width, stack panels vertically with knowledge graph on top
</details>
