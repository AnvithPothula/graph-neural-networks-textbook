<!--
Original MicroSim specification for ch15-hetero-graph-explorer.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Typed Node and Edge Explorer

- sim-id: ch15-hetero-graph-explorer
- chapter: 15-heterogeneous-graphs
- bloom: Apply
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive heterogeneous graph with toggleable node/edge types and meta-path highlighting</summary>
Type: MicroSim
**sim-id:** ch15-hetero-graph-explorer<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Understand and Apply — students manipulate a heterogeneous graph by toggling node types on/off and selecting meta-paths to see which nodes become reachable. Bloom's: Understanding (recognizing how meta-paths define neighborhoods), Applying (constructing meta-path neighborhoods for a query node).

**Canvas:** 900×520px, responsive to window resize.

**Graph layout:**
- 15 paper nodes (blue circles, medium), 8 author nodes (orange circles, small), 5 field nodes (green squares), and 3 institution nodes (grey diamonds)
- Fixed spring-layout positions computed on canvas setup. Nodes do not animate after initial placement.
- Edges drawn as thin grey lines by default; each edge colored by meta-relation when "Show Edge Types" is toggled on: blue for paper→cites→paper, orange for author→writes→paper, green for paper→has\_topic→field, grey for author→affiliated\_with→institution

**Controls panel (right side, 220px wide):**
- **Node type toggles** (4 checkboxes): Paper / Author / Field / Institution — toggling OFF removes those nodes and all incident edges from the display
- **Meta-path selector** (dropdown or 4 radio buttons): None / PP / APA / PAP / PFP — selecting a meta-path highlights the selected query node's meta-path neighborhood in gold, with the intermediate nodes shown at reduced opacity
- **Query node selector** (click directly on graph): click any paper node to make it the query node; it turns red with a white border
- **Show Edge Types** toggle: colorizes edges by meta-relation

**Behavior:**
- Default state: all node types visible, no meta-path selected, no query node
- Click paper node → it becomes the query node (red)
- Select meta-path (e.g., APA = Author→writes→Paper←writes←Author): all authors who wrote the query paper are highlighted in bright orange; all other papers co-authored with the query paper by any of those authors are highlighted in gold; all other nodes fade to 20% opacity
- Tooltip on hover over any node: shows node type, ID, and degree
- Tooltip on hover over any edge: shows meta-relation triple (source\_type, relation, target\_type)
- When a node type is toggled off, the toggle immediately updates the visible graph; meta-path results update accordingly

**Color scheme:** dark background (#1a1a2e). Paper nodes: #4fc3f7 (blue). Author nodes: #ffb74d (orange). Field nodes: #81c784 (green). Institution nodes: #90a4ae (grey). Query node: #ef5350 (red). Highlighted neighborhood: #ffd54f (gold). Faded opacity: 0.2.

**Educational note displayed below canvas:** "Select a meta-path to see which nodes become the neighborhood of the highlighted (red) paper node. Notice how different meta-paths define different notions of similarity."

Implementation: p5.js with pre-computed adjacency lists for all four meta-relations. Meta-path neighbor computation is done in JavaScript on click, not pre-materialized, to demonstrate dynamic expansion.
</details>
