<!--
Original MicroSim specification for ch04-node2vec-walk (node2vec Biased Random Walk Explorer).
Excluded from the built site via exclude_docs (**/sim-spec.md).
Kept so the spec survives after the <details> block is removed from the chapter.
-->

# Spec: node2vec Biased Random Walk Explorer

- sim-id: ch04-node2vec-walk
- chapter: 04-node-embeddings
- bloom: Apply
- library: p5.js

## Original specification block

<details markdown="1">
<summary>Interactive node2vec Walk Visualizer with p/q Parameter Control</summary>
Type: MicroSim
**sim-id:** ch04-node2vec-walk<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective (Understanding → Applying — Bloom's Levels 2–3):** Students manipulate the return parameter \(p\) and in-out parameter \(q\) and observe how walk trajectories change from local BFS-like exploration to global DFS-like exploration, building intuition for the correspondence between walk bias and the type of graph structure captured in the resulting embeddings.

**Canvas:** 900 × 540 px, responsive (resizes on window resize). Left panel (620 px): graph canvas with animated walk path. Right panel (280 px): controls and statistics.

**Default graph:** The Zachary Karate Club graph (34 nodes, 78 edges), drawn using a force-directed layout precomputed at initialization and fixed thereafter (no physics during interaction). Nodes colored by faction: steelblue (Mr. Hi's group) and crimson (John A.'s group).

**Walk rendering (left panel):**
- The current walk is displayed as an animated path: each visited node is drawn in gold; the most recently visited node has a pulsing glow (3-frame animation at 10 fps).
- The walk trace is shown as a gradient line from dark indigo (start) to bright gold (current position), drawn over the graph edges.
- The previous node \(s_{\text{prev}}\) is marked with an orange halo to indicate which transitions count as "return" (d=0), "shared neighbor" (d=1), or "outward" (d=2).
- After each step, the three candidate next nodes are briefly highlighted: green if d=1 (neighbor of both \(s_{\text{prev}}\) and current), blue if d=0 (return), orange if d=2 (outward). The width of the arrow to each candidate encodes its transition probability.

**Controls (right panel):**
- **p slider:** Range [0.1, 4.0], default 1.0, step 0.1. Label: "Return parameter p = 1.0".
- **q slider:** Range [0.1, 4.0], default 1.0, step 0.1. Label: "In-out parameter q = 1.0".
- **Walk length slider:** Range [5, 80], default 30, step 5. Label: "Walk length L = 30".
- **Step button:** Advances the walk by one step using the current p/q values. Shows the chosen transition and its probability.
- **Auto Walk button:** Runs the walk at 3 steps/second until walk length reached, then stops. Label toggles "Auto Walk" / "Pause".
- **Reset button:** Clears the walk trace, selects a new random start node, resets to step 0.
- **Start node selector:** Dropdown of node IDs 0–33, default random.

**Statistics panel (right panel, below controls):**
- "Current step: 0 / 30"
- "Fraction of time in same faction: XX%" — counts what fraction of visited nodes share the faction label with the start node. High fraction = homophily captured; random = 50%.
- "Farthest node reached: X hops from start" — maximum graph distance from start node among all visited nodes.
- Walk mode inference: shows "BFS-biased" if p < 1 and q > 1, "DFS-biased" if p > 1 and q < 1, "Balanced" otherwise.

**Interaction — node click:** Clicking any node selects it as the start node, resets the walk from that node with the current p/q settings.

**Interaction — edge hover tooltip:** Hovering over an edge shows its transition probability under the current p/q settings from the current walk position (if applicable): "Edge (3→5): π = 0.24 (d=1 neighbor)".

**Implementation notes:** Pre-compute the Karate Club adjacency list as a hard-coded JavaScript constant. For each walk step, compute the unnormalized transition weights for all neighbors of the current node given the previous node, normalize, and sample using a weighted random choice. The p/q sliders should update transition probabilities immediately without restarting the walk. Use p5.js `frameRate(10)` for the pulsing glow animation.

</details>
