<!--
Original MicroSim specification for ch16-lightgcn-explorer.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: User-Item Graph with Multi-Hop Propagation Visualization

- sim-id: ch16-lightgcn-explorer
- chapter: 16-recommender-systems
- bloom: Apply
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive bipartite graph showing LightGCN propagation and neighborhood discovery</summary>
Type: MicroSim
**sim-id:** ch16-lightgcn-explorer<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Understand and Apply — students observe how multi-hop paths through the bipartite graph generate recommendation candidates for a selected user, and how increasing the number of LightGCN layers expands the receptive field of collaborative signals.

**Canvas:** 900×520px, two-panel layout.

**Left panel (600px wide): Bipartite graph**
- 10 user nodes (blue circles, left column) and 15 item nodes (orange squares, right column)
- Edges drawn as light grey lines; random bipartite graph with 25 edges, pre-generated with seed for reproducibility
- Click any user node to make it the "query user" (highlighted in red)
- When a query user is selected and layer depth K is set:
  - 1-hop items (directly interacted) highlighted in bright orange
  - 2-hop users (users who also interacted with 1-hop items) highlighted in bright blue
  - 3-hop items (items interacted with by 2-hop users but not by query user) highlighted in gold — these are the K=2 recommendations
  - All other nodes fade to 20% opacity

**Right panel (280px wide): Controls and info**
- **Layer depth K** (slider 1–3): controls how many hops of propagation are shown
- **Show edge weights** (toggle): overlays normalized edge weights (1/√d_u * 1/√d_i) as edge thickness
- **Query user summary box**: shows number of 1-hop items, 2-hop users, 3-hop recommendation candidates
- **Popularity distribution bar chart** (bottom 120px of right panel): horizontal bar per item showing its degree in the full graph, sorted by degree; selected items highlighted in the bar chart when they appear in the propagation visualization

**Behavior:**
- Default: no query user, all nodes shown at full opacity
- Click a user node → set query user, run propagation to depth K, update visualization
- Move K slider → re-run propagation for same query user at new depth
- Tooltip on hover over any node: shows type, ID, degree, and (for items in the visualization) shortest path length from query user
- Tooltip on hover over any edge: shows interaction weight (1/√d_u * 1/√d_i)
- "Reset" button: clears query user selection

**Color scheme:** dark background (#1a1a2e). User nodes: #4fc3f7 (blue). Item nodes: #ffb74d (orange). Query user: #ef5350 (red). 1-hop items: #ff8a65 (bright orange). 2-hop users: #64b5f6 (bright blue). 3-hop recommendations: #ffd54f (gold). Faded opacity: 0.2.

**Educational note below canvas:** "The gold items are LightGCN's K=2 recommendation candidates for the red user — items liked by users who share tastes with you, even though you've never interacted with them directly."
</details>
