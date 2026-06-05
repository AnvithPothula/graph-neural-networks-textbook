<!--
Original MicroSim specification for ch14-kg-embedding-space.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Cross-KG Structure Transfer

- sim-id: ch14-kg-embedding-space
- chapter: 14-kg-foundation-models
- bloom: Analyze
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive 2D embedding space showing how ULTRA represents structural patterns across two different KGs</summary>
Type: MicroSim
**sim-id:** ch14-kg-embedding-space<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Apply and Analyze — students observe how structurally equivalent entities in two different KGs (drawn side-by-side) obtain similar embedding positions despite having completely different entity IDs, illustrating that ULTRA represents structural role rather than identity. Bloom's: Applying (recognizing structural equivalence), Analyzing (comparing embedding clusters across KGs).

**Canvas:** 900×500px total, split into two 420×440 panels (left: training KG, right: test KG) with a 60px divider. Responsive to window resize.

**Left panel — training KG:**
- Display 12 nodes arranged in a mixed topology: 4 high-degree hub nodes (large circles, degree ≥ 4), 4 mid-degree nodes (medium circles, degree 2–3), 4 leaf nodes (small circles, degree 1).
- Edge colors encode relation types: 3 types, drawn as colored lines (teal, orange, purple).
- Node fill color: a 2D embedding computed by a fixed random projection of the structural fingerprint (node degree + relation-type distribution). This projection is identical for both panels.
- On hover, a tooltip shows the node's structural role ("Hub: 5 outgoing edges, types [0,2]" etc.) and 2D embedding coordinates.

**Right panel — test KG:**
- Display 12 entirely different entities (different IDs, completely new KG), but arranged in a structurally identical topology (same adjacency pattern, same relation types).
- Use the same structural fingerprint → 2D projection. Hub nodes in the test KG should cluster near hub nodes in the training KG panel (shown via a faint connecting line when "Show Transfer" is toggled on).
- "Show Transfer" toggle button (below center divider): when ON, draws faint arcs from each test node to its nearest structural neighbor in the training panel, with line opacity proportional to structural similarity score.

**Controls:**
- "Shuffle Entity IDs" button: re-labels all entity integers randomly. The node positions and colors do NOT change (demonstrating that the representation is ID-invariant).
- "Show Transfer" toggle: reveals cross-KG structural similarity arcs.
- "Perturb Structure" button: randomly rewires 2 edges in the test KG, showing that the embedding of affected nodes drifts while structurally stable nodes remain stationary.
- Relation type selector (3 checkboxes): toggle which relation types are shown as edges.

**Behavior:**
- Initial state: both KGs visible, no transfer arcs, entity IDs shown as small integers on nodes.
- After "Shuffle Entity IDs": IDs change, embeddings stay the same — reinforcing entity equivariance.
- After "Show Transfer" toggle: arcs appear between structurally similar nodes across the two panels. The user should observe that hub↔hub, leaf↔leaf connections dominate.
- After "Perturb Structure": affected node updates its color in real time (interpolated over 0.5 seconds), while unaffected nodes stay fixed.

**Color scheme:** dark background (#1a1a2e), teal edges for relation type 0 (#4ecdc4), orange for type 1 (#ff6b6b), purple for type 2 (#c77dff), node fill from viridis scale based on 2D embedding x-coordinate.

Implementation: p5.js with hand-coded force-directed layout (fixed positions stored as arrays, not dynamic). Structural fingerprint computation done once on canvas setup and stored. Cross-KG arcs drawn with bezier curves between the two panels.
</details>
