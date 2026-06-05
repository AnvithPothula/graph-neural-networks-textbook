<!--
Original MicroSim specification for ch09-wl-refinement (WL Refinement MicroSim).
Excluded from the built site via exclude_docs (**/sim-spec.md).
-->

# Spec: WL Refinement MicroSim

- sim-id: ch09-wl-refinement
- chapter: 09-gnn-theory
- bloom: Apply
- library: p5.js

## Original specification block

<details markdown="1">
<summary>ch09-wl-refinement — Interactive WL Color Refinement Visualizer</summary>

**sim-id:** ch09-wl-refinement  
**Library:** p5.js  
**Status:** scaffold

**Purpose:** Demonstrate 1-WL color refinement on two non-isomorphic graphs that WL *can* distinguish and one pair that it *cannot*, making the algorithm's power and limitations tangible.

**Layout:**

- **Top row**: Graph A (left) and Graph B (right), displayed side by side with nodes colored by their current WL color. Each node is a circle; color encodes the WL label (distinct colors = distinct labels, using a vivid palette).
- **Bottom panel**: a "Color histogram" for each graph showing the count of each distinct color. If the histograms differ, label: "DISTINGUISHABLE ✓". If identical, label: "INDISTINGUISHABLE — same color histogram".

**Controls:**

- **Step button**: advance one WL iteration; all nodes update their colors simultaneously, with a brief animation showing the hash operation as a glowing "merge" effect on each node.
- **Reset button**: returns to iteration 0 (all same color).
- **Graph selector dropdown**: choose from three preset graph pairs:
  - Pair 1: two small non-isomorphic graphs (WL distinguishes at step 1)
  - Pair 2: two 3-regular graphs on 6 nodes (K₃,₃ vs. prism; WL cannot distinguish at any step)
  - Pair 3: two trees that WL cannot distinguish (same degree sequence, different topology)
- **"Run to convergence" button**: automatically step until no colors change.

**Interaction:**

- Hover over any node → tooltip shows "Iteration k: color = [hash value]" and the multiset of neighbor colors used in the hash.
- Click any node → highlight all nodes with the same color in both graphs.

</details>
