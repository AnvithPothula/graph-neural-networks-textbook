<!--
Original MicroSim specification for ch05-label-propagation-stepper (Label Propagation Step-by-Step Simulator).
Excluded from the built site via exclude_docs (**/sim-spec.md).
Kept so the spec survives after the <details> block is removed from the chapter.
-->

# Spec: Label Propagation Step-by-Step Simulator

- sim-id: ch05-label-propagation-stepper
- chapter: 05-label-propagation
- bloom: Analyze
- library: p5.js

## Original specification block

<details markdown="1">
<summary>Interactive label propagation visualization on the Karate Club graph</summary>

**sim-id:** ch05-label-propagation-stepper<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Understanding (Bloom's Level 2) — Students observe how label information flows outward from seed nodes across a graph, developing intuition for the harmonic averaging property and convergence behavior of label propagation.

**Canvas:** 750×520px, responsive to window resize. Force-directed layout (D3-force or equivalent physics) for the 34-node Karate Club graph; node positions stable after initialization.

**Initial state:**
- All nodes rendered as gray circles (radius 12px), except:
  - Node 0: blue (class 0, Mr. Hi faction)
  - Node 33: red (class 1, Officer faction)
- Edges rendered as thin gray lines
- Iteration counter at top: "Iteration: 0 | Status: Initialized"
- Bottom-right corner: accuracy readout "Accuracy: — (step to begin)"

**Controls (below canvas, horizontal row):**
- **Step** button: advance one iteration of label propagation (update all unlabeled nodes)
- **Auto** button: run iterations automatically at 0.6s intervals until convergence (toggle — clicking again pauses)
- **Reset** button: return to initial state (only nodes 0 and 33 labeled)
- **α slider** (0.1 – 0.9, default 0.8): switches between label propagation (α ≈ 0) and label spreading (α ≈ 0.9). Show label "Graph influence α"

**Node color encoding:**
- Color is a continuous blend: pure blue (#3b82f6) for \( f_v = 0 \), pure red (#ef4444) for \( f_v = 1 \), interpolated as `lerpColor(blue, red, f_v)` for intermediate values
- Border ring: dashed outline for seed/labeled nodes; solid outline for propagated nodes; thin outline for still-uncertain nodes (|f_v - 0.5| < 0.05)
- On hover: tooltip showing node ID, current \( f_v \) value (2 decimal places), and degree

**Behavior per step:**
- Synchronous update: compute all new \( f_v^{(t+1)} \) values before applying (not in-place)
- Labeled nodes (0 and 33) hold fixed regardless of α
- Convergence: declare converged when \( \max_v |f_v^{(t+1)} - f_v^{(t)}| < 0.001 \)
- On convergence: display "Converged at iteration K | Accuracy: X%" (compare to ground-truth factions)

**Accuracy computation:** use known faction labels for all 34 nodes; threshold \( f_v \geq 0.5 \) for class 1. Update accuracy readout after every step.

**Implementation notes:** embed the Karate Club graph edge list directly in the JavaScript as a hardcoded array of 78 edge pairs. Derive node positions from a fixed seed force simulation run for 300 ticks at load time so positions are deterministic. Ensure the sim remains readable at 480px wide (mobile).

</details>
