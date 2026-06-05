<!--
Original MicroSim specification for ch05-sir-epidemic-network (SIR Epidemic Dynamics on Network Structures).
Excluded from the built site via exclude_docs (**/sim-spec.md).
Kept so the spec survives after the <details> block is removed from the chapter.
-->

# Spec: SIR Epidemic Dynamics on Network Structures

- sim-id: ch05-sir-epidemic-network
- chapter: 05-label-propagation
- bloom: Analyze
- library: p5.js

## Original specification block

<details markdown="1">
<summary>Interactive SIR simulation comparing homogeneous-mixing ODE vs. network-structured spread</summary>

**sim-id:** ch05-sir-epidemic-network<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Analyzing (Bloom's Level 4) — Students compare SIR dynamics on a well-mixed population (classical ODE) versus an explicit network graph, identifying how network heterogeneity changes epidemic behavior.

**Canvas:** 800×560px, split into two panels:
- **Left panel (400×560):** Network simulation — 60-node Karate Club graph (or small ER random graph with 60 nodes, ⟨k⟩=4), node size proportional to degree, color = SIR state (blue=S, red=I, green=R)
- **Right panel (400×560):** Time series plot — S(t), I(t), R(t) curves for both the ODE model (dashed lines) and the network simulation (solid lines) on the same axes

**Controls (below canvas):**
- **β slider** (0.01 – 0.5, step 0.01, default 0.1): transmission rate
- **γ slider** (0.01 – 0.3, step 0.01, default 0.05): recovery rate
- **R₀ display** (computed automatically as β/γ): colored red if >1, green if <1
- **Network selector** dropdown: "Karate Club (heterogeneous)", "Erdős-Rényi (homogeneous)", "Star (extreme hub)"
- **Seed** button: randomly infect 2 nodes to begin the simulation
- **Reset** button: return all nodes to Susceptible

**Behavior:**
- Continuous-time simulation: at each frame (30fps), for each I-node, independently activate each S-neighbor with probability \( 1 - e^{-\beta \Delta t} \) and recover with probability \( 1 - e^{-\gamma \Delta t} \) (discrete-time Gillespie approximation with Δt = 0.1)
- ODE curves use Euler integration with Δt = 0.01, initialized to S(0) = N-2, I(0) = 2, R(0) = 0
- Right panel axes: x = time (0 to 200), y = fraction of population (0 to 1)
- Show vertical dashed line at current time step connecting left and right panels

**Hover interaction:** hovering a node in the left panel shows tooltip: "Node [id] | State: [S/I/R] | Degree: [k] | Time infected: [T or —]"

**Implementation notes:** hardcode both the Karate Club edge list (78 edges) and a pre-generated ER graph. Star graph: 1 hub connected to 59 leaves. All three networks have identical mean degree ≈ 4 for fair comparison.

</details>
