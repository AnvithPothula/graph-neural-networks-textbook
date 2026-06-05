<!--
Original MicroSim specification for ch22-traffic-temporal.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Traffic Forecasting Architecture — MicroSim

- sim-id: ch22-traffic-temporal
- chapter: 22-temporal-graphs
- bloom: Understand
- library: p5.js

## Original specification

<details markdown="1">
<summary>Temporal Graph — Traffic Sensor Network Visualization</summary>
Type: microsim
**sim-id:** ch22-traffic-temporal
**Library:** p5.js
**Status:** Specified

Interactive p5.js MicroSim showing a simplified road sensor network evolving over time. Canvas: 780×460px, responsive to window resize events.

**Layout:**
- Left panel (380×400px): road network graph with 15 sensor nodes arranged in a loose grid mimicking a small city road network. Edges represent road connections. Node color encodes current traffic speed (green = fast / blue = normal / yellow = slow / red = congested).
- Right panel (380×400px): time-series chart for the selected node, showing speed (y-axis, 0–70 mph) vs. time (x-axis, last 60 timesteps = 5 hours). A vertical "current time" line sweeps left to right.

**Controls (below panels):**
- "Play / Pause" button: animates the time scrubber; each frame advances by one 5-minute timestep
- Speed slider (×1 to ×10): controls animation speed
- "Select Node" — clicking any node in the left panel selects it and shows its time series in the right panel

**Behavior:**
- At each timestep, node colors update based on pre-generated synthetic speed data that includes: a morning rush (timesteps 30–50, speed drops), a midday lull (timesteps 60–80, normal), an evening rush (timesteps 90–110, drops again)
- When a node is selected, the right panel shows its complete time series with the current timestep highlighted
- Hovering any node shows a tooltip: "Node N | Current speed: X mph | Trend: ↑/↓/→"
- Clicking any edge shows a tooltip: "Road segment: N1 ↔ N2 | Avg speed correlation: 0.73"

**Learning objective (Understanding — Bloom's Taxonomy):** Students observe how spatial dependencies (neighboring sensors show correlated speed drops) and temporal dependencies (rush hours follow predictable patterns) motivate the joint spatio-temporal modeling approach of STGCN and DCRNN.

Implementation: p5.js with synthetic sinusoidal speed data plus Gaussian noise. Node positions hardcoded as a 3×5 grid with some random offsets. Responsive via `windowResized()` callback.
</details>
