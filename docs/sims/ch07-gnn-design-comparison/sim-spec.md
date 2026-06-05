<!--
Original MicroSim specification for ch07-gnn-design-comparison (GNN Design Space Interactive Comparison).
Excluded from the built site via exclude_docs (**/sim-spec.md).
-->

# Spec: GNN Design Space Interactive Comparison

- sim-id: ch07-gnn-design-comparison
- chapter: 07-gnn-design-space
- bloom: Create
- library: Chart.js

## Original specification block

<details markdown="1">
<summary>Interactive benchmark chart comparing GCN, GraphSAGE, and GAT across design axes</summary>

**sim-id:** ch07-gnn-design-comparison<br/>
**Library:** Chart.js<br/>
**Status:** Specified

**Learning objective:** Evaluating (Bloom's Level 5) — Students compare architecture choices across multiple performance dimensions, developing the judgment to select appropriate GNN designs for given task constraints.

**Canvas:** 720×480px, responsive to window resize.

**Chart type:** Grouped bar chart with 4 bar groups (one per dataset: Cora, CiteSeer, ogbn-arxiv, QM9-molecule) and 3 bars per group (one per model: GCN, GraphSAGE, GAT).

**Bar data (test accuracy or MAE):**
- Cora: GCN 81.5%, GraphSAGE 82.0%, GAT 83.0%
- CiteSeer: GCN 70.3%, GraphSAGE 71.3%, GAT 72.5%
- ogbn-arxiv: GCN 71.7%, GraphSAGE 71.5%, GAT 73.9%
- QM9 (MAE, lower is better, shown inverted as 1 − normalized_MAE): GCN 0.65, GraphSAGE 0.66, GAT 0.70

**Bar colors:** GCN = #6366f1 (indigo), GraphSAGE = #10b981 (green), GAT = #f59e0b (amber)

**Axes:** Y-axis labeled "Test Accuracy (%) or Performance Score"; X-axis labeled "Dataset"; gridlines at 10% intervals.

**Controls (below chart):**
- **Metric selector** toggle: "Accuracy" | "Relative Improvement over GCN" — switching shows delta bars
- **Architecture filter** checkboxes: GCN, GraphSAGE, GAT — toggling hides/shows bars
- **Design axis overlay** dropdown: "None" | "Skip connections effect" | "Aggregation effect" | "Depth effect" — selecting an overlay adds a line series showing the effect of that design change on each dataset

**Hover on any bar:** tooltip showing "Model: [name] | Dataset: [name] | Score: [value] | Parameters: [count]"

**Implementation notes:** Use Chart.js with the grouped bar chart type. Hardcode the data arrays in the JavaScript. For the overlay line series, hardcode the delta values from the You et al. (2020) design space study.

</details>
