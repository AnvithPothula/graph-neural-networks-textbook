<!--
Original MicroSim specification for ch25-tool-use-graph.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Agent Tool-Use Graph — Interactive Planner

- sim-id: ch25-tool-use-graph
- chapter: 25-agents-and-graphs
- bloom: Understand
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive tool-use graph showing dependency traversal and parallel execution</summary>
Type: MicroSim
**sim-id:** ch25-tool-use-graph<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Students observe how a tool-use graph structures an agent's execution plan and how dependency edges determine which tools can run in parallel versus sequentially (Bloom's: Understanding — explaining how graph structure enables agent planning).

**Canvas:** 900 × 500 px, responsive to window resize. Background: #0d1117 (dark mode).

**Layout:** Full canvas force-directed graph with 8 tool nodes. Nodes are large circles (radius 40px) with icons (text labels inside) and category-colored borders:
- Web/search tools: blue
- Code execution tools: orange
- Data/DB tools: green
- Output/write tools: purple
- Analysis tools: teal

Edges are directed arrows. Dependency edges (solid, white) point from prerequisite to dependent. Optional parallel-safe annotations appear as dashed green arcs between independent sibling nodes.

**Preloaded scenario:** "Research report agent" with 8 tools:
1. web-search (blue) → 2. retrieve-papers (blue)
2. retrieve-papers → 4. summarize (teal)
4. summarize → 7. write-report (purple)
3. python-exec (orange) → 7. write-report
5. sql-query (green) → 6. data-analysis (teal) → 7. write-report
7. write-report → 8. email-send (purple)

**Execution animation (triggered by "Run Plan" button):**
- Step 1: nodes with in-degree 0 (web-search, python-exec, sql-query) glow gold simultaneously — label "Can run in parallel"
- Step 2: retrieve-papers unlocks and glows (web-search done)
- Step 3: summarize unlocks, data-analysis unlocks (both predecessors done)
- Step 4: write-report unlocks (summarize, python-exec, data-analysis done)
- Step 5: email-send unlocks
- Speed: 1.5 seconds per step, with completed nodes turning green

**Controls:**
- "Run Plan" button: starts the step-by-step execution animation
- "Reset" button: returns all nodes to idle state
- "Add Tool" button: opens a modal to add a new tool node and specify which existing tools it depends on
- "Remove Edge" button: click two nodes to remove the dependency between them; re-run to see the updated plan
- Scenario dropdown: "Research agent | Code debugging agent | Data pipeline agent"

**Interactions:**
- Clicking a node shows a tooltip: tool name, description, expected inputs/outputs, current status (idle/running/done)
- Clicking an edge shows: which tool produces the input for which tool
- Hovering a "parallel group" of nodes (those with in-degree 0 at a given step) highlights them with a shared glow

**Responsiveness:** graph scales to fill canvas; nodes reposition via force layout on resize
</details>
