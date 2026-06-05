<!--
Original MicroSim specification for ch17-table-to-graph.
Excluded via exclude_docs (**/sim-spec.md).
-->

# Spec: Relational Schema to Heterogeneous Graph

- sim-id: ch17-table-to-graph
- chapter: 17-relational-deep-learning
- bloom: Understand
- library: p5.js

## Original specification

<details markdown="1">
<summary>Interactive visualization of relational-to-graph construction for a three-table e-commerce database</summary>
Type: MicroSim
**sim-id:** ch17-table-to-graph<br/>
**Library:** p5.js<br/>
**Status:** Specified

**Learning objective:** Understand — students observe how foreign key relationships in a three-table relational database (Customers, Orders, Products) map to a heterogeneous graph with typed nodes and typed edges.

**Canvas:** 900×520px, two-panel layout.

**Left panel (380px): Tabular view**
- Three scrollable HTML-style table widgets rendered with p5.js: Customers (4 rows shown), Orders (6 rows shown), Products (3 rows shown)
- Each table has a distinct header color: Customers (blue), Orders (green), Products (orange)
- FK columns highlighted in yellow (Orders.customer\_id, OrderItems.order\_id, OrderItems.product\_id)
- On hover over an FK value (e.g., customer\_id = 3 in the Orders table), the referenced row in the Customers table also highlights
- A "selected row" state: click any row to make it the focus row (highlighted with a border)

**Right panel (500px): Graph view**
- Heterogeneous graph with the same three node types (blue circles for customers, green squares for orders, orange diamonds for products)
- 8 customer nodes, 12 order nodes, 5 product nodes, positioned using a force-directed layout with type-grouped initialization
- Edges drawn as thin lines, colored by meta-relation type (blue for places→, green for for→, orange for includes→)
- When a table row is selected in the left panel, the corresponding graph node highlights (red border + bold), and all its 1-hop and 2-hop neighbors are highlighted at decreasing opacity (1-hop bright, 2-hop dim)
- When hovering over a graph node, a tooltip shows: node type, row ID, and feature values (from the table)

**Interaction:**
- Click any row in the left panel → highlights the corresponding node in the right panel and its neighborhood
- Click any node in the right panel → highlights the corresponding table row in the left panel and the node's graph neighborhood
- Toggle "Show reverse edges" checkbox: toggles display of reverse FK edges (customer→has\_order→order etc.)
- Toggle "Show meta-relation labels" checkbox: labels each edge with its meta-relation type

**Animation (optional):** when a node is first selected, a brief ripple animation shows messages propagating outward to 1-hop, then 2-hop neighbors, illustrating the GNN propagation concept.

**Color scheme:** dark background (#1a1a2e). Customer nodes: #4fc3f7. Order nodes: #81c784. Product nodes: #ffb74d. Selected node: #ef5350. FK values in table: #ffd54f. Highlighted table rows: match node color at 30% opacity.

**Caption below canvas:** "Every foreign key in a relational database becomes an edge in a graph. Click a row in any table to see which graph node it becomes and how information propagates through its neighborhood."
</details>
