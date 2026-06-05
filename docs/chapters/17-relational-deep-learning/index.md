---
title: "Chapter 17: Relational Deep Learning"
description: "Introduces Relational Deep Learning as a unified framework for applying GNNs directly to relational databases, with RelBench as the standard benchmark and RelGNN as the reference architecture."
generated_by: claude skill chapter-content-generator
date: 2026-05-29 22:49:35
version: 0.08
---

# Chapter 17: Relational Deep Learning

**Part 4: Graphs in the Wild**

## Summary

Introduces Relational Deep Learning as a unified framework for learning on relational databases represented as heterogeneous graphs, with RelBench as the standard benchmark.

## Concepts Covered

This chapter covers the following 5 concepts from the learning graph:

1. Relational Database as Graph
2. Relational Deep Learning
3. RelBench
4. RelGNN
5. RelBench (Benchmark)

## Prerequisites

This chapter builds on:

- [Chapter 6: GNN Foundations: Message Passing and GCN](../06-gnn-foundations/index.md)
- [Chapter 7: GNN Design Space: GraphSAGE and GAT](../07-gnn-design-space/index.md)
- [Chapter 15: Heterogeneous Graphs](../15-heterogeneous-graphs/index.md)
- [Chapter 16: GNNs for Recommender Systems](../16-recommender-systems/index.md)

---

!!! mascot-welcome "Every Database Is Already a Graph"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving welcome">
    The most abundant source of structured data in the world is not the benchmark datasets that populate machine learning papers — it is the relational databases that underpin every enterprise application, e-commerce platform, healthcare system, and financial institution on the planet. These databases contain trillions of rows organized into tables linked by foreign key relationships, and they encode extraordinarily rich information about entities and their interactions. The frustrating irony is that most machine learning pipelines discard this structure: they flatten the tables into a single wide feature matrix through manual SQL joins and aggregations, losing the higher-order relational patterns in the process. Relational Deep Learning (RDL) is a framework that refuses this flattening by observing a fundamental fact: a relational database with foreign keys is already a heterogeneous graph, and the entire arsenal of GNN methods developed in the preceding chapters applies directly — without a single additional architectural idea — once the database is correctly converted to graph form.

---

## 17.1 Motivating Example: Customer Churn Prediction in an E-Commerce Database

Consider a moderately sized e-commerce company running its business on a standard relational database with five tables:

| Table | Description | Columns |
|---|---|---|
| **Customers** | One row per registered user | customer_id, age, signup_date, country |
| **Orders** | One row per purchase | order_id, customer_id *(FK)*, order_date, total_amount, status |
| **OrderItems** | One row per line item in an order | item_id, order_id *(FK)*, product_id *(FK)*, quantity, unit_price |
| **Products** | One row per product | product_id, category, brand, price, rating |
| **Reviews** | One row per customer review | review_id, customer_id *(FK)*, product_id *(FK)*, rating, review_date |

The business task is **customer churn prediction**: given a cutoff date \( T \), predict for each customer whether they will place at least one order in the 30 days following \( T \), using only data observable before \( T \). This is a binary classification task on customer entities with a temporal structure — the label is a future event, the features must be strictly historical.

The traditional machine learning pipeline for this task involves an analyst writing SQL queries to compute customer-level aggregates: total number of orders in the past 90 days, average order value, number of distinct product categories purchased, average review rating given, days since last order. Each of these aggregates requires a join across tables, and the selection of which aggregates to compute requires domain knowledge. The resulting feature vector contains at most a few dozen hand-crafted signals, and the analyst must explicitly decide to discard the relational structure — the sequence of orders, the specific products reviewed, the overlap between a customer's product preferences and their purchase history.

Relational Deep Learning takes a different path: convert the entire database schema into a heterogeneous graph, assign row-level feature encodings to each node, and run a GNN whose message passing propagates information across foreign-key links automatically. The GNN discovers which higher-order patterns are predictive during training, without requiring the analyst to articulate them in SQL.

---

## 17.2 Formal Construction: From Relational Schema to Heterogeneous Graph

A relational database schema \( \mathcal{S} = (\mathcal{T}, \mathcal{F}) \) consists of a set of tables \( \mathcal{T} = \{T_1, T_2, \ldots, T_m\} \) and a set of foreign key relationships \( \mathcal{F} \). Each foreign key relationship \( f \in \mathcal{F} \) is a triple \( (T_i, c_i, T_j) \), meaning that column \( c_i \) in table \( T_i \) references the primary key of table \( T_j \). For example, in the e-commerce database, the Orders table has a foreign key Orders.customer\_id → Customers.customer\_id, meaning each order row references exactly one customer row.

The **relational-to-graph construction** maps this schema to a heterogeneous graph \( G = (\mathcal{V}, \mathcal{E}, \tau, \phi) \) as follows:

**Nodes.** For each table \( T_k \in \mathcal{T} \) and each row \( r \in T_k \), create a node \( v_r \). Assign node type \( \tau(v_r) = T_k \). The node set is:

\[
\mathcal{V} = \bigcup_{T_k \in \mathcal{T}} \{v_r : r \in T_k\}
\]

**Edges.** For each foreign key relationship \( (T_i, c_i, T_j) \in \mathcal{F} \) and each row \( r_i \in T_i \) with value \( r_i[c_i] = k \), create an edge \( (v_{r_i}, v_{r_j}) \) where \( r_j \) is the unique row in \( T_j \) with primary key value \( k \). The edge receives relation type \( \phi(e) = (T_i, \text{fk}_{c_i}, T_j) \). Additionally, the reverse edge \( (v_{r_j}, v_{r_i}) \) is added with a distinct reverse relation type — this allows information to flow in both directions across the FK relationship, so a customer node can aggregate from its orders and an order node can also aggregate from its customer.

**Node features.** Each row's non-primary-key, non-foreign-key column values become initial node features. Numeric columns are standardized. Categorical columns with low cardinality are one-hot encoded; high-cardinality categoricals (e.g., product descriptions) may be encoded with a text encoder. Timestamp columns require special treatment — the raw timestamp is not directly informative, but features derived from it (time elapsed since signup, day of week, month) are useful and are extracted before encoding.

!!! mascot-thinking "The Foreign Key Is an Edge"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinking carefully">
    The conceptual leap of Relational Deep Learning is disarmingly simple: a foreign key column, in database theory, is nothing more than a reference from one row to another — a pointer that connects two entities. In graph theory, a reference from one node to another is an edge. These are the same object described in two different vocabularies. The reason this equivalence is powerful is that decades of database normalization practice have already done the work of identifying which entities should be connected: the ER (Entity-Relationship) diagram that every database designer draws before writing the first CREATE TABLE statement is literally a graph schema, specifying node types and edge types. RDL is, in a precise sense, the reverse of the normalization process: normalization decomposes a flat feature matrix into tables to reduce redundancy and preserve referential integrity; RDL reconstructs a graph from those tables to enable structure-aware learning. The remarkable implication is that every normalized relational database, without any additional schema design, yields a well-structured heterogeneous graph that is ready for GNN processing.

For the e-commerce database, the resulting heterogeneous graph has five node types (Customers, Orders, OrderItems, Products, Reviews) and the following meta-relations (plus their reverses):

| Source table | Relation | Target table | Semantic meaning |
|---|---|---|---|
| Orders | fk\_customer | Customers | This order was placed by this customer |
| OrderItems | fk\_order | Orders | This line item belongs to this order |
| OrderItems | fk\_product | Products | This line item is for this product |
| Reviews | fk\_customer | Customers | This review was written by this customer |
| Reviews | fk\_product | Products | This review is about this product |

The customer churn prediction task is a node-level classification task on Customers nodes, with labels computed from the Orders table after cutoff time \( T \). The GNN receives the full graph (up to cutoff time) as input and produces a prediction logit for each customer node.

---

## 17.3 The Feature Engineering Baseline and Its Limits

Before examining what RDL enables, it is worth understanding precisely what traditional feature engineering captures and what it discards. A principled feature engineering approach for the customer churn task would compute, for each customer \( u \), a set of SQL-derived aggregates over a historical time window \( [T - \Delta, T] \):

\[
\mathbf{x}_u = \left[\text{num-orders},\; \text{avg-order-value},\; \text{num-categories},\; \text{avg-review-rating},\; \text{days-since-last-order},\; \ldots\right]
\]

Systems like Featuretools (automated feature engineering) can generate dozens or hundreds of such aggregates by systematically enumerating join paths in the schema — effectively implementing a fixed-depth version of the RDL construction without learning which paths are predictive. The resulting feature matrix can then be fed to a gradient-boosted tree (LightGBM, XGBoost) or a shallow MLP.

The information that this pipeline discards falls into three categories. First, **higher-order relational patterns**: features computed via a single join (e.g., "number of orders") miss patterns that require two joins (e.g., "how many distinct brands has this customer purchased across their top-reviewed product categories"). In principle, Featuretools can compute multi-hop aggregates, but the combinatorial explosion of join paths makes comprehensive coverage intractable beyond depth 2–3. Second, **structural context**: the structure of the subgraph around a customer — which other customers share their product preferences, how central they are in the co-purchase network — is not captured by per-customer aggregates. Third, **sequential and temporal ordering**: aggregates collapse ordered sequences to statistics, losing the information encoded in the sequence of events (e.g., a customer who has gradually decreased order frequency over six months versus one who has been consistently infrequent).

GNN-based RDL addresses the first two limitations directly: multi-hop message passing propagates information across the join graph to arbitrary depth (bounded by the number of layers), and the graph structure is an explicit input to the model. The third limitation — temporal ordering — requires special handling described in §17.5, and its full treatment is deferred to Chapter 22 on temporal graph networks.

---

## 17.4 Relational Deep Learning: The Unified Framework

**Relational Deep Learning** (Fey et al. 2024) is a framework that formalizes the relational-to-graph construction and defines a standardized pipeline for applying GNNs to predictive tasks on relational data. The framework has three components:

**Graph construction layer.** Given a database schema and a task specification (target table, target column, prediction time \( T \)), the construction layer automatically instantiates the heterogeneous graph \( G \) by executing the relational-to-graph mapping described in §17.2, filtering all rows to include only those observable before time \( T \), and computing initial node features from column values.

**GNN encoder layer.** A heterogeneous GNN (R-GCN, HGT, or a task-specific variant) is applied to the constructed graph, propagating information through the FK-defined edge structure. The output is a dense embedding \( \mathbf{h}_v \in \mathbb{R}^d \) for each node \( v \), encoding the node's features and its relational neighborhood.

**Prediction head.** A task-specific MLP or link predictor maps the entity embedding(s) to the prediction target: a classification logit for node-level classification, a regression output for node-level regression, or a pair score for link prediction.

The end-to-end pipeline — database → graph → GNN → prediction — is trained jointly with gradient descent, so the GNN learns to propagate exactly those relational signals that are predictive for the target task, without requiring explicit feature specification.

!!! mascot-tip "The Temporal Cutoff Is Not Optional"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage offering practical advice">
    The most consequential implementation detail in RDL is the temporal cutoff: every row included in the training graph must have a timestamp strictly less than the prediction time \( T \). This sounds obvious, but it is violated in subtle ways by naive implementations. Consider the e-commerce database: when computing node features for a customer at cutoff time \( T_\text{train} \), a developer might accidentally include the Orders table without filtering by order date, incorporating orders that occur after \( T_\text{train} \). The model trained on this data will appear to perform well (it is effectively peeking at the future), but it will fail catastrophically in production. The same leakage can occur through multi-hop paths: a review written after the cutoff but referencing a product that a customer purchased before the cutoff can expose information about the post-cutoff interaction. The RelBench benchmark makes temporal cutoff enforcement a first-class design requirement, providing a validated cutoff mechanism that filters all tables before graph construction — a design decision that the original literature identifies as responsible for a substantial fraction of the performance gap between offline evaluation and production deployment in historical enterprise ML systems.

---

## 17.5 Temporal Structure in Relational Data

Relational databases maintain timestamps for most business-critical events: order dates, review submission times, login timestamps, transaction datetimes. This temporal metadata enables a temporally-aware graph construction that is strictly more informative than a static snapshot, but it also introduces data leakage risks that require careful handling.

For a prediction task with prediction time \( T \), the **temporally-filtered graph** \( G_T \) includes only rows \( r \) satisfying \( r[\text{timestamp}] \leq T \) (for tables that have timestamps) and all rows in tables without timestamps (static reference tables like Products that do not change over time). Training, validation, and test instances correspond to different prediction times \( T_\text{train} < T_\text{val} < T_\text{test} \), and the graph used for each instance uses the corresponding filtered view of the database.

This temporal structure has an important consequence for evaluation: unlike standard node classification benchmarks where all training and test nodes coexist in the same static graph, RDL tasks use a **walk-forward evaluation** protocol. Models are trained on instances at time \( T_\text{train} \), tuned on instances at \( T_\text{val} \), and evaluated on instances at \( T_\text{test} \), where the three graphs \( G_{T_\text{train}}, G_{T_\text{val}}, G_{T_\text{test}} \) are strictly nested. This mirrors the realistic deployment scenario: a model trained on data through month 6 and evaluated on month 9 sees genuinely new entities and new relationships at test time.

Temporal attributes also enable encoding the **age** of each entity and each relationship relative to the cutoff time. An order placed 3 days before cutoff carries different predictive signal than an order placed 300 days before cutoff, even if their other attributes are identical. RelGNN (§17.7) encodes this temporal recency as part of its edge feature representation. The full treatment of models that continuously update entity representations as new events arrive — rather than using a static cutoff snapshot — is the subject of Chapter 22 (Temporal Graphs), where TGN and TGAT provide architectures specifically designed for event-based temporal graphs.

---

## 17.6 RelBench: Standardizing Relational Learning Evaluation

**RelBench** (Robinson et al. 2024) is a benchmark suite designed to standardize evaluation for deep learning on relational databases, addressing the reproducibility challenges that plagued earlier work in this space — where different papers used different database schemas, different train/val/test splits, different feature engineering baselines, and different evaluation metrics, making it effectively impossible to compare results across papers.

RelBench provides seven real-world relational datasets spanning multiple domains:

| Dataset | Domain | Tables | Rows (total) | Task type | Metric |
|---|---|---|---|---|---|
| rel-amazon | E-commerce | 5 | 12M | Entity classification | Average Precision |
| rel-f1 | Formula 1 racing | 5 | 0.5M | Entity regression | MAE |
| rel-hm | Fashion retail | 4 | 5M | Link prediction | NDCG@10 |
| rel-stack | Q&A community | 5 | 15M | Entity classification | Average Precision |
| rel-trial | Clinical trials | 4 | 3M | Entity classification | Average Precision |
| rel-avito | Classified ads | 6 | 8M | Link prediction | NDCG@10 |
| rel-event | Event platform | 4 | 2M | Entity regression | RMSE |

Each dataset ships with a standardized graph construction (verified to be leakage-free), temporal train/val/test splits aligned to realistic deployment scenarios, and a Python API (`relbench` package) that returns PyTorch Geometric `HeteroData` objects directly from a SQL database. This eliminates a major source of irreproducibility: researchers implementing their own database-to-graph conversions frequently introduce subtle schema differences that confound method comparisons.

RelBench also provides **baseline implementations** for LightGBM with automated feature engineering (Featuretools DFS), SAGE on the relational graph, HGT, and a temporal GNN baseline, enabling head-to-head comparison of the full methodology stack — not just the GNN architecture in isolation. The benchmark's design principle is that the feature engineering baseline should be as competitive as possible (using the best available AutoML tools), so that performance gaps reflect genuine architectural advantages rather than weak baselines.

---

## 17.7 RelGNN: Composite Message Passing for Relational Data

**RelGNN** (2025) is the reference GNN architecture for relational deep learning, designed to address two structural properties of relational graphs that distinguish them from the standard heterogeneous graphs of Chapter 15: **temporal edge attributes** (each FK-defined edge carries the timestamp of the relationship) and **many-to-many relationships** realized through junction tables (the OrderItems table in the e-commerce example is a junction table connecting Orders to Products with line-item-level attributes).

### 17.7.1 Architecture Overview

!!! mascot-encourage "Relational Graphs Have More Structure Than Generic Heterogeneous Graphs"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage encouraging the reader">
    RelGNN inherits everything from Chapter 15's heterogeneous GNN toolkit — type-specific projections, relation-specific weight matrices, attention over typed neighborhoods — and adds two components specific to the relational setting. The first is **temporal recency encoding**: each edge carries a time-to-cutoff feature \( \Delta t = T - t_e \) (where \( t_e \) is the edge's timestamp and \( T \) is the prediction time), which is encoded using a time2vec or sinusoidal position encoding and concatenated to the edge's feature representation. The second is **junction table handling**: many-to-many relationships realized through junction tables create three-node paths (e.g., Customer → Order → OrderItem → Product), and RelGNN's composite message passing explicitly models these three-node patterns rather than treating them as two independent two-node FK links. If the complexity of this sounds daunting, remember that you already understand R-GCN's relation-specific weight matrices and HGT's meta-relation attention — RelGNN adds temporal recency and junction path handling on top of those same primitives, not a wholly new mechanism. Take it one component at a time.

The RelGNN message-passing update for a target node \( v \) of type \( \tau(v) \) at layer \( l \) is:

\[
\mathbf{h}_v^{(l+1)} = \sigma\!\left(\mathbf{W}_{\tau(v)}^{(l)} \mathbf{h}_v^{(l)} + \sum_{r \in \mathcal{R}} \text{AGG}_r^{(l)}\!\left(\left\{\mathbf{m}_{uv}^{r,(l)} : u \in \mathcal{N}_r(v)\right\}\right)\right)
\]

where \( \mathcal{R} \) is the set of meta-relations and the **composite message** from source node \( u \) to target node \( v \) via relation \( r \) is:

\[
\mathbf{m}_{uv}^{r,(l)} = \mathbf{W}_r^{(l)} \mathbf{h}_u^{(l)} + \mathbf{W}_r^{\text{time},(l)}\, \psi(\Delta t_{uv})
\]

where:

- \( \mathbf{W}_r^{(l)} \in \mathbb{R}^{d \times d} \) is the relation-specific weight matrix, identical to R-GCN's formulation
- \( \Delta t_{uv} = T - t_e \) is the time elapsed between edge timestamp \( t_e \) and the prediction cutoff \( T \)
- \( \psi(\Delta t) \in \mathbb{R}^{d_t} \) is a sinusoidal time encoding: \( [\sin(f_1 \Delta t), \cos(f_1 \Delta t), \ldots, \sin(f_{d_t/2} \Delta t), \cos(f_{d_t/2} \Delta t)] \) with learned frequency parameters \( \{f_k\} \)
- \( \mathbf{W}_r^{\text{time},(l)} \in \mathbb{R}^{d \times d_t} \) maps the time encoding to the embedding dimension

The aggregation function \( \text{AGG}_r^{(l)} \) is typically mean pooling or attention pooling over all messages of relation type \( r \). Basis decomposition (§15.5) can be applied to \( \mathbf{W}_r^{(l)} \) when the number of meta-relations is large.

### 17.7.2 Junction Table Handling

A **junction table** (also called an association table or linking table) represents a many-to-many relationship by introducing an intermediate table whose rows each reference two other tables via foreign keys. In the e-commerce schema, OrderItems is a junction table: each OrderItems row references one Orders row and one Products row, and a given order can contain many products while a given product can appear in many orders.

Naive graph construction creates a path Customer → Order → OrderItem → Product — a three-hop path before any information from a product reaches a customer node. RelGNN introduces **path-compressed edges** for junction table patterns: for a three-table pattern \( A \xrightarrow{\text{fk}_1} J \xrightarrow{\text{fk}_2} B \) where \( J \) is a junction table, RelGNN adds a direct \( (A, B) \) edge with features aggregated from the intermediate junction node, effectively collapsing the two-hop path into a single typed edge. This halves the effective depth required for products to influence customer representations and reduces the number of GNN layers needed for the same receptive field.

### 17.7.3 Relation to Chapter 15 Architectures

RelGNN can be understood as R-GCN extended with two orthogonal additions. If the temporal recency encoding is removed (setting all \( \Delta t = 0 \)) and junction table compression is disabled, RelGNN's update equation reduces exactly to R-GCN with per-relation weight matrices. The temporal component is a form of edge feature incorporation — treating the time delta as an edge attribute that modulates the message from source to target. The junction table compression is a preprocessing step, not an architectural change to the GNN itself. This decomposition is useful for debugging: when building an RDL pipeline, it is advisable to first run standard R-GCN (Chapter 15) to establish a baseline, then add temporal encoding, then add junction compression, verifying at each step that performance improves.

---

## 17.8 Complete Code: Relational Database to Graph and Prediction

The following code demonstrates the complete RDL pipeline: constructing a toy three-table relational database (Customers, Orders, Products), converting it to a PyTorch Geometric `HeteroData` object, applying a heterogeneous GNN for customer classification, and evaluating on a held-out validation set. The implementation uses standard PyG components (HeteroConv, SAGEConv) rather than a proprietary RelGNN implementation, since the latter requires the `relbench` package — the goal here is to demonstrate the graph construction step, which is the conceptually critical piece.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

# ── 1. Construct toy relational database ──────────────────────────────────────

def make_relational_db(
    num_customers: int = 300,
    num_products:  int = 50,
    num_orders:    int = 800,
    num_items_per_order: int = 3,   # average line items per order
    seed: int = 42,
) -> dict:
    """
    Creates a toy e-commerce relational database as Python dicts.

    Tables:
      customers  : customer_id, age_norm, country_emb (2d)
      products   : product_id, price_norm, category_onehot (5d)
      orders     : order_id, customer_id (FK), days_to_cutoff
      order_items: item_id, order_id (FK), product_id (FK), quantity

    Labels: binary churn label per customer (simulated).
    """
    rng = np.random.default_rng(seed)

    # Customers table
    customer_age   = rng.uniform(0, 1, num_customers).astype(np.float32)
    customer_country = rng.integers(0, 3, num_customers)  # 3 countries

    # Products table
    product_price  = rng.uniform(0, 1, num_products).astype(np.float32)
    product_cat    = rng.integers(0, 5, num_products)   # 5 categories

    # Orders table: each order references one customer
    order_customer = rng.integers(0, num_customers, num_orders)
    order_days     = rng.uniform(0, 365, num_orders).astype(np.float32)  # days before cutoff

    # OrderItems: each item references one order and one product
    total_items = num_orders * num_items_per_order
    item_order   = np.repeat(np.arange(num_orders), num_items_per_order)
    item_product = rng.integers(0, num_products, total_items)
    item_qty     = rng.integers(1, 5, total_items).astype(np.float32)

    # Churn labels: customers with few recent orders → label=1 (churned)
    orders_per_customer = np.bincount(order_customer, minlength=num_customers)
    churn = (orders_per_customer < np.percentile(orders_per_customer, 40)).astype(np.int64)

    return {
        'customer_age':     customer_age,
        'customer_country': customer_country,
        'product_price':    product_price,
        'product_cat':      product_cat,
        'order_customer':   order_customer,
        'order_days':       order_days,
        'item_order':       item_order,
        'item_product':     item_product,
        'item_qty':         item_qty,
        'churn':            churn,
        'num_customers':    num_customers,
        'num_products':     num_products,
        'num_orders':       num_orders,
        'total_items':      total_items,
    }


# ── 2. Convert relational DB to HeteroData ────────────────────────────────────

def db_to_hetero(db: dict) -> HeteroData:
    """
    Implements the relational-to-graph construction:
      - One node type per table (customer, product, order, order_item)
      - One edge type per foreign key (plus reverses)
      - Column values → initial node features
    """
    data = HeteroData()

    # ── Node features (column values → tensors) ──────────────────────────────

    # Customer nodes: [age, country_onehot(3)]
    country_oh = F.one_hot(
        torch.tensor(db['customer_country'], dtype=torch.long), num_classes=3
    ).float()
    data['customer'].x = torch.cat([
        torch.tensor(db['customer_age']).unsqueeze(1),
        country_oh,
    ], dim=1)   # (num_customers, 4)

    # Product nodes: [price, category_onehot(5)]
    cat_oh = F.one_hot(
        torch.tensor(db['product_cat'], dtype=torch.long), num_classes=5
    ).float()
    data['product'].x = torch.cat([
        torch.tensor(db['product_price']).unsqueeze(1),
        cat_oh,
    ], dim=1)   # (num_products, 6)

    # Order nodes: [days_to_cutoff]  — the temporal recency feature
    data['order'].x = torch.tensor(db['order_days']).unsqueeze(1)  # (num_orders, 1)

    # OrderItem nodes: [quantity]
    data['order_item'].x = torch.tensor(db['item_qty']).unsqueeze(1)  # (total_items, 1)

    # ── Labels ───────────────────────────────────────────────────────────────
    data['customer'].y = torch.tensor(db['churn'], dtype=torch.long)
    nc = db['num_customers']
    perm = torch.randperm(nc)
    train_mask = torch.zeros(nc, dtype=torch.bool)
    val_mask   = torch.zeros(nc, dtype=torch.bool)
    train_mask[perm[:int(0.7*nc)]] = True
    val_mask[perm[int(0.7*nc):int(0.85*nc)]] = True
    data['customer'].train_mask = train_mask
    data['customer'].val_mask   = val_mask

    # ── Edges: FK relationships (both directions) ─────────────────────────────
    # orders.customer_id → customers.customer_id
    oc_src = torch.tensor(db['order_customer'], dtype=torch.long)
    oc_dst = torch.arange(len(db['order_customer']), dtype=torch.long)
    # Edge: order → customer (FK direction)
    data['order', 'placed_by', 'customer'].edge_index = torch.stack([
        torch.arange(db['num_orders'], dtype=torch.long),
        oc_src,
    ])
    # Reverse: customer → order
    data['customer', 'has_order', 'order'].edge_index = torch.stack([
        oc_src,
        torch.arange(db['num_orders'], dtype=torch.long),
    ])

    # order_items.order_id → orders.order_id
    io_src = torch.tensor(db['item_order'], dtype=torch.long)
    data['order_item', 'in_order', 'order'].edge_index = torch.stack([
        torch.arange(db['total_items'], dtype=torch.long),
        io_src,
    ])
    data['order', 'has_item', 'order_item'].edge_index = torch.stack([
        io_src,
        torch.arange(db['total_items'], dtype=torch.long),
    ])

    # order_items.product_id → products.product_id
    ip_src = torch.tensor(db['item_product'], dtype=torch.long)
    data['order_item', 'references', 'product'].edge_index = torch.stack([
        torch.arange(db['total_items'], dtype=torch.long),
        ip_src,
    ])
    data['product', 'referenced_by', 'order_item'].edge_index = torch.stack([
        ip_src,
        torch.arange(db['total_items'], dtype=torch.long),
    ])

    return data


# ── 3. GNN model (R-GCN style with HeteroConv) ───────────────────────────────

class RelationalGNN(nn.Module):
    """
    Two-layer heterogeneous GNN applied to the relational graph.
    Uses SAGEConv per meta-relation (equivalent to R-GCN with mean aggregation).
    Prediction head: binary classification on customer nodes.

    Parameters
    ----------
    metadata   : (node_types, edge_types) from HeteroData.metadata()
    hidden_dim : embedding dimension
    num_classes: output classes (2 for binary churn)
    """
    def __init__(self, metadata, hidden_dim=64, num_classes=2):
        super().__init__()
        node_types, edge_types = metadata

        # Type-specific input projections (§15.6): map raw features to hidden_dim
        # Linear(-1, hidden_dim) uses lazy initialization for different input dims
        self.proj = nn.ModuleDict({
            ntype: Linear(-1, hidden_dim) for ntype in node_types
        })

        # Layer 1: one SAGEConv per meta-relation
        self.conv1 = HeteroConv(
            {etype: SAGEConv((-1, -1), hidden_dim) for etype in edge_types},
            aggr='mean'
        )
        # Layer 2
        self.conv2 = HeteroConv(
            {etype: SAGEConv((-1, -1), hidden_dim) for etype in edge_types},
            aggr='mean'
        )
        # Binary classification head on customer embeddings
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x_dict, edge_index_dict):
        # Step 1: type-specific projection to shared embedding space
        h = {ntype: F.relu(self.proj[ntype](x)) for ntype, x in x_dict.items()}

        # Step 2: two rounds of heterogeneous message passing
        h = self.conv1(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)

        # Step 3: classify customer nodes
        return self.head(h['customer'])   # (num_customers, num_classes)


# ── 4. Training loop ──────────────────────────────────────────────────────────

def train_relational_gnn(epochs: int = 60, hidden_dim: int = 64, seed: int = 42):
    """
    End-to-end RDL pipeline:
      1. Construct synthetic relational database
      2. Convert to HeteroData
      3. Train RelationalGNN for customer churn prediction
      4. Report validation accuracy
    """
    torch.manual_seed(seed)
    db   = make_relational_db(seed=seed)
    data = db_to_hetero(db)

    print("Relational graph summary:")
    print(f"  Node types: {data.node_types}")
    print(f"  Edge types: {[str(e) for e in data.edge_types]}")
    for ntype in data.node_types:
        print(f"  {ntype}: {data[ntype].num_nodes} nodes, "
              f"feature dim = {data[ntype].x.shape[1]}")

    model = RelationalGNN(
        metadata=data.metadata(),
        hidden_dim=hidden_dim,
        num_classes=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)  # (num_customers, 2)
        mask = data['customer'].train_mask
        loss = F.cross_entropy(out[mask], data['customer'].y[mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x_dict, data.edge_index_dict)
            val_mask = data['customer'].val_mask
            pred = out[val_mask].argmax(dim=-1)
            val_acc = (pred == data['customer'].y[val_mask]).float().mean()
            print(f"Epoch {epoch:3d} | Loss {loss:.4f} | Val Acc {val_acc:.4f}")

    return model, data

if __name__ == '__main__':
    model, data = train_relational_gnn(epochs=60)
```

Three design decisions in this code deserve commentary. First, the `db_to_hetero` function adds edges in **both directions** for every FK relationship: an `order → customer` edge (following the FK) and a `customer → order` reverse edge. This bidirectional construction is standard practice because a unidirectional graph would prevent information from flowing from orders back to customers during GNN propagation. Second, the `Linear(-1, hidden_dim)` type-specific projections use PyG's lazy initialization, which automatically handles the fact that customers have 4-dimensional features while products have 6-dimensional features and orders have 1-dimensional features. Third, the prediction head is applied only to customer nodes — the GNN processes all nodes jointly (products, orders, order items, and customers), but only customer embeddings are supervised for the churn classification task.

---

## 17.9 Benchmark Results on RelBench

The following table reports results from the RelBench leaderboard on three representative tasks. The Feature Engineering + LightGBM baseline uses Featuretools automated deep feature synthesis (DFS) to generate hundreds of SQL-equivalent aggregates, providing a strong comparison point for what is achievable without graph learning.

| Method | rel-amazon (Avg. Precision ↑) | rel-hm NDCG@10 ↑ | rel-stack (Avg. Precision ↑) |
|---|---|---|---|
| Shallow: LightGBM + DFS features | 0.241 | 0.081 | 0.498 |
| SAGE on relational graph | 0.271 | 0.102 | 0.531 |
| HGT on relational graph | 0.284 | 0.111 | 0.549 |
| RelGNN | 0.312 | 0.134 | 0.568 |
| RelGNN + temporal encoding | **0.328** | **0.148** | **0.581** |

*Results are approximate; see the RelBench leaderboard (relbench.github.io) for current state-of-the-art numbers, as the benchmark continues to receive new submissions.*

The pattern across all three tasks is consistent: graph-based methods outperform the flat feature engineering baseline, and temporal encoding provides a substantial additional gain on top of the static graph model. The gain of GNN over DFS features is most pronounced for the link prediction task (rel-hm NDCG@10: 0.081 → 0.148), which requires discovering non-trivial structural relationships between customers and items that aggregation-based features cannot easily express. The churn classification task (rel-amazon) shows a smaller gain (0.241 → 0.328), reflecting that temporal aggregates like "number of orders in the past 90 days" are already strong signals that DFS captures adequately — the graph model's advantage comes primarily from incorporating the product-category co-purchase structure.

---

## 17.10 MicroSim: Table-to-Graph Converter

#### Diagram: Relational Schema to Heterogeneous Graph


<iframe src="../../sims/ch17-table-to-graph/main.html" width="100%" height="562px" scrolling="no"></iframe>
[Run Relational Schema to Heterogeneous Graph Fullscreen](../../sims/ch17-table-to-graph/main.html)

---

## 17.11 Common Pitfalls

!!! mascot-warning "Four Ways Relational Graph Learning Goes Wrong"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warning about common mistakes">
    Converting a database to a graph is straightforward in principle but error-prone in practice. The four failure modes that appear most frequently are: **(1) Temporal leakage** — including rows with timestamps after the prediction cutoff in the training graph; this is the most damaging error because the model learns to cheat on training data but degrades sharply at deployment time (see §17.5); **(2) Missing reverse edges** — constructing only FK-direction edges (order→customer) without their reverses (customer→order) prevents customers from aggregating information from their orders, which is typically the most informative signal for customer-level tasks; **(3) Feature encoding for high-cardinality categoricals** — naively one-hot encoding a product ID column with 100,000 distinct values creates 100,000-dimensional features that overwhelm the model; instead, use learned embeddings or hashed representations; **(4) Ignoring scale imbalance** — a customer with 500 orders has 500 incoming edges from the order nodes, while a customer with 2 orders has 2; mean aggregation handles this correctly (it normalizes by neighborhood size), but sum aggregation will make high-order customers dominate the loss and gradient signal, biasing the model toward predicting churn based on order count rather than behavioral patterns.

**The correct level of graph construction is the row, not the table.** A common conceptual mistake is to add one node per table (a node for "all customers," a node for "all orders") and one edge between tables (a "has-orders" edge from the customer super-node to the order super-node). This collapses all the individual-row information into a single node and is equivalent to computing global aggregates — exactly the information-destroying operation that RDL is designed to avoid. The graph must have one node per row, not per table.

**Null foreign keys require a design decision.** Not all FK columns are mandatory — some orders may have a NULL value in an optional FK column (e.g., a coupon\_id FK when no coupon was used). A NULL FK does not correspond to an edge in the graph — there is simply no edge for that row. This is the correct representation and should be handled by omitting the edge rather than by imputing a default target node. Imputing a default (e.g., linking all NULL-FK orders to a dummy node) creates artificial connections that will corrupt the GNN's representation of the dummy node.

**Basis decomposition becomes important at large schema scale.** Enterprise databases with 50–100 tables and hundreds of FK relationships generate an equivalently large set of meta-relations. Without basis decomposition (§15.5), the per-meta-relation weight matrices account for millions of parameters that overfit rapidly on schemas where most meta-relations appear in only a few rows. The heuristic \( B = \lceil\sqrt{|\mathcal{R}|}\rceil \) from Chapter 15 applies directly here.

---

## 17.12 Exercises

The following 12 exercises span all six levels of Bloom's taxonomy.

**Remember**

1. Define the relational-to-graph construction formally, specifying how tables, rows, columns, and foreign keys map to node types, nodes, node features, and edges. What are the two directions in which edges can be oriented for a FK relationship, and why are both typically included?
2. List the four table types in the e-commerce database example (Customers, Orders, OrderItems, Products) and state the node type, approximate feature dimensionality (before GNN processing), and task relevance for each.

**Understand**

3. Explain why temporal leakage is a more severe and harder-to-detect problem in relational databases than in standard graph benchmarks. Give a specific example of how a FK join across a timestamp boundary could expose future information to the training model.
4. The RelBench feature engineering baseline uses Featuretools DFS (Deep Feature Synthesis). Explain what DFS computes, how it relates to multi-hop graph aggregation, and at what depth DFS aggregation becomes impractical.

**Apply**

5. Given the following three-table schema — Library (library\_id, city), Borrowers (borrower\_id, library\_id FK, age), Loans (loan\_id, borrower\_id FK, book\_id FK, loan\_date) — draw the entity-relationship diagram and then specify the heterogeneous graph construction: list all node types, all meta-relations (with direction), and the column values that become node features for each node type.
6. For the e-commerce churn prediction task with prediction time \( T \), implement a SQL query (or Python/Pandas equivalent) that extracts the flat feature engineering baseline: for each customer, compute the number of orders in the 90 days before \( T \), average order value in that window, and the number of distinct products purchased. Then explain which relational patterns this feature set cannot capture.

**Analyze**

7. Compare the RDL approach (rows → nodes, FK → edges, GNN) to the traditional feature engineering approach (SQL aggregates → feature vector → gradient boosting) on four dimensions: (a) sensitivity to schema changes (adding or removing a table), (b) ability to capture 3-hop relational patterns, (c) interpretability of the learned model, (d) computational cost for a database with 50 tables and 100M rows.
8. The RelBench results show that temporal encoding provides a larger performance gain over the static graph baseline for some tasks than others. Hypothesize which properties of a prediction task make temporal recency encoding most and least impactful, and explain the mechanism by which temporal encoding introduces those gains.

**Evaluate**

9. A data scientist argues: "RDL is just automated feature engineering — instead of writing SQL to join and aggregate tables, we let the GNN do the joins implicitly. The result should be the same as a sufficiently expressive DFS." Evaluate this claim carefully: in what sense is it correct, and in what sense does GNN-based RDL provably compute representations that DFS cannot, even with unlimited depth?
10. The RelGNN architecture handles junction tables via path-compressed edges (§17.7.2). Critically evaluate this design decision: what information is preserved and what is lost when the three-node path \( A \to J \to B \) is compressed into a single \( A \to B \) edge? Under what conditions would the uncompressed three-node path be strictly preferable?

**Create**

11. Design an RDL system for a hospital Electronic Health Record (EHR) database with the following tables: Patients (patient\_id, age, diagnosis\_codes), Visits (visit\_id, patient\_id FK, physician\_id FK, visit\_date, department), Prescriptions (rx\_id, visit\_id FK, drug\_code, dosage), Physicians (physician\_id, specialty). The prediction task is 30-day hospital readmission. Specify: (a) the complete heterogeneous graph construction, (b) the temporal cutoff mechanism, (c) which node type(s) should be predicted on, (d) any patient privacy concerns that affect the graph representation design.
12. Propose an extension to the RelGNN temporal encoding that distinguishes between two events at the same time-to-cutoff but with different temporal patterns: a customer who placed 10 orders in the last week (high recency, concentrated) versus one who placed 10 orders uniformly over the past year (high recency for the most recent order but sparse overall). Formalize the extended encoding, explain what additional information it captures, and describe how you would train it.

---

## 17.13 Further Reading

**Relational Deep Learning: Graph Representation Learning on Relational Databases** — Fey, Hu, Huang, Lenssen, Rajan, Robinson, Throckmorton, Yue (2024). The foundational paper establishing the relational-to-graph construction as a principled framework, providing formal definitions, the temporal leakage analysis, and empirical results across multiple relational learning tasks. The mathematical treatment of why GNNs on the relational graph are strictly more expressive than depth-bounded feature aggregation is particularly valuable.

**RelBench: A Benchmark for Deep Learning on Relational Databases** — Robinson, Rajan, Hu, Fey, Leskovec (2024). The benchmark paper, documenting the seven RelBench datasets, the evaluation protocol, and the baseline implementations. The appendix containing data construction details for each dataset is essential reading for practitioners who want to understand the temporal structure and schema diversity of the benchmark.

**Graph Neural Networks for Relational Databases** — Cvitkovic (2020). An earlier independent formulation of the relational-to-graph construction, predating the RelBench framework but anticipating many of its key ideas. Particularly useful for its discussion of how different database normalization forms (1NF, 2NF, 3NF) affect the structure of the constructed graph, and how denormalization for performance optimization can inadvertently remove graph structural information.

**Entity Embeddings of Categorical Variables** — Guo, Berkhahn (2016). Provides the foundational treatment of learned embeddings for categorical database columns, which underlies the initial feature encoding step in the RDL pipeline. The paper demonstrates on structured tabular data that learned embeddings consistently outperform one-hot encoding for high-cardinality categoricals, a finding directly applicable to the node feature initialization in relational graphs.

**Featuretools: An Open Source Framework for Automated Feature Engineering** — Kanter, Veeramachaneni (2015). Documents the automated feature engineering system used as the RDL baseline in RelBench. Reading this alongside the RelBench results reveals precisely which patterns DFS captures (depth-2 SQL aggregates) and misses (structural patterns, long relational chains), clarifying why GNN-based methods provide additional value.

**Temporal Graph Networks for Deep Learning on Dynamic Graphs** — Rossi, Chamberlain, Frasca, Eynard, Monti, Bronstein (2020). The TGN paper, introducing memory modules for continuous-time temporal graphs — the natural extension of the static relational graph construction to settings where entity representations should update as new events arrive. RelBench's temporal evaluation protocol is directly inspired by TGN's temporal graph formulation, and TGN is one of the baselines evaluated in the RelBench temporal tasks.

**Deep Feature Synthesis: Towards Automating Data Science Endeavors** — Kanter, Veeramachaneni (2015). The original DFS paper, presenting the algorithm for automatically generating features from relational databases by stacking aggregation and transformation primitives along join paths. Understanding DFS's computational complexity (exponential in join depth) makes the case for GNN-based alternatives more concrete: DFS at depth 4 on a 10-table schema generates \( O(10^4) \) candidate features, of which most are irrelevant, versus the GNN's learned and compact relational propagation.

---

!!! mascot-celebration "You've Linked the Tables"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrating">
    The insight at the heart of Relational Deep Learning — that a relational database is already a graph, and its foreign keys are already edges — is one of those ideas that seems obvious once stated and yet took decades of separate development in database research and graph learning research before anyone formalized it explicitly. With this insight, the GNN toolkit developed across Chapters 6–16 becomes immediately applicable to the most abundant form of structured data in the world: not benchmark datasets curated for machine learning competitions, but the operational databases that run hospitals, banks, retailers, and governments. In Chapter 18, we shift focus from how information propagates through graphs to how graphs themselves are organized into communities — a question that is central to understanding social networks, biological systems, and the clustering structure of any large-scale relational system.

[See Annotated References](./references.md)
