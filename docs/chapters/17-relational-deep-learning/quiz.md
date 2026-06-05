# Quiz: Relational Deep Learning

Test your understanding of relational databases as graphs, RelBench, RelGNN, and the RDL framework.

---

#### Question 1

Relational Deep Learning (RDL) claims that "every relational database is already a graph." What specific database feature makes this claim precise?

<div class="upper-alpha" markdown>
1. Database tables can be stored as adjacency matrices in PostgreSQL
2. Foreign key constraints between tables define edges: each foreign key value in table T₁ pointing to a primary key in table T₂ creates a directed edge from the corresponding row in T₁ to the row in T₂
3. SQL JOIN operations are equivalent to graph traversal operations in all database engines
4. Row indices in database tables directly correspond to node identifiers in standard graph libraries
</div>

??? question "Show Answer"
    The correct answer is **B**. Every foreign key FK(T₁.col → T₂.pk) defines an edge from each row in T₁ to the referenced row in T₂. In an e-commerce database: orders.customer_id → customers.id becomes an edge from each order node to the customer node who placed it; order_items.order_id → orders.id becomes an edge from each line item to its parent order. The schema (table names, column types) defines the node and edge types of a heterogeneous graph. No additional architectural ideas are needed — the database IS the graph.

    **Concept Tested:** Relational Database as Graph, Relational Deep Learning

---

#### Question 2

The traditional ML pipeline for relational data involves feature engineering via SQL joins. What structural information does this approach lose compared to directly applying a GNN to the relational graph?

<div class="upper-alpha" markdown>
1. Column-level statistics like mean and standard deviation cannot be computed via SQL joins
2. Manual feature engineering flattens multi-hop relational structure into a single row, losing higher-order patterns (e.g., the friends of a customer's friends, or the category distribution of all items purchased by similar users)
3. SQL joins cannot handle tables with more than 100 columns
4. Manual features require labeled data for all rows, while GNNs work with semi-supervised labels
</div>

??? question "Show Answer"
    The correct answer is **B**. A flat feature row can capture "this customer's total spend" (1-hop join) but not "the purchase patterns of customers who bought similar products to this customer" (3-hop path through order → product ← order ← customer). Multi-hop relational patterns — the kind that GNNs naturally compute through layered message passing — require either deeply nested subqueries or explicit hand-engineering. GNNs on the relational graph automatically discover and use these patterns without manual specification, which is the core value proposition of RDL.

    **Concept Tested:** Relational Deep Learning

---

#### Question 3

RelBench provides a standardized benchmark for relational deep learning. What types of prediction tasks does it include?

<div class="upper-alpha" markdown>
1. Only graph-level classification tasks across multiple databases
2. Link prediction tasks for inferring missing foreign keys between tables
3. Node-level prediction tasks (predicting properties of individual rows/entities) across diverse real-world relational databases from multiple domains
4. Schema inference tasks for predicting the structure of an unknown database
</div>

??? question "Show Answer"
    The correct answer is **C**. RelBench includes node-level prediction tasks: given a target table row (e.g., a customer, a product, a user), predict a property of that row (e.g., will this customer churn? what is this product's future sales?). Tasks span diverse domains — e-commerce (Amazon reviews), social (Stack Overflow), medical (clinical trials) — with varying table counts, schemas, and temporal structures. Each task uses the full relational graph as input rather than flattened features, enabling apples-to-apples comparison of RDL methods.

    **Concept Tested:** RelBench (Benchmark)

---

#### Question 4

RelGNN is the reference GNN architecture for RelBench. It applies heterogeneous message passing to the relational graph. What is the role of the entity encoder in RelGNN?

<div class="upper-alpha" markdown>
1. It predicts which entities will be connected by new foreign keys in future database updates
2. It converts raw column values (strings, numbers, dates, categoricals) from each table row into a unified initial feature vector for that entity's GNN node
3. It maps entity names to pre-trained word embeddings from a language model
4. It normalizes all numerical columns to zero mean and unit variance before GNN processing
</div>

??? question "Show Answer"
    The correct answer is **B**. Each row in a relational database has heterogeneous column types: strings (product name), integers (order quantity), floats (price), dates (timestamp), categoricals (status). RelGNN's entity encoder processes these column-level features with type-appropriate sub-encoders (tokenizer → MLP for text, numeric scaler for numbers, embedding lookup for categoricals) and concatenates them into a unified node feature vector. This "table-to-vector" step is what allows the standard heterogeneous GNN message passing to operate on database rows.

    **Concept Tested:** RelGNN

---

#### Question 5

Temporal leakage is a critical concern in RelBench evaluation. What is temporal leakage and how does RelBench prevent it?

<div class="upper-alpha" markdown>
1. Using test labels to compute training features, which RelBench prevents by using separate train/test tables
2. Leaking the schema information from the target table to the feature tables, prevented by masking target columns during GNN processing
3. Training on data from multiple time periods that introduce distribution shift, prevented by using a single year of data
4. Using future data (rows created after the prediction timestamp) as features when predicting a past target — RelBench defines strict temporal cutoffs so GNN aggregation only uses rows with timestamps before the prediction time
</div>

??? question "Show Answer"
    The correct answer is **D**. In temporal prediction tasks (e.g., "predict this customer's purchases in the next month"), you must not use future transactions as features. Standard database joins without time-awareness would include post-prediction-date rows, inflating performance with inadmissible signals. RelBench defines each prediction example with a precise cutoff timestamp t*: the GNN may only aggregate rows where timestamp < t* when computing features for a prediction at time t*. This faithful temporal masking is what separates rigorous RDL evaluation from naive feature engineering pipelines.

    **Concept Tested:** RelBench (Benchmark)

---

#### Question 6

A key challenge in applying GNNs to relational databases is handling multi-table schemas with varying numbers of foreign keys. How does the heterogeneous GNN framework handle this naturally?

<div class="upper-alpha" markdown>
1. Each foreign key relationship becomes a distinct edge type in the heterogeneous graph; type-specific GNN weight matrices handle the varying semantics of different cross-table relationships without any schema-specific code
2. By padding all tables to the same number of columns before GNN processing
3. By running a separate GNN for each pair of connected tables and averaging the results
4. By converting all foreign keys to a single "reference" edge type to simplify the architecture
</div>

??? question "Show Answer"
    The correct answer is **A**. The heterogeneous GNN framework (Chapter 15) assigns a distinct weight matrix W_r to each edge type r. In the relational graph, each foreign key relationship (e.g., "order → customer") becomes a distinct edge type with its own weight matrix. The GNN can thus learn different aggregation semantics for "aggregate the items in this order" vs. "aggregate the orders of this customer" vs. "aggregate the reviews of this product" — all within a single unified model, without any table-specific engineering.

    **Concept Tested:** Relational Deep Learning, RelGNN

---

#### Question 7

The RDL framework processes relational databases as graphs without requiring explicit feature engineering. What is the primary argument for why this approach can outperform traditional AutoML on tabular data?

<div class="upper-alpha" markdown>
1. GNNs are always more accurate than gradient boosting for tabular data
2. RDL requires less computational resources than gradient boosting on large tables
3. Traditional AutoML treats each row independently, missing relational signals; RDL automatically captures cross-table patterns (e.g., "customers who bought the same products have similar churn rates") without manual SQL feature specification
4. Traditional AutoML cannot handle missing values; RDL imputes them via graph diffusion
</div>

??? question "Show Answer"
    The correct answer is **C**. Methods like XGBoost and LightGBM treat tabular data as independent rows — each prediction depends only on that row's columns. They are very effective when all relevant information is in a single row, but struggle when the signal is relational (e.g., "how many similar users churned last month?"). Manually computing these relational features requires domain expertise and SQL engineering. RDL's GNN automatically discovers multi-hop relational patterns during end-to-end training, potentially uncovering higher-order interactions that no human would think to engineer manually.

    **Concept Tested:** Relational Deep Learning

---

#### Question 8

A product recommendation task in RelBench asks: "Given a user's purchase history, predict which product they will purchase next." How does GNN message passing on the relational graph naturally capture the relevant signals?

<div class="upper-alpha" markdown>
1. Multi-hop propagation from the user node through past orders to products (and their categories, brands, reviews) automatically aggregates collaborative filtering signals and content features into the user's representation
2. The GNN predicts the product directly from the user's demographic features in the customers table
3. The GNN uses attention over the product catalog to select the highest-scoring item
4. The GNN aggregates only direct user-product edges (1-hop), ignoring longer-range patterns
</div>

??? question "Show Answer"
    The correct answer is **A**. Starting from a user node, 1-hop aggregation captures direct purchase history (orders linked by customer_id). 2-hop aggregation reaches the products in those orders (via order_items → products). 3-hop aggregation reaches other users who bought the same products (collaborative filtering signal) and product attributes like categories and brands (content signal). The GNN combines all these signals in a unified representation without any manual feature specification — the relational graph structure directly encodes the information flow.

    **Concept Tested:** Relational Deep Learning, Relational Database as Graph

---

#### Question 9

RelBench includes tasks from multiple domains (e-commerce, social, medical). Why is cross-domain generalization important for evaluating RDL methods?

<div class="upper-alpha" markdown>
1. Methods that work only on specific schemas overfit to particular table structures; cross-domain results reveal whether a method's inductive biases (e.g., heterogeneous GNN architecture) generalize to arbitrary relational structures
2. Cross-domain evaluation tests whether the method can share parameters between completely different databases
3. Medical databases have stricter privacy requirements, making them harder to work with
4. Cross-domain tasks require larger models, testing scalability rather than generalization
</div>

??? question "Show Answer"
    The correct answer is **A**. A method that achieves state-of-the-art on Amazon review data by exploiting specific product-review-user structure might fail on Stack Overflow (question-answer-user structure) or clinical trial data (patient-visit-diagnosis structure). RelBench's diversity tests whether architectural choices — how to represent heterogeneous node types, how to handle temporal ordering, how to aggregate across different foreign key multiplicities — are genuinely general rather than tailored to a specific domain. Strong cross-domain performance is the mark of a truly general RDL framework.

    **Concept Tested:** RelBench (Benchmark)

---

#### Question 10

The RDL framework can be seen as a generalization of existing GNN-based methods. Which of the following methods is a special case of RDL applied to a specific two-table schema?

<div class="upper-alpha" markdown>
1. PageRank applied to a web graph
2. NBFNet applied to a knowledge graph
3. GCN applied to a citation network
4. LightGCN for recommender systems, which is RDL applied to a two-table schema of users and items with a single interaction table
</div>

??? question "Show Answer"
    The correct answer is **D**. LightGCN operates on a user-item interaction graph derived from two tables: a users table, an items table, and an interactions table. The interactions table's foreign keys (user_id, item_id) define the bipartite graph edges. This is exactly the RDL setup with three tables and two foreign key relationships. RDL generalizes this to arbitrary numbers of tables, arbitrary schema structures, and arbitrary foreign key multiplicities — LightGCN is the degenerate case with the simplest possible relational schema.

    **Concept Tested:** Relational Deep Learning, Relational Database as Graph
