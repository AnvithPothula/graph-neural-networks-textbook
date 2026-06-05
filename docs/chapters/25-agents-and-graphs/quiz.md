# Quiz: Agents and Graphs

Test your understanding of agent memory graphs, tool-use graphs, scene graphs, and graph-structured AI systems.

---

#### Question 1

In agent architectures, representing agent memory as a knowledge graph provides specific advantages over a flat list or vector memory. What is the primary advantage?

<div class="upper-alpha" markdown>
1. Relational structure in the memory allows multi-hop retrieval: the agent can answer "who worked at the company that made the drug that treats this disease?" without storing an explicit triple for that specific question
2. Graph memory requires less storage than flat list memory for the same number of facts
3. Graph memory is always faster to query than vector databases because it uses exact matching
4. Graph memory guarantees that all stored facts are logically consistent
</div>

??? question "Show Answer"
    The correct answer is **A**. A flat list stores individual facts (entity, relation, entity) independently — answering a multi-hop question requires either pre-computing all chains or searching all fact combinations. A knowledge graph memory stores the same facts as nodes and edges, enabling multi-hop traversal: follow edges from entity A to B to C, answering chained queries naturally. An agent reasoning over a KG memory can answer questions that require combining 3-5 facts, something that flat retrieval handles poorly without explicit multi-hop indexing.

    **Concept Tested:** Agent Memory as Graph

---

#### Question 2

A tool-use graph represents an AI agent's available tools as nodes and their dependencies or interaction patterns as edges. What does an edge (tool_A, tool_B) typically represent?

<div class="upper-alpha" markdown>
1. Tool A and Tool B have identical functionality and can be used interchangeably
2. Tool A costs more tokens to call than Tool B
3. The output of Tool A is often used as input to Tool B — a data dependency or sequential usage pattern that the agent should consider when planning multi-step tool calls
4. Tool A must be called before Tool B in every possible tool-use sequence
</div>

??? question "Show Answer"
    The correct answer is **C**. In a tool-use graph, edges encode workflow dependencies: "search_web → extract_information → summarize" represents a common pipeline. The agent can use the tool-use graph as a planning aid: given a query, traverse the graph to find which tools need to be chained, in what order, and how to pass outputs between them. This graph-structured representation of tool capabilities is more powerful than a flat list of tool descriptions because it encodes the compositional structure of complex workflows.

    **Concept Tested:** Tool-Use Graph

---

#### Question 3

Scene graphs are used in visual AI to represent the objects and relationships detected in an image. What information do scene graph nodes and edges encode?

<div class="upper-alpha" markdown>
1. Nodes encode pixel coordinates of detected objects; edges encode color similarity between nearby objects
2. Nodes represent image patches at different resolutions; edges connect patches that are spatially adjacent
3. Nodes represent objects detected in the image (with category, bounding box, and attribute features); edges represent spatial or semantic relationships between object pairs (e.g., "cat sitting_on chair", "table left_of window")
4. Nodes represent image frames in a video; edges represent temporal transitions between frames
</div>

??? question "Show Answer"
    The correct answer is **C**. A scene graph G = (O, R) has nodes O representing detected objects with attributes (object type: "cat", color: "orange", size: "small") and edges R representing typed relationships between object pairs ("cat is sitting_on mat", "person is holding cup"). Scene graphs enable symbolic visual reasoning: a model can answer "Is the cat above or below the table?" by traversing the scene graph rather than re-processing the full image. GNNs on scene graphs support visual question answering, image captioning, and visual relationship detection.

    **Concept Tested:** Scene Graph

---

#### Question 4

Multi-hop reasoning agents traverse a knowledge graph to answer complex questions. What is the "confidence propagation" challenge in multi-hop graph traversal?

<div class="upper-alpha" markdown>
1. The agent must recompute confidence scores for every node in the graph at each hop
2. Confidence scores from the knowledge graph embedding model cannot be used for reasoning
3. Uncertainty compounds across hops: if each reasoning step has 80% confidence, a 3-hop chain has only \(0.8^3 = 51\%\) confidence — tracking and propagating uncertainty through the chain is essential for reliable answers
4. Multi-hop reasoning always achieves 100% confidence because the knowledge graph only contains verified facts
</div>

??? question "Show Answer"
    The correct answer is **C**. In multi-hop reasoning, errors at each step multiply: a 5-hop chain with 90% per-step accuracy has only \(0.9^5 \approx 59\%\) overall accuracy. Tracking uncertainty is critical for the agent to know when to seek additional evidence, retry with alternative paths, or decline to answer. Neural Bellman-Ford networks and beam search agents maintain per-path confidence scores that propagate through each reasoning step, allowing the agent to prefer high-confidence paths and flag uncertain answers.

    **Concept Tested:** Agent Memory as Graph, Tool-Use Graph

---

#### Question 5

Agent memory graphs often use a temporal dimension to track when information was acquired or last accessed. Why is temporal ordering important for agent memory management?

<div class="upper-alpha" markdown>
1. Recent information is often more reliable and relevant than stale information — temporal ordering enables recency-based retrieval, forgetting of outdated facts, and identification of information that may need updating
2. Temporal ordering enables the agent to answer "what happened on date X?" queries
3. Temporal ordering converts the static knowledge graph into a temporal graph, enabling TGN-style processing
4. Temporal ordering is required for the graph to remain a valid DAG (directed acyclic graph)
</div>

??? question "Show Answer"
    The correct answer is **A**. An agent operating over time accumulates information at different timestamps: a fact learned an hour ago may contradict one learned yesterday (world changed), or a retrieved document may have been published in 2019 (outdated). Temporal ordering allows the agent to: (1) prefer more recently acquired facts when there are conflicts; (2) decay the relevance of old information (similar to TGN's time encoding); (3) identify potentially stale facts (high age + domain where things change quickly = review needed). Without temporal structure, the agent cannot reason about information currency.

    **Concept Tested:** Agent Memory as Graph

---

#### Question 6

In a tool-use planning scenario, a GNN can be used to predict the optimal tool-use sequence. What input features would node and edge embeddings typically encode?

<div class="upper-alpha" markdown>
1. Node embeddings encode tool capabilities (input/output types, cost, latency, success rate); edge embeddings encode the frequency of tool-A-then-tool-B sequences in historical agent traces and the input-output compatibility between the tools
2. Node embeddings encode tool names as word embeddings; edge embeddings encode edge existence (0 or 1)
3. Node embeddings encode the tool's code implementation; edge embeddings encode API call overhead
4. Both node and edge embeddings are uniformly zero until the agent executes a tool and receives feedback
</div>

??? question "Show Answer"
    The correct answer is **A**. For GNN-based tool planning, informative features are: (1) node features — what does each tool do, what are its input/output types, how reliable/fast is it based on past usage?; (2) edge features — how often have pairs of tools been chained successfully in past agent traces, and whether the output schema of tool A is compatible with the input schema of tool B. A GNN trained on historical agent traces can learn to predict which tool to call next given the current task state and past tool outputs represented as a partial tool-use subgraph.

    **Concept Tested:** Tool-Use Graph

---

#### Question 7

Scene graph generation from images is itself a structured prediction problem. What makes it harder than object detection?

<div class="upper-alpha" markdown>
1. Scene graph generation requires higher-resolution images than object detection
2. Scene graph generation must predict both object labels AND pairwise relationship types for all object pairs — the number of candidate relationships is \(O(N^2)\) and relationships involve compositional reasoning ("the horse is jumping over the fence" requires understanding both objects and their spatial configuration)
3. Scene graph generation cannot use convolutional neural networks because the output is a graph
4. Scene graph generation requires a pre-trained knowledge graph to define all possible relationship types
</div>

??? question "Show Answer"
    The correct answer is **B**. Object detection locates and classifies individual objects — \(N\) predictions for \(N\) objects. Scene graph generation additionally requires predicting relationships for all \(O(N^2)\) object pairs, where each relationship is a typed edge (e.g., "riding," "holding," "left_of," "made_of"). The relationship type depends on both objects' features, their relative positions, and semantic context. For \(N=50\) objects, there are approximately 2450 candidate pairs. The combinatorial explosion, combined with long-tail relationship distributions (some relations appear rarely), makes scene graph generation substantially harder than detection.

    **Concept Tested:** Scene Graph

---

#### Question 8

How does representing an agent's plan as a graph differ from representing it as a linear sequence of steps?

<div class="upper-alpha" markdown>
1. Graph plans require symbolic reasoning; linear plans use neural sequence models
2. Graph plans are always shorter than linear plans for the same task
3. Graph plans can only represent DAG-structured tasks; linear plans handle all task types
4. Graph plans allow parallel execution of independent sub-tasks (nodes with no dependency edges can run simultaneously); linear plans enforce strict sequential execution
</div>

??? question "Show Answer"
    The correct answer is **D**. A linear plan executes step 1, then step 2, then step 3 — inherently sequential. A graph plan (task dependency graph) encodes which steps depend on which: if task B depends on task A, they must be sequential; if tasks C and D are both independent of A, they can execute in parallel after A completes. For a complex multi-step task (research + write + illustrate + format a report), a graph plan enables parallel tool invocations wherever dependencies permit, reducing total wall-clock time from O(T_total) to O(T_critical_path).

    **Concept Tested:** Tool-Use Graph, Agent Memory as Graph

---

#### Question 9

Knowledge graphs used as long-term memory for LLM agents have been shown to outperform flat text memory for specific query types. For which query type is graph memory most clearly superior?

<div class="upper-alpha" markdown>
1. Semantic similarity queries (find facts semantically similar to this sentence)
2. Open-ended synthesis queries (summarize what this author believed about neural networks)
3. Direct lookup queries (what is the capital of France?)
4. Multi-hop relational queries (who is the author of the paper that introduced the method used by the system described in document X?)
</div>

??? question "Show Answer"
    The correct answer is **D**. For direct lookups, a vector database or simple key-value store performs as well as a KG. For semantic similarity, vector databases with embedding search are optimal. For multi-hop relational queries, graph memory is clearly superior: the query requires chaining 3+ relations (paper → method → system → document), which is a natural graph traversal but requires complex retrieval-augmented reasoning from flat text. The relational structure in KG memory makes multi-hop queries tractable without exponential candidate expansion.

    **Concept Tested:** Agent Memory as Graph

---

#### Question 10

The integration of GNNs with agentic AI systems represents a convergence of two previously separate fields. What foundational property of GNNs makes them particularly well-suited for agent reasoning over graphs?

<div class="upper-alpha" markdown>
1. GNNs are computationally cheaper than LLMs for all reasoning tasks
2. GNNs are permutation equivariant and can naturally process relational structure — they can reason over the topology of an agent's memory, tool dependencies, or environment model in a way that is invariant to the arbitrary labeling of entities
3. GNNs always produce deterministic outputs, making agent behavior predictable and safe
4. GNNs can be deployed without GPU hardware, enabling agents to run on edge devices
</div>

??? question "Show Answer"
    The correct answer is **B**. The core value proposition of GNNs for agentic systems is permutation equivariance: the agent's reasoning should not depend on whether node 5 is labeled "Tool_Search" or "Tool_003" — it should depend on the tool's structural role in the tool graph. Similarly, memory queries should retrieve semantically relevant facts regardless of the arbitrary IDs assigned to KG nodes. This structural invariance, combined with the ability to propagate information through multi-hop paths, makes GNNs the natural computational primitive for agents that must reason over relational structures.

    **Concept Tested:** Agent Memory as Graph, Tool-Use Graph
