# Quiz: Temporal and Dynamic Graphs

Test your understanding of temporal graph representations, TGN, TGAT, and temporal graph applications.

---

#### Question 1

What is the key difference between a snapshot-based temporal graph representation and an event-based (continuous-time) representation?

<div class="upper-alpha" markdown>
1. Snapshot graphs store all historical edges; event-based graphs store only the most recent edges
2. Snapshot graphs use directed edges; event-based graphs use undirected interactions
3. Snapshot graphs require synchronized time steps while event-based graphs allow asynchronous updates
4. Snapshot graphs discretize time into intervals, creating a sequence of static graph snapshots; event-based graphs store each interaction as a timestamped event (u, v, t, features) without discretization
</div>

??? question "Show Answer"
    The correct answer is **D**. Snapshot models cut the timeline into windows (daily, weekly) and create a static graph for each window — simple but loses within-window ordering and requires choosing a window size. Event-based (continuous-time) models treat every interaction as a timestamped event (u, v, t, m) where m is optional message features. Events maintain their natural ordering and allow arbitrary time precision. TGN is an event-based model; methods like EvolveGCN use snapshots.

    **Concept Tested:** Temporal Graph, Dynamic Graph

---

#### Question 2

The Temporal Graph Network (TGN) maintains a memory state s_v for each node that persists between events. What does this memory capture?

<div class="upper-alpha" markdown>
1. A compressed summary of the node's interaction history — the long-term behavioral pattern of the node up to the current time
2. The node's current degree in the graph at the most recent snapshot
3. The node's static feature vector from the initial graph construction
4. The gradient of the loss function with respect to the node's embedding
</div>

??? question "Show Answer"
    The correct answer is **A**. TGN's memory \(\mathbf{s}_v \in \mathbb{R}^d\) is updated every time node \(v\) is involved in an event. The memory module (typically a GRU) processes the incoming message and updates: \(\mathbf{s}_v(t) = \text{MEM}(\mathbf{s}_v(t^-), \text{msg}_v(t))\) where \(\text{msg}_v(t)\) incorporates the interaction message and the other node's memory. Over time, \(\mathbf{s}_v\) accumulates a compressed history of all of \(v\)'s past interactions — who it interacted with, when, and with what messages. This persistent memory allows TGN to model long-term behavioral patterns that a memoryless GNN cannot capture.

    **Concept Tested:** Temporal GNN (TGN)

---

#### Question 3

TGAT (Temporal Graph Attention) uses a time encoding function \(\varphi(t)\) to incorporate temporal information. What property must \(\varphi(t)\) satisfy to work well in combination with the attention mechanism?

<div class="upper-alpha" markdown>
1. \(\varphi(t)\) must be monotonically increasing so recent events receive higher attention
2. \(\varphi(t)\) must map time to a high-dimensional vector (e.g., using Fourier features) so that time differences can be captured by inner products in the attention computation
3. \(\varphi(t)\) must normalize timestamps to \([0, 1]\) to prevent numerical overflow
4. \(\varphi(t)\) must be equal to zero at time \(t = 0\) to initialize the model correctly
</div>

??? question "Show Answer"
    The correct answer is **B**. TGAT encodes time using Bochner's theorem: \(\varphi(t) = [\cos(\omega_1 t + \phi_1), \ldots, \cos(\omega_d t + \phi_d)]\) — a vector of sinusoidal functions with different frequencies (time2vec-inspired). This allows the attention mechanism to compute time differences via dot products: \(\mathbf{q}(t_q) \cdot \mathbf{k}(t_k)\) captures the relative timing. The multi-frequency encoding allows the model to learn both short-term recency effects (high-frequency components) and long-term patterns (low-frequency components) simultaneously.

    **Concept Tested:** TGAT

---

#### Question 4

In traffic forecasting with GNNs, the road network is modeled as a graph. What do nodes and edges represent, and what is being predicted?

<div class="upper-alpha" markdown>
1. Nodes are traffic lights and edges represent synchronized signal groups; the GNN optimizes light timing
2. Nodes are vehicles and edges are road segments; the GNN predicts vehicle collision probability
3. Nodes are sensors at road locations and edges connect geographically close sensors; the GNN predicts future traffic speed or flow at each sensor, using both temporal patterns and spatial correlations via message passing
4. Nodes are city blocks and edges represent pedestrian crossing patterns; the GNN classifies congestion severity
</div>

??? question "Show Answer"
    The correct answer is **C**. In traffic forecasting (e.g., DCRNN, WaveNet+GCN), nodes are road sensors (loop detectors, GPS probes) and edges connect sensors that are road-adjacent or geographically near. The graph structure encodes spatial dependencies: traffic slowdowns at one sensor propagate to neighboring sensors. The prediction task is spatio-temporal: given the last T time steps of traffic speed measurements at all sensors, predict the next T' time steps at all sensors. Models combine GCN (spatial dependencies) with RNNs or attention (temporal patterns).

    **Concept Tested:** Traffic Forecasting (Graph), Temporal Graph

---

#### Question 5

TGN's training uses a "memory updater" that runs before each prediction. Why must memory be updated before computing the embedding, rather than after?

<div class="upper-alpha" markdown>
1. Updating after prediction would create circular dependencies in the computation graph
2. Post-update memory would violate causality by including information from the future event being predicted
3. Memory updating requires gradients from the prediction loss, which are not available before the forward pass
4. Memory must reflect the current state of the node before embedding computation — stale memory from before recent events would produce incorrect embeddings for the current prediction
</div>

??? question "Show Answer"
    The correct answer is **D**. Before computing node v's embedding for a prediction at time t, TGN first processes all events involving v that occurred before t to bring s_v up to date. If memory is stale (hasn't incorporated recent interactions), the embedding will not reflect v's current behavioral state — degrading prediction quality. The update must happen before embedding, not after, to ensure the embedding is computed from the most current state. After the prediction event is processed, the memory is updated again to incorporate the new interaction.

    **Concept Tested:** Temporal GNN (TGN)

---

#### Question 6

Dynamic graphs can be classified as "discretely dynamic" or "continuously dynamic." Which application domain typically uses continuously dynamic graphs?

<div class="upper-alpha" markdown>
1. Traffic forecasting, where sensor readings arrive at discrete hourly intervals
2. Citation networks, where papers are published at discrete moments and are static afterward
3. Financial transaction networks, where money transfers occur at millisecond granularity with precise timestamps
4. Social network graphs, where friendship graphs are typically updated in daily snapshots
</div>

??? question "Show Answer"
    The correct answer is **C**. Financial transaction networks are naturally continuous-time: each wire transfer, credit card transaction, or trade has an exact timestamp to millisecond precision, with thousands of events per second. This continuous stream of events is best modeled by event-based temporal graph methods like TGN. In contrast, traffic sensor networks typically produce readings at fixed intervals (every 5 minutes) and are better modeled with snapshot methods or discrete-time STGNNs. Citation graphs and social networks are even more discrete.

    **Concept Tested:** Temporal Graph, Dynamic Graph

---

#### Question 7

TGAT uses a temporal attention mechanism to aggregate a node's temporal neighborhood. How does it handle the variable number of past interactions at different time points?

<div class="upper-alpha" markdown>
1. It truncates all temporal histories to the k most recent interactions before attention computation
2. It pads all temporal neighborhoods to a fixed size using zero vectors
3. It uses multi-head attention over all past interactions, where each interaction is represented as a feature vector concatenated with its time encoding — the attention mechanism naturally handles variable-length histories
4. It averages all past feature vectors into a single vector before applying the time encoding
</div>

??? question "Show Answer"
    The correct answer is **C**. TGAT applies self-attention over a node's temporal neighborhood: all past interactions \(\{(\text{neighbor}_i, t_i, \text{features}_i)\}\) are encoded as keys/values and the current query is the target node at query time \(t_q\). Each past interaction is represented as \(\text{CONCAT}(\text{neighbor-embedding},\, \varphi(t_q - t_i))\) where \(\varphi\) is the time encoding. Multi-head attention over this variable-length sequence naturally assigns more weight to relevant past interactions. The attention mechanism's permutation invariance and variable-sequence handling make it ideal for asynchronous temporal neighborhoods.

    **Concept Tested:** TGAT

---

#### Question 8

What is the "transductive vs. inductive" challenge in temporal link prediction evaluation?

<div class="upper-alpha" markdown>
1. Transductive evaluation uses the full graph; inductive evaluation uses a randomly sampled subgraph
2. Transductive evaluation tests on node pairs seen during training; inductive evaluation tests on at least one new node not present during training, requiring the model to generalize to completely new entities
3. Transductive evaluation uses future timestamps; inductive evaluation uses past timestamps
4. Transductive evaluation requires labeled edges; inductive evaluation uses only unlabeled graph structure
</div>

??? question "Show Answer"
    The correct answer is **B**. In temporal link prediction, transductive evaluation tests whether the model can predict future interactions between nodes it has already observed during training. Inductive evaluation is stricter: test edges involve at least one completely new node with no history — the model must generalize its temporal reasoning to entities it has never seen. TGN achieves strong inductive performance because its memory module and GNN encoder can compute embeddings for new nodes from their first few interactions, without requiring any pre-trained entity-specific parameters.

    **Concept Tested:** Temporal GNN (TGN), Dynamic Graph

---

#### Question 9

Traffic forecasting models must capture both spatial and temporal dependencies simultaneously. How does a spatio-temporal GNN model both?

<div class="upper-alpha" markdown>
1. It alternates between graph convolution (spatial: aggregate information from neighboring sensors) and sequence modeling (temporal: capture trends, periodicities) — often as GCN layers interleaved with RNN or Transformer layers
2. It converts all temporal information into static node features before applying a standard GNN
3. It uses a separate GNN for space and a separate RNN for time, with no information sharing between them
4. It uses only the most recent timestep's data, ignoring temporal history beyond the last observation
</div>

??? question "Show Answer"
    The correct answer is **A**. Standard architectures for traffic forecasting (DCRNN, STGCN, Graph WaveNet) combine: (1) graph convolution over the road network to propagate spatial signals (a slowdown at sensor A affects sensor B downstream); (2) sequence modeling (RNN, temporal convolution, or attention) to capture temporal dynamics (rush-hour patterns, weekday vs. weekend). These two components are interleaved: GCN → TCN → GCN → TCN, allowing spatial patterns to inform temporal trends and vice versa.

    **Concept Tested:** Traffic Forecasting (Graph)

---

#### Question 10

TGN's message function computes a message \(m_v(t)\) for node v when it is involved in an event at time t. What inputs does the message function typically combine?

<div class="upper-alpha" markdown>
1. Only the edge features of the event (the content of the interaction)
2. The embeddings of the K-hop neighborhood of the node at the time of the event
3. Only the gradient of the loss function with respect to the node's previous memory
4. The current memory states of both interacting nodes, the raw event features, and the elapsed time since the node's last interaction — all combined via a learned function to produce the memory update signal
</div>

??? question "Show Answer"
    The correct answer is **D**. TGN's message function computes \(m_v(t) = \text{msg}(\mathbf{s}_v(t^-), \mathbf{s}_u(t^-), \Delta t, e_{uv}(t))\) where \(\mathbf{s}_v(t^-)\) is v's current memory, \(\mathbf{s}_u(t^-)\) is the interacting node u's memory, \(\Delta t = t - t_{\text{last}}(v)\) is the time elapsed since v's last event, and \(e_{uv}(t)\) is the event's raw feature vector. Combining all four components allows the message to encode: "what v knew before this event," "what u knew before this event," "how long v has been idle," and "what this specific interaction was about." The GRU memory updater then integrates this message into v's new memory.

    **Concept Tested:** Temporal GNN (TGN)
