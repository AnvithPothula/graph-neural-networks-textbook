# Quiz: Theory of GNNs: Expressiveness and the WL Test

Test your understanding of GNN expressiveness, the WL hierarchy, over-squashing, and permutation equivariance.

---

#### Question 1

The 1-WL test (Weisfeiler-Lehman test) is used to assess graph isomorphism. What does it mean when two graphs are declared "non-distinguishable" by 1-WL?

<div class="upper-alpha" markdown>
1. The graphs have the same number of nodes and edges but different topologies
2. The 1-WL color histograms are identical at convergence — the test cannot rule out isomorphism, though the graphs may still be non-isomorphic
3. The graphs are provably isomorphic and any GNN will assign them identical representations
4. The graphs both contain regular subgraphs of the same degree
</div>

??? question "Show Answer"
    The correct answer is **B**. The 1-WL test iteratively refines node color labels based on neighbor multisets. If the final histograms differ, the graphs are definitively non-isomorphic. If they match, the test is inconclusive — it cannot prove isomorphism. GNNs are bounded by 1-WL: if 1-WL cannot distinguish two graphs, no message-passing GNN can assign them different representations. But 1-WL matching does not mean the graphs are isomorphic — there exist non-isomorphic graphs that fool 1-WL.

    **Concept Tested:** Weisfeiler-Leman Test (1-WL), GNN Expressiveness

---

#### Question 2

GIN (Graph Isomorphism Network) achieves the maximum expressiveness of any message-passing GNN bounded by 1-WL. What two properties must its aggregation function satisfy?

<div class="upper-alpha" markdown>
1. Lipschitz continuity and differentiability everywhere
2. Commutativity and associativity, ensuring the aggregation order does not matter
3. Injectivity over multisets (distinct multisets map to distinct outputs) and use of sum aggregation
4. Permutation invariance and output normalization to the unit sphere
</div>

??? question "Show Answer"
    The correct answer is **C**. Xu et al. (2019) prove that a GNN is as expressive as the 1-WL test if and only if its aggregation is injective over multisets — different multisets of neighbor representations map to different aggregate values. Sum aggregation over injective functions achieves this; mean and max do not (they can conflate distinct multisets). GIN implements this as \(\mathbf{h}_v^{(l+1)} = \text{MLP}\!\left((1+\varepsilon)\,\mathbf{h}_v^{(l)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)}\right)\), where the \((1+\varepsilon)\) term ensures the center node's contribution is distinguishable.

    **Concept Tested:** Graph Isomorphism Network (GIN), Sum Aggregation Injectivity

---

#### Question 3

Regular graphs are a known failure case for 1-WL and all message-passing GNNs. Why can't a GNN distinguish two non-isomorphic k-regular graphs?

<div class="upper-alpha" markdown>
1. Regular graphs are always isomorphic to each other, so no distinction is possible or needed
2. The adjacency matrix of a regular graph is always a permutation matrix, preventing eigenvalue analysis
3. Regular graphs have no cycles, making the WL color refinement trivially converge in one step
4. All nodes in a k-regular graph have the same degree k, so every node receives identical messages at each layer — no structural information differentiates nodes or distinguishes the graphs
</div>

??? question "Show Answer"
    The correct answer is **D**. In a k-regular graph, every node has exactly k neighbors. At layer 1 of WL, every node receives k copies of the initial color — the same aggregate for all nodes. After any number of layers, all nodes in any k-regular graph on the same number of nodes receive identical aggregate messages. Therefore, no message-passing GNN can assign different representations to nodes across two non-isomorphic k-regular graphs, even though the graphs have completely different structures.

    **Concept Tested:** WL Test Limitations (Regular Graphs), Aggregator Expressiveness (Sum vs. Mean vs. Max)

---

#### Question 4

Over-squashing in deep GNNs refers to information loss in long-range propagation. What causes it?

<div class="upper-alpha" markdown>
1. The gradient of the loss vanishing as it flows backward through more than three GNN layers
2. Applying too many non-linear activations, which progressively remove negative values from representations
3. Exponentially many nodes' information being compressed into a fixed-size vector as the receptive field grows exponentially with depth
4. Edge weights being normalized below a threshold, effectively removing edges from the computation
</div>

??? question "Show Answer"
    The correct answer is **C**. With \(L\) message passing layers and average degree \(d\), the \(L\)-hop neighborhood contains \(O(d^L)\) nodes — exponentially many. All their information must be compressed into a single fixed-dimensional vector at the center node. The "squashing" occurs because a \(d\)-dimensional vector has a fixed information capacity, yet it must summarize exponentially many inputs. Long-range signals from distant nodes are diluted to near-zero, making deep GNNs effectively blind to distant structural information.

    **Concept Tested:** Over-Squashing, Cheeger Constant

---

#### Question 5

The Cheeger constant h of a graph measures the quality of the minimum edge cut relative to volume. How does h relate to over-squashing?

<div class="upper-alpha" markdown>
1. Graphs with h = 0 (disconnected graphs) have no over-squashing because messages never cross component boundaries
2. Graphs with high Cheeger constant suffer more over-squashing because the graph is too well-connected for message passing to converge
3. The Cheeger constant measures the rate of gradient vanishing, not information squashing
4. Graphs with low Cheeger constant (bottleneck edges) suffer more over-squashing because information must pass through narrow cuts with exponentially diluted signals
</div>

??? question "Show Answer"
    The correct answer is **D**. The Cheeger constant h quantifies how "bottle-necked" a graph is. Low h means there is a small edge cut separating two large portions of the graph — information flowing between those portions must pass through few edges. In message passing, this creates over-squashing: a node on one side of the bottleneck must compress information from the entire other side through those few edges. Alon et al. (2021) formalized this connection, showing over-squashing is worst when the graph has low expansion (low h).

    **Concept Tested:** Cheeger Constant, Over-Squashing

---

#### Question 6

A GNN that is permutation equivariant satisfies \(f(P_\sigma \mathbf{X},\, P_\sigma \mathbf{A} P_\sigma^\top) = P_\sigma f(\mathbf{X}, \mathbf{A})\). What does this mean in plain terms?

<div class="upper-alpha" markdown>
1. The GNN produces the same output regardless of the input graph's topology
2. If you permute the node ordering, the GNN's output is permuted in the same way — the model is sensitive to node identity, not node index
3. The GNN produces the same scalar output for all permutations of the node ordering (permutation invariant, not equivariant)
4. The GNN requires node features to be sorted before processing
</div>

??? question "Show Answer"
    The correct answer is **B**. Permutation equivariance means that relabeling nodes (applying permutation \(\sigma\) to row/column indices) changes the output in the same consistent way. If node 3 and node 7 swap labels, the output representation for those nodes swaps accordingly — no information is created or destroyed. This is the correct property for node-level tasks. Graph-level tasks need permutation *invariance* (the output is unchanged under any relabeling), which can be achieved by applying a permutation-invariant readout after equivariant layers.

    **Concept Tested:** Permutation Equivariance, Permutation Invariance

---

#### Question 7

The k-WL hierarchy provides GNNs with greater expressiveness than 1-WL by using higher-order color refinement. What is the fundamental cost of moving from 1-WL to 2-WL?

<div class="upper-alpha" markdown>
1. 2-WL requires computing eigenvalues of the adjacency matrix, adding \(O(N^3)\) preprocessing
2. 2-WL can only process directed graphs, limiting its applicability
3. 2-WL assigns colors to pairs of nodes instead of individual nodes, increasing complexity from \(O(N)\) to \(O(N^2)\) per refinement step
4. 2-WL is not implementable as a neural network layer
</div>

??? question "Show Answer"
    The correct answer is **C**. 1-WL operates on individual nodes: the state space is \(N\)-dimensional. 2-WL operates on pairs of nodes: the state space is \(N^2\)-dimensional, requiring \(O(N^2)\) memory and \(O(N^3)\) or \(O(N^2 E)\) computation per step. Higher-order \(k\)-WL uses \(k\)-tuples of nodes with \(O(N^k)\) cost. This exponential blowup makes \(k\)-WL directly impractical for large graphs, motivating approximations like ring graph neural networks and subgraph GNNs that trade expressiveness for scalability.

    **Concept Tested:** k-WL Hierarchy

---

#### Question 8

Mean aggregation cannot distinguish the multiset {1, 1, 2} from {1, 2, 2} when the neighborhood sizes are different. Which specific counter-example demonstrates max aggregation's failure?

<div class="upper-alpha" markdown>
1. Max aggregation cannot distinguish a node with neighbors {1, 2, 3} from one with neighbors {3, 3, 3}
2. Max aggregation cannot distinguish a node with neighbors {5} from one with neighbors {5, 5, 5, 5}
3. Max aggregation cannot distinguish a node with neighbors {1, 1, 2} from one with neighbors {2, 2, 1}
4. Max aggregation cannot distinguish a node with neighbors {2, 3} from one with no neighbors
</div>

??? question "Show Answer"
    The correct answer is **B**. Max aggregation takes the element-wise maximum and ignores how many times a value appears. A node with one neighbor whose representation is 5 and a node with four neighbors all with representation 5 both produce max output 5 — the counts are erased. This means max aggregation cannot represent "how many neighbors have feature value X" — a critical piece of structural information. Sum preserves these counts; mean and max both lose them in different ways.

    **Concept Tested:** Aggregator Expressiveness (Sum vs. Mean vs. Max)

---

#### Question 9

GIN's update rule is \(\mathbf{h}_v^{(l+1)} = \text{MLP}\!\left((1+\varepsilon)\,\mathbf{h}_v^{(l)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)}\right)\). Why is the \((1+\varepsilon) \cdot \mathbf{h}_v^{(l)}\) term necessary?

<div class="upper-alpha" markdown>
1. It ensures the center node's own representation is distinguishable from its neighbors' contributions, preventing conflation of the center with an additional identical neighbor
2. It adds a residual connection that prevents over-smoothing in deep networks
3. It normalizes the self-contribution by the node's degree to prevent high-degree nodes from dominating
4. It scales the learning rate per node based on the local graph structure
</div>

??? question "Show Answer"
    The correct answer is **A**. Without the \((1+\varepsilon)\) term, GIN's aggregation would be \(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \mathbf{h}_u\) — treating the center node as just another neighbor. If a node \(v\) has a neighbor \(u\) with \(\mathbf{h}_u = \mathbf{h}_v\), the center's contribution is indistinguishable from adding one more copy of that neighbor's features. The \((1+\varepsilon)\) scalar (\(\varepsilon\) can be learned or set to 0) ensures \(v\)'s self-embedding has a different weight than its neighbors', preserving the distinction. This is crucial for achieving the 1-WL bound.

    **Concept Tested:** Graph Isomorphism Network (GIN), Sum Aggregation Injectivity

---

#### Question 10

Over-smoothing (spectral view) can be understood through the graph Laplacian's eigenvalues. What happens to the frequency components of node signals as GCN layers increase?

<div class="upper-alpha" markdown>
1. High-frequency components (sharp differences between neighbors) are progressively suppressed, leaving only the low-frequency global signal
2. All frequency components are amplified equally, causing representation norms to explode
3. High-frequency components are amplified while low-frequency components are suppressed
4. The signal alternates between high and low frequency at every other layer, causing oscillations
</div>

??? question "Show Answer"
    The correct answer is **A**. The GCN propagation matrix \(\tilde{D}^{-1/2} \tilde{A}\, \tilde{D}^{-1/2}\) has eigenvalues in \([-1, 1]\), with eigenvalue 1 for the smoothest (constant) signal and eigenvalue close to \(-1\) for the most oscillatory (high-frequency) signal. After \(L\) applications, eigenvalue \(\lambda_k\) is raised to the \(L\)-th power: \(\lambda_k^L \to 0\) for \(|\lambda_k| < 1\). Only the DC component (eigenvalue 1, constant signal) survives — all other frequency components decay exponentially. This is the spectral explanation for why deep GCNs make all node representations identical.

    **Concept Tested:** Over-Smoothing (Spectral View), Weisfeiler-Leman Test (1-WL)
