# Quiz: Label Propagation and Semi-Supervised Learning

Test your understanding of label propagation, belief propagation, influence models, and epidemic dynamics on graphs.

---

#### Question 1

Label propagation on a graph starts with a few labeled seed nodes. What is the core rule applied at each iteration?

<div class="upper-alpha" markdown>
1. Each unlabeled node is assigned the label of its highest-degree neighbor
2. Each node adopts the label of the globally most frequent class across all its neighbors
3. Labeled nodes propagate their labels to all nodes within exactly two hops
4. Each node updates its label distribution to the weighted average of its neighbors' label distributions
</div>

??? question "Show Answer"
    The correct answer is **D**. In each label propagation iteration, every unlabeled node \(u\) updates its label score vector \(f_u \leftarrow \sum_{v \in \mathcal{N}(u)} (w_{uv} / d_u)\, f_v\), where \(w_{uv}\) is the edge weight and \(d_u\) is \(u\)'s weighted degree. This is a local weighted average of neighbors' current label distributions. Labeled seed nodes are clamped to their true labels throughout. After sufficient iterations, the algorithm converges to the harmonic solution of the graph Laplacian energy minimization.

    **Concept Tested:** Label Propagation

---

#### Question 2

In the SIR epidemic model on a network, what do the three compartments S, I, and R represent?

<div class="upper-alpha" markdown>
1. Susceptible (can be infected), Infected (currently spreading), and Recovered (immune/removed)
2. Start, Iterate, and Recover — the three phases of the power iteration algorithm
3. Source nodes, Intermediate relay nodes, and Receiving leaf nodes in a tree graph
4. Static, Interactive, and Relational — the three types of graph node attributes
</div>

??? question "Show Answer"
    The correct answer is **A**. The SIR model partitions all nodes into three states: S (susceptible — healthy and can become infected), I (infected — currently contagious and spreading), and R (recovered/removed — no longer susceptible or infectious). Transitions occur stochastically: S→I with rate \(\beta\) per infected neighbor, I→R with rate \(\gamma\). The basic reproduction number \(R_0 = \beta/\gamma\) determines whether an epidemic grows (\(R_0 > 1\)) or dies out (\(R_0 < 1\)).

    **Concept Tested:** SIR Epidemic Model

---

#### Question 3

The Independent Cascade Model of influence propagation on a network proceeds in discrete steps. How does influence spread at each step?

<div class="upper-alpha" markdown>
1. Every neighbor of every active node becomes active with probability equal to the global average edge weight
2. Each newly activated node v attempts to activate each inactive neighbor u independently with probability \(p_{v,u}\); each activation attempt happens exactly once
3. All nodes within k hops of any active node become simultaneously active at step k
4. Each active node activates its single highest-probability neighbor and then becomes inactive
</div>

??? question "Show Answer"
    The correct answer is **B**. In the Independent Cascade Model, when node v first becomes active it gets exactly one chance to activate each currently inactive neighbor u with probability \(p_{v,u}\). Crucially, each edge is tried at most once — if v fails to activate u, no future activation of other nodes can trigger v to try u again. Influence propagation is stochastic and cascades outward, but the "one chance" rule prevents double-counting and makes the model memoryless.

    **Concept Tested:** Independent Cascade Model

---

#### Question 4

Influence maximization asks: given a budget of k seed nodes, which k nodes should be initially activated to maximize expected spread? Why is solving this problem exactly NP-hard?

<div class="upper-alpha" markdown>
1. Because the number of possible k-node seed sets grows as C(N,k), making exhaustive search infeasible for large N
2. Because the graph must first be converted to a tree structure, which requires \(O(N^3)\) time
3. Because the SIR model is only defined for continuous-time dynamics, making discrete optimization impossible
4. Because influence maximization requires eigenvalue decomposition of the full N × N adjacency matrix
</div>

??? question "Show Answer"
    The correct answer is **A**. The search space for \(k\) seed nodes has \(\binom{N}{k}\) possible subsets. For \(N = 1\) million and \(k = 50\), this is astronomically large. The influence maximization problem was proven NP-hard by Kempe, Kleinberg, and Tardos (2003). However, the expected spread function is monotone submodular, and a greedy algorithm that adds the node with the highest marginal gain at each step achieves a \((1 - 1/e) \approx 63\%\) approximation guarantee — the best possible in polynomial time.

    **Concept Tested:** Influence Maximization

---

#### Question 5

Belief propagation passes messages along edges of a graphical model. What is the message \(\mu_{v \to u}\) intuitively communicating?

<div class="upper-alpha" markdown>
1. The degree of node v normalized by the total degree of the graph
2. Node v's current belief about its own label, passed directly to u unchanged
3. What node v "believes" about node u's state, based on v's own local evidence and messages received from v's other neighbors
4. The probability that the edge (v, u) exists in the underlying true graph
</div>

??? question "Show Answer"
    The correct answer is **C**. The message \(\mu_{v \to u}(x_u)\) conveys node \(v\)'s assessment of the probability that \(u\) is in state \(x_u\), computed from \(v\)'s own observation and the messages \(v\) received from all its other neighbors (excluding \(u\) — to avoid circular reasoning). When all messages converge, each node's belief is proportional to the product of its local evidence and all incoming messages. On trees, belief propagation is exact; on loopy graphs it is an approximation.

    **Concept Tested:** Belief Propagation

---

#### Question 6

Label propagation minimizes the energy \(E(f) = \sum_{(u,v) \in E} w_{uv} (f_u - f_v)^2\). What does this energy functional penalize?

<div class="upper-alpha" markdown>
1. Label assignments where connected nodes have very different label scores
2. Nodes that have label values far from 0 or 1
3. Seed nodes whose labels differ from the graph's majority class
4. Unlabeled nodes that are more than two hops from any seed
</div>

??? question "Show Answer"
    The correct answer is **A**. The energy E penalizes large differences in label values between nodes connected by high-weight edges. Minimizing E encourages smoothness: nearby nodes (high edge weight) should have similar label assignments. The labeled seed nodes act as constraints (boundary conditions), and the harmonic solution that minimizes E with those constraints is exactly the fixed point of the iterative label propagation algorithm. This connection to the graph Laplacian makes label propagation a principled semi-supervised method.

    **Concept Tested:** Label Propagation

---

#### Question 7

The Linear Threshold Model of influence differs from the Independent Cascade Model in what fundamental way?

<div class="upper-alpha" markdown>
1. The Linear Threshold Model assigns influence weights only to outgoing edges; ICM uses only incoming edge weights
2. The Linear Threshold Model uses continuous-time differential equations; ICM uses discrete time steps
3. In the Linear Threshold Model, a node becomes active when the total weight of active neighbors exceeds a threshold; activation is deterministic once the threshold is crossed
4. The Linear Threshold Model only applies to bipartite graphs; ICM works on all graph types
</div>

??? question "Show Answer"
    The correct answer is **C**. In the Linear Threshold (LT) Model, each node \(v\) has a threshold \(\theta_v\) (often drawn uniformly from \([0,1]\)). Node \(v\) activates when \(\sum_{u \in \mathcal{N}(v),\, u\text{ active}} w_{uv} \geq \theta_v\) — i.e., when the cumulative influence from active neighbors crosses the threshold. Unlike ICM (which is stochastic per edge), the LT model is deterministic given the thresholds. LT captures the idea that peer pressure or social proof activates adoption once enough neighbors have adopted.

    **Concept Tested:** Linear Threshold Model

---

#### Question 8

In fraud detection on a transaction graph, why is label propagation preferable to training a classifier on node features alone?

<div class="upper-alpha" markdown>
1. Label propagation requires no computation, while feature-based classifiers are computationally expensive
2. Label propagation can process graphs with more than one million nodes; feature classifiers cannot
3. Label propagation works only when node features are missing, which is typical in fraud data
4. Label propagation exploits the homophily assumption — fraudulent accounts cluster together — using graph structure that feature classifiers cannot access
</div>

??? question "Show Answer"
    The correct answer is **D**. Graph-based methods like label propagation exploit homophily: fraudulent accounts tend to transact with other fraudulent accounts. A flat feature classifier treats each account independently, ignoring this guilt-by-association signal. Label propagation spreads the "fraud" label through the transaction graph to accounts that haven't been manually verified but are structurally connected to known fraudsters. The graph structure itself is a powerful signal that feature engineering alone cannot capture.

    **Concept Tested:** Label Propagation, Homophily (referenced from Ch04)

---

#### Question 9

What is \(R_0\) in the SIR model, and what does \(R_0 = 1\) represent as a threshold?

<div class="upper-alpha" markdown>
1. \(R_0\) is the ratio of recovered to susceptible nodes at the end of the epidemic
2. \(R_0\) is the network's average degree; \(R_0 = 1\) means the graph is a tree
3. \(R_0 = \beta/\gamma\) is the basic reproduction number; \(R_0 = 1\) marks the epidemic threshold below which outbreaks die out and above which they grow
4. \(R_0\) is the number of seed nodes required to start an epidemic in an Erdős–Rényi random graph
</div>

??? question "Show Answer"
    The correct answer is **C**. The basic reproduction number \(R_0 = \beta/\gamma\) measures the average number of new infections caused by a single infected node in a fully susceptible population. When \(R_0 < 1\), each infected node infects fewer than one new node on average, so the outbreak dies out exponentially. When \(R_0 > 1\), infections grow exponentially — an epidemic. The threshold \(R_0 = 1\) is the critical point (phase transition) separating these regimes, and it has direct implications for vaccination strategies and network intervention.

    **Concept Tested:** SIR Epidemic Model

---

#### Question 10

Label propagation converges to the harmonic solution, which minimizes Laplacian smoothness energy while satisfying label constraints. What makes this solution unique?

<div class="upper-alpha" markdown>
1. It is the only solution where every node's label equals the majority vote of its neighbors
2. It is the minimum-norm solution in the reproducing kernel Hilbert space induced by the graph Laplacian, guaranteed to exist and be unique for connected graphs with at least one labeled node per connected component
3. It assigns the highest label confidence to nodes with the highest betweenness centrality
4. It is unique only when all edge weights are equal (unweighted graph)
</div>

??? question "Show Answer"
    The correct answer is **B**. The harmonic solution \(f^*\) minimizes \(f^\top L f\) subject to the constraint that labeled nodes have their true labels. For a connected graph with at least one labeled node, the Laplacian sub-matrix restricted to unlabeled nodes is positive definite, guaranteeing a unique global minimum. The solution is the harmonic extension of the boundary conditions — the unlabeled nodes collectively form a "smooth interpolation" between the labeled seed values through the graph.

    **Concept Tested:** Label Propagation, Belief Propagation
