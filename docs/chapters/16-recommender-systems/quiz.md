# Quiz: GNNs for Recommender Systems

Test your understanding of collaborative filtering, matrix factorization, LightGCN, and PinSage.

---

#### Question 1

Collaborative filtering predicts user preferences based on the assumption that similar users like similar items. In the graph view, what structure does this correspond to?

<div class="upper-alpha" markdown>
1. A directed graph where edges point from items to users who purchased them
2. A bipartite user-item interaction graph where users and items are two disjoint node sets and edges encode observed interactions
3. A homogeneous graph where all users and items are the same node type
4. A hypergraph where each edge connects one user to multiple items simultaneously
</div>

??? question "Show Answer"
    The correct answer is **B**. Collaborative filtering naturally maps to a bipartite graph \(B = (U, I, E)\) where user nodes \(U\) and item nodes \(I\) are disjoint, and each edge \((u, i) \in E\) indicates that user \(u\) interacted with item \(i\) (purchased, rated, clicked). Two users are "similar" if their neighborhoods in this bipartite graph overlap (they interacted with many common items). GNNs on this bipartite graph propagate collaborative signals through multi-hop paths: user \(u\)'s 2-hop neighborhood includes all items rated by users similar to \(u\).

    **Concept Tested:** Recommender System (Graph), Collaborative Filtering

---

#### Question 2

Classical matrix factorization decomposes the user-item interaction matrix \(R \approx P \cdot Q^\top\). What do the factor matrices \(P\) and \(Q\) represent?

<div class="upper-alpha" markdown>
1. \(P\) is the user adjacency matrix; \(Q\) is the item adjacency matrix
2. \(P\) encodes explicit user ratings; \(Q\) encodes implicit user interactions
3. \(P\) is the training set interaction matrix; \(Q\) is the test set interaction matrix
4. \(P \in \mathbb{R}^{|U| \times d}\) contains latent factor vectors for each user; \(Q \in \mathbb{R}^{|I| \times d}\) contains latent factor vectors for each item; their dot product scores how well item \(i\) matches user \(u\)'s preferences
</div>

??? question "Show Answer"
    The correct answer is **D**. Matrix factorization embeds each user \(u\) as \(\mathbf{p}_u \in \mathbb{R}^d\) and each item \(i\) as \(\mathbf{q}_i \in \mathbb{R}^d\). The predicted preference score is \(\hat{r}_{u,i} = \mathbf{p}_u \cdot \mathbf{q}_i\). Training minimizes the reconstruction loss on observed interactions: \(\sum_{(u,i) \in E} (r_{u,i} - \mathbf{p}_u \cdot \mathbf{q}_i)^2\). The latent factors capture unobserved characteristics: a high third-factor value for both a user and a movie might encode "enjoys science fiction." Collaborative signals are captured because users who liked similar items learn similar factor vectors.

    **Concept Tested:** Matrix Factorization (Rec)

---

#### Question 3

LightGCN simplifies GCN for recommendation by removing feature transformation and non-linear activation. What is the LightGCN propagation rule for user \(u\)?

<div class="upper-alpha" markdown>
1. \(\mathbf{e}_u^{(l+1)} = \sum_{i \in \mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}} \mathbf{e}_i^{(l)}\) (normalized aggregation, no weight matrix or activation)
2. \(\mathbf{e}_u^{(l+1)} = \text{MLP}\!\left(\text{CONCAT}\!\left(\mathbf{e}_u^{(l)},\ \text{mean}(\{\mathbf{e}_i : i \in \mathcal{N}(u)\})\right)\right)\)
3. \(\mathbf{e}_u^{(l+1)} = \sigma\!\left(W \cdot \mathbf{e}_u^{(l)} + \sum_{i \in \mathcal{N}(u)} \frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}} \mathbf{e}_i^{(l)}\right)\)
4. \(\mathbf{e}_u^{(l+1)} = \mathbf{e}_u^{(l)} + \text{attention}(\{\mathbf{e}_i : i \in \mathcal{N}(u)\})\)
</div>

??? question "Show Answer"
    The correct answer is **A**. LightGCN (He et al., 2020) ablated standard GCN and found that weight matrices and non-linearities hurt recommendation performance while adding computation. The resulting propagation rule is simply symmetric-normalized aggregation of neighbors: for user \(u\), aggregate all connected item embeddings; for item \(i\), aggregate all connected user embeddings. The final representation is a weighted sum across all \(L\) layers: \(\mathbf{e}_u = \sum_{l=0}^{L} \alpha_l \mathbf{e}_u^{(l)}\). This simplicity, combined with Bayesian personalized ranking (BPR) loss, achieves state-of-the-art results.

    **Concept Tested:** LightGCN

---

#### Question 4

Neural Collaborative Filtering (NCF) extends matrix factorization by replacing the dot product with an MLP. What specific limitation of the dot product does this address?

<div class="upper-alpha" markdown>
1. The dot product cannot be computed for sparse matrices; MLP handles sparsity naturally
2. The dot product cannot handle binary interaction signals; MLP supports continuous ratings
3. The dot product assumes linear interaction between user and item latent factors; MLP can capture non-linear, complex interaction patterns that simple dot products cannot represent
4. The dot product does not support training on implicit feedback; MLP handles both explicit and implicit
</div>

??? question "Show Answer"
    The correct answer is **C**. The dot product \(\mathbf{p}_u \cdot \mathbf{q}_i = \sum_k p_{u,k} q_{i,k}\) is a bilinear model — it can only represent preferences that decompose as a sum of linear factor products. If a user's preference for an item depends on a non-linear combination of latent factors (e.g., "I like action movies, but only if the runtime is under 2 hours"), the dot product cannot capture this interaction. NCF concatenates user and item embeddings and passes them through an MLP, enabling arbitrary non-linear preference modeling.

    **Concept Tested:** Neural Collaborative Filter

---

#### Question 5

PinSage was developed at Pinterest to recommend pins (images with associated text) to users. What specific engineering challenge motivated its design, and how does it address it?

<div class="upper-alpha" markdown>
1. Pinterest pins have both text and image features; PinSage uses separate encoders for each modality before GNN aggregation
2. Pinterest's graph has 10 billion edges; PinSage uses random-walk-based importance sampling to focus computation on the most relevant neighbors rather than using the full neighborhood
3. Pinterest's user graph is directed; PinSage extends LightGCN to handle directed edges
4. Pinterest requires real-time inference; PinSage caches precomputed embeddings in a distributed key-value store
</div>

??? question "Show Answer"
    The correct answer is **B**. Pinterest's graph has 3 billion nodes and 18 billion edges — far too large for standard full-batch GNN training. PinSage addresses scalability via importance-based neighbor sampling using random walks: for each node, it runs short random walks and selects the top-k most visited neighbors as the "effective neighborhood." This focuses computation on the most structurally relevant neighbors rather than uniformly sampling. PinSage also uses mini-batch training with on-the-fly feature computation, enabling industrial-scale deployment.

    **Concept Tested:** PinSage

---

#### Question 6

The "cold start" problem in recommender systems refers to difficulty recommending items or to new users. How does a GNN-based approach handle new items better than pure matrix factorization?

<div class="upper-alpha" markdown>
1. GNNs solve the cold start problem by assigning the new item the average embedding of all existing items
2. GNNs use item metadata (description, category) as initial node features, allowing representation even before any interactions are observed
3. GNNs require at least 10 interactions before generating an embedding; matrix factorization requires 100
4. GNNs handle cold start by querying external knowledge bases for item metadata
</div>

??? question "Show Answer"
    The correct answer is **B**. Matrix factorization assigns a learned embedding per item — a new item with no interactions has no embedding. GNN methods like PinSage initialize new item representations from their content features (image embedding, text features, category) and compute GNN representations by aggregating from structurally similar items in the graph. A new pin, for example, can be embedded based on the visual and textual similarity to existing pins without needing any user interactions — partially solving cold start.

    **Concept Tested:** Recommender System (Graph), Collaborative Filtering

---

#### Question 7

LightGCN's final representation is a weighted sum of embeddings from all \(L\) layers: \(\mathbf{e} = \sum_l \alpha_l \mathbf{e}^{(l)}\). What is the typical choice of \(\alpha_l\) and why?

<div class="upper-alpha" markdown>
1. \(\alpha_l = 1/l^2\) (inverse-square decay, emphasizing early layers)
2. \(\alpha_l = 1\) for all \(l\) (uniform sum, giving all layers equal weight)
3. \(\alpha_l = 1/(L+1)\) for all \(l\) (uniform average across layers), which empirically outperforms learned weights and eliminates over-fitting risk
4. \(\alpha_l\) is learned via attention over layer outputs
</div>

??? question "Show Answer"
    The correct answer is **C**. He et al. (2020) empirically found that uniform averaging (\(\alpha_l = 1/(L+1)\)) performs as well or better than learned layer weights. This is surprising but practical: equal weighting provides a natural ensemble over different receptive field sizes — layer 0 captures self-features, layer 1 captures direct interactions, layer 2 captures 2-hop collaborative signals. Learning \(\alpha_l\) adds parameters and can overfit, while the simple uniform average already captures the multi-scale collaborative signal. LightGCN typically uses \(L \in \{2, 3, 4\}\).

    **Concept Tested:** LightGCN

---

#### Question 8

Bayesian Personalized Ranking (BPR) is the loss function used to train LightGCN. What is the training signal it uses?

<div class="upper-alpha" markdown>
1. Pairwise ranking: for each observed interaction \((u, i)\), sample an unobserved item \(j\) and maximize the score gap \(\hat{r}_{u,i} - \hat{r}_{u,j}\)
2. Mean squared error between predicted ratings and observed star ratings
3. Binary cross-entropy on observed vs. unobserved interactions, treating each independently
4. Triplet loss: push the user embedding closer to interacted items than to a random negative
</div>

??? question "Show Answer"
    The correct answer is **A**. BPR optimizes for relative ranking rather than absolute prediction. For each (user \(u\), positive item \(i\)) pair, it samples a negative item \(j\) (one \(u\) hasn't interacted with) and maximizes \(P(\hat{r}_{u,i} > \hat{r}_{u,j})\) via the log-sigmoid: \(\mathcal{L} = -\sum \log \sigma(\hat{r}_{u,i} - \hat{r}_{u,j})\). This reflects the actual recommendation goal: ranking items the user would like above items they wouldn't, rather than predicting exact rating values. BPR naturally handles implicit feedback (clicks, purchases) where no explicit ratings are available.

    **Concept Tested:** LightGCN, Collaborative Filtering

---

#### Question 9

PinSage uses a "curriculum training" strategy that progressively increases the difficulty of negative samples. Why does this improve recommendation quality?

<div class="upper-alpha" markdown>
1. Curriculum training reduces the total number of training examples, speeding up convergence
2. Easy negatives (random pins) provide sufficient gradient signal throughout training
3. Easy random negatives are trivially distinguished from positives early in training, providing weak gradient signal; harder negatives (visually or semantically similar but not relevant) force the model to learn finer distinctions
4. Progressive negatives prevent the model from memorizing training data by changing examples each epoch
</div>

??? question "Show Answer"
    The correct answer is **C**. Early in training, random negative samples (arbitrary pins from the entire corpus) are clearly different from positives — the model easily distinguishes them with large margin, generating tiny gradients. As training progresses, PinSage introduces "hard negatives": items that are close in embedding space to the positive item (visually similar pins from different categories, for example). These harder negatives force the model to learn precise boundaries. This curriculum mirrors how humans learn: easy examples first, then progressively harder distinctions.

    **Concept Tested:** PinSage

---

#### Question 10

Popularity bias is a well-known problem in collaborative filtering systems. Why does GNN propagation potentially amplify rather than reduce this bias?

<div class="upper-alpha" markdown>
1. GNNs use softmax normalization that inherently favors popular items
2. Popularity bias only affects content-based filtering, not collaborative filtering
3. GNNs cannot represent long-tail items because they have few edges and therefore near-zero representations
4. Popular items have high degree in the user-item graph; during multi-hop propagation, popular items appear in many users' neighborhoods, causing their embeddings to dominate the aggregation and making them even more likely to be recommended
</div>

??? question "Show Answer"
    The correct answer is **D**. In the user-item bipartite graph, popular items have high degree (many users have interacted with them). During GNN propagation, high-degree items appear in many users' 1-hop neighborhoods and even more users' 2-hop neighborhoods. The symmetric normalization \(1/\sqrt{d_u d_i}\) reduces but does not eliminate this effect. Popular items receive more gradient updates, their embeddings are more stable (lower variance), and they dominate aggregation. Addressing popularity bias requires explicit debiasing techniques like inverse propensity scoring or causal deconfounding.

    **Concept Tested:** Recommender System (Graph)
