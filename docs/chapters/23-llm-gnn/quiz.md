# Quiz: LLMs and GNNs: Text-Attributed Graphs and Joint Training

Test your understanding of text-attributed graphs, LLM+GNN integration, and graph foundation models.

---

#### Question 1

A text-attributed graph assigns a text string to each node. What is the fundamental challenge of applying standard GNNs to text-attributed graphs?

<div class="upper-alpha" markdown>
1. Standard GNNs assume fixed-dimensional node features, but different nodes may have text of very different lengths and semantics — a generic bag-of-words encoding loses semantic richness that LLMs can capture
2. Text features are too high-dimensional for graph attention to process efficiently
3. Standard GNNs cannot process variable-length text strings as node features
4. Standard GNNs require node features to be numeric, and text strings cannot be converted to numbers
</div>

??? question "Show Answer"
    The correct answer is **A**. Standard GNNs expect a fixed-dimensional feature vector per node. Text can be converted to bag-of-words or TF-IDF vectors (fixed-dim, but ignoring semantics and word order) or pre-trained LLM embeddings (rich semantics, but treating each node independently without graph context). The challenge is that neither approach alone is optimal: bag-of-words loses semantics, and LLM-only encoding ignores graph structure. The LLM+GNN integration research question is how to combine the semantic richness of LLMs with the structural reasoning of GNNs.

    **Concept Tested:** Text-Attributed Graph, LLM + GNN Integration

---

#### Question 2

The "LLM-as-encoder" paradigm uses a frozen or fine-tuned LLM to produce node features, which are then passed to a downstream GNN. What is the key limitation of using a completely frozen LLM?

<div class="upper-alpha" markdown>
1. Frozen LLMs cannot process text longer than 512 tokens, limiting node descriptions
2. Frozen LLMs require 100× more GPU memory than fine-tuned LLMs
3. A frozen LLM produces generic text embeddings that are not optimized for the specific graph task — the LLM has no knowledge of graph structure or the downstream prediction objective, potentially misaligning text representations with what the GNN needs
4. A frozen LLM cannot produce embeddings for text that contains scientific jargon
</div>

??? question "Show Answer"
    The correct answer is **C**. A frozen LLM produces embeddings calibrated for general text understanding tasks (next token prediction, masked language modeling). These may not emphasize the aspects of text most relevant for graph node classification. For example, in an academic citation network, a frozen LLM embeds abstract content without knowing that "GNN" and "graph neural network" should be highly similar for this specific task. Fine-tuning (TAPE, G-Texts) adapts the LLM's text representations to the graph task, typically improving node classification accuracy.

    **Concept Tested:** LLM + GNN Integration

---

#### Question 3

Graph instruction tuning adapts an LLM to answer questions about graph structure by including graph-structured context in the prompt. What is the key difference from standard LLM fine-tuning?

<div class="upper-alpha" markdown>
1. Graph instruction tuning uses larger batch sizes and higher learning rates than standard fine-tuning
2. Graph instruction tuning replaces the LLM's attention mechanism with graph attention
3. Graph instruction tuning serializes graph structure (adjacency, node attributes, subgraphs) into text format as part of the instruction, training the LLM to reason about relational structure rather than just text content
4. Graph instruction tuning only works for knowledge graph question answering, not for general graphs
</div>

??? question "Show Answer"
    The correct answer is **C**. Graph instruction tuning (methods like InstructGLM) creates training examples where the input is a text-formatted graph description — e.g., "Node 1 (Abstract: 'GNNs for drug discovery') is connected to Node 2 (Abstract: 'Drug-protein binding'). What is the research topic of Node 1?" — and the output is the correct label. The LLM learns to extract both textual content and structural context from the serialized graph. This is fundamentally different from standard fine-tuning on text: the model must learn to interpret node neighborhoods described in natural language.

    **Concept Tested:** Graph Instruction Tuning

---

#### Question 4

TAPE (Text and Graph Learning) uses an LLM to first predict node labels via text, then uses those predictions as additional features for a downstream GNN. What does this two-stage approach achieve?

<div class="upper-alpha" markdown>
1. It allows the GNN to incorporate both the LLM's text understanding (as soft labels or explanations) and the graph's structural information, combining semantic and relational signals in a single representation
2. It eliminates the need for any labeled data by using the LLM as a zero-shot labeler
3. It reduces the dimensionality of LLM embeddings to match the GNN's expected feature dimension
4. It prevents the LLM from being fine-tuned on the graph task, preserving general text understanding
</div>

??? question "Show Answer"
    The correct answer is **A**. TAPE's two-stage approach: (1) an LLM reads each node's text description and produces a probability distribution over classes (soft label) plus a natural language explanation; (2) the GNN uses both the original LLM embeddings and the soft label predictions as node features, then applies message passing over the graph structure. This allows the GNN to propagate not just text semantics but also the LLM's class predictions — effectively using the LLM's understanding as a prior that is refined by graph structure.

    **Concept Tested:** LLM + GNN Integration, Text-Attributed Graph

---

#### Question 5

A "graph foundation model" aims to train a single GNN that can transfer across diverse graph datasets without task-specific retraining. What is the primary obstacle to building such a model compared to language model foundation models?

<div class="upper-alpha" markdown>
1. GNNs have too few parameters to benefit from pre-training at scale
2. Different graphs have incompatible node feature spaces (different attributes, different schemas) and different graph structures — there is no shared "vocabulary" analogous to text tokens that enables direct parameter sharing across all graph domains
3. Graph datasets are too small to provide the training signal needed for foundation models
4. GNNs cannot be pre-trained with self-supervised objectives because graphs have no natural "next token" prediction task
</div>

??? question "Show Answer"
    The correct answer is **B**. Language models share a universal vocabulary — any text in any domain maps to the same token set, enabling direct pre-training across all corpora. Graphs lack this: a molecular graph has atom features (element, charge), a citation graph has TF-IDF vectors, a social graph has demographic features. Node feature spaces are incompatible across domains, and graph structures vary wildly (molecule: 15 nodes; road network: 1M nodes). Building graph foundation models requires either domain-specific pre-training (ULTRA for KGs) or structural feature alignment across graphs.

    **Concept Tested:** Graph Foundation Model

---

#### Question 6

One-For-All (OFA) is a graph foundation model that uses LLMs to create a unified node feature space. How does it achieve this?

<div class="upper-alpha" markdown>
1. It uses an LLM to convert all node attributes (regardless of type — molecular properties, paper abstracts, user profiles) into natural language descriptions, producing a unified semantic embedding space that works across all graph domains
2. It trains a single GNN jointly on all datasets using multi-task learning with a shared output layer
3. It normalizes all node features to zero mean and unit variance before cross-dataset training
4. It uses random feature projection to map all node features to the same dimensionality
</div>

??? question "Show Answer"
    The correct answer is **A**. OFA's key insight: any node attribute can be verbalized as text (an atom becomes "Carbon atom in an aromatic ring with charge 0"), and an LLM converts all verbalized descriptions to a shared semantic embedding space. A paper abstract, a molecule's SMARTS description, and a user's profile can all be embedded in the same space using the LLM's semantic understanding. This unified feature space then enables a single GNN to perform message passing on graphs from different domains, enabling cross-dataset transfer.

    **Concept Tested:** Graph Foundation Model, Text-Attributed Graph

---

#### Question 7

Joint fine-tuning of LLM+GNN involves updating both the LLM encoder and the GNN simultaneously. What specific training challenge does this create?

<div class="upper-alpha" markdown>
1. Joint fine-tuning always leads to worse performance than training each component separately
2. Joint fine-tuning requires the entire graph to fit in GPU memory for the LLM's attention computation
3. Joint fine-tuning cannot use the Adam optimizer because the LLM requires second-order optimization
4. The LLM has billions of parameters while the GNN has millions — gradient magnitudes differ by orders of magnitude, causing instability; learning rate scheduling, gradient clipping, and LoRA/adapter layers are needed to stabilize joint training
</div>

??? question "Show Answer"
    The correct answer is **D**. Joint LLM+GNN training is challenging because: (1) an LLM may have 7B+ parameters vs. a GNN's few million, creating gradient scale mismatches; (2) the LLM's gradients flow through the GNN's loss and can destabilize the LLM's pre-trained representations (catastrophic forgetting); (3) memory constraints from holding both a large LLM and a full graph simultaneously. Solutions include using LoRA for parameter-efficient LLM fine-tuning (few trainable parameters), gradient clipping, different learning rates for LLM vs. GNN, and frozen LLM layers for the first training epochs.

    **Concept Tested:** LLM + GNN Integration

---

#### Question 8

In graph instruction tuning, graphs can be represented as sequences of tokens in different ways. What is the key trade-off between linearizing the full adjacency as text vs. using GNN embeddings as soft tokens?

<div class="upper-alpha" markdown>
1. Full adjacency text is more accurate; GNN soft tokens are faster to compute
2. Full adjacency text is interpretable; GNN soft tokens cannot be decoded back to graph structure
3. Full adjacency text requires graph canonicalization; GNN soft tokens are permutation invariant
4. Full adjacency text grows quadratically with graph size and quickly exceeds LLM context limits; GNN soft tokens compress graph structure into a fixed number of vectors, enabling efficient processing of large graphs
</div>

??? question "Show Answer"
    The correct answer is **D**. Encoding a 1000-node graph as adjacency text ("Node 1 connects to 2, 5, 7, 18...") produces thousands of tokens — far exceeding a typical 4K-32K context window, and scaling as O(N + E). GNN soft tokens compress the entire graph into a fixed set of d-dimensional vectors (e.g., one vector per community or one vector for the whole graph), which are then prepended to the text prompt as "virtual tokens." This compression enables LLMs to reason about large graphs within their context window constraints.

    **Concept Tested:** Graph Instruction Tuning, LLM + GNN Integration

---

#### Question 9

The citation network ogbn-arxiv is a text-attributed graph where each paper node has an abstract. Why does using an LLM to embed abstracts significantly outperform bag-of-words features for node classification?

<div class="upper-alpha" markdown>
1. LLM embeddings have higher dimensionality than bag-of-words, providing more information
2. LLMs capture semantic similarity (e.g., "graph neural network" and "GNN" are similar) and long-range context (the beginning of an abstract informs the meaning of the end), while bag-of-words treats all words as independent and equal
3. LLM embeddings are always better than bag-of-words regardless of the downstream task
4. Bag-of-words cannot handle abstracts with more than 500 words, while LLMs can
</div>

??? question "Show Answer"
    The correct answer is **B**. Bag-of-words represents each word independently: "graph neural network" and "GNN" are treated as completely different vocabulary items with no similarity. LLMs capture: (1) semantic equivalence (synonyms and abbreviations map to similar embeddings); (2) contextual meaning (the same word has different embeddings in different contexts); (3) long-range dependencies (the claim in sentence 1 is informed by the evidence in sentence 5). For research paper classification, this semantic understanding dramatically improves the quality of node features passed to the GNN.

    **Concept Tested:** Text-Attributed Graph

---

#### Question 10

Graph foundation models represent an emerging paradigm. What distinguishes a "graph foundation model" from a pre-trained GNN in the classical sense?

<div class="upper-alpha" markdown>
1. Graph foundation models are trained with reinforcement learning; pre-trained GNNs use supervised learning
2. Graph foundation models have more parameters than 1 billion; pre-trained GNNs are always smaller
3. A graph foundation model can transfer to multiple downstream tasks across different graph domains with minimal or no fine-tuning; classical pre-trained GNNs are pre-trained on one domain and fine-tuned on a specific task within that domain
4. Graph foundation models use Transformer architectures; pre-trained GNNs use message passing
</div>

??? question "Show Answer"
    The correct answer is **C**. Classical pre-training (e.g., GNN pre-trained on molecular property prediction) creates a better initialization for a specific molecular downstream task. A graph foundation model aspires to work across domains and tasks without fine-tuning — like GPT-4 answering both coding questions and creative writing. ULTRA is a KG foundation model (one model, many KGs); OFA is a cross-domain foundation model (molecules + citation + social). The defining property is cross-domain zero-shot or few-shot transfer, not just same-domain fine-tuning.

    **Concept Tested:** Graph Foundation Model
