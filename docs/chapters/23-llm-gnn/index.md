---
title: "Chapter 23: LLMs and GNNs: Text-Attributed Graphs and Joint Training"
description: "How large language models and graph neural networks combine to tackle text-attributed graphs — LLM-as-encoder, graph instruction tuning, and the emerging era of graph foundation models."
generated_by: claude skill chapter-content-generator
date: 2026-05-30 12:00:00
version: 0.08
---

# Chapter 23: LLMs and GNNs: Text-Attributed Graphs and Joint Training

**Part 6: Frontiers**

## Summary

Explores LLM+GNN integration for text-attributed graphs — LLM-as-encoder, joint fine-tuning, graph instruction tuning — and the One-For-All foundation model for cross-dataset transfer.

## Concepts Covered

This chapter covers the following 4 concepts from the learning graph:

1. Graph Foundation Model
2. LLM + GNN Integration
3. Text-Attributed Graph
4. Graph Instruction Tuning

## Prerequisites

This chapter builds on:

- [Chapter 6: GNN Foundations: Message Passing and GCN](../06-gnn-foundations/index.md)
- [Chapter 7: GNN Design Space: GraphSAGE and GAT](../07-gnn-design-space/index.md)
- [Chapter 11: Graph Transformers](../11-graph-transformers/index.md)
- [Chapter 12: Knowledge Graph Embeddings](../12-knowledge-graph-embeddings/index.md)

---

## 23.1 When Words and Graphs Meet

!!! mascot-welcome "Welcome to the Frontier"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waves at the chapter opening">
    Welcome to Chapter 23! You have spent chapters building up GNNs that operate on **structure** — edges, neighborhoods, adjacency matrices. But most real-world graphs carry something else entirely: **language**. Academic papers have abstracts. Products have descriptions. Social media users write posts. This chapter is about what happens when the representational power of large language models meets the structural reasoning of GNNs — and why the combination is more powerful than either alone.

Consider the **ogbn-arxiv** citation network you have seen throughout this textbook. Each node is an arXiv paper, and each directed edge represents a citation. The graph structure alone tells you *which* papers are cited by *which* others. But each paper also carries a title and abstract — dense natural language that describes the paper's contribution, methodology, and field. A GNN that ignores this text is working with one hand tied behind its back. A language model that ignores the citation structure misses the relational context that explains why a paper is influential.

The challenge is that these two sources of information live in different worlds. GNNs consume vectors — fixed-dimensional arrays of real numbers that represent nodes. Language models consume tokens — discrete units of text processed through attention layers tuned on billions of documents. Bridging these two worlds requires careful architectural decisions. Over the past several years, three distinct strategies have emerged: using a language model as a **feature extractor** that pre-processes text before the GNN sees it, using a language model as a **reasoner** that interprets graph structure expressed in natural language, and training both components **jointly** so they share representations end-to-end.

This chapter covers each strategy in depth. It then turns to the emerging question of **graph foundation models** — pre-trained systems that aspire to generalize across datasets, tasks, and domains the way large language models generalize across text corpora. By the end, you will understand why text-attributed graphs represent one of the richest frontiers in graph machine learning, and you will be equipped to design systems that exploit both language and structure.

## 23.2 Text-Attributed Graphs

A **text-attributed graph** (TAG) is a graph where nodes, edges, or both carry natural language descriptions in addition to (or instead of) fixed numerical features. Formally, given a graph \( G = (V, E) \), a text-attributed graph augments each node \( v \in V \) with a text string \( t_v \) and optionally each edge \( (u, v) \in E \) with a text string \( t_{uv} \). The strings \( t_v \) can be anything: a paper abstract, a product description, a user's biographical summary, a protein's functional annotation, or a location's Wikipedia entry.

Text-attributed graphs are not a niche academic construct — they describe the majority of real-world graph datasets that matter in practice:

- **Academic citation networks** (ogbn-arxiv, ogbn-papers100M): papers as nodes, abstracts as text, citations as edges
- **E-commerce product graphs** (Amazon, eBay): products as nodes, descriptions and reviews as text, co-purchase or co-view as edges
- **Social networks** (Twitter, Reddit): users and posts as nodes, profile text and post content as text, follows and replies as edges
- **Knowledge graphs** (Wikidata, Freebase): entities as nodes with descriptions, relations as typed edges
- **Molecular graphs** with IUPAC names, functional annotations, and literature references on atoms and bonds

The defining challenge of TAGs is the **feature space mismatch**. Classical GNNs assume node features \( \mathbf{x}_v \in \mathbb{R}^d \) — they operate on fixed-width numerical vectors. Text strings have variable length, contain rich semantic information, and require tokenization and contextualization before they can be compared or aggregated. The core technical question is: what is the right way to turn \( t_v \) into something a GNN can work with?

!!! mascot-thinking "Two Sources of Signal"
    <img src="../../img/mascot/thinking.png" class="mascot-admonition-img" alt="Sage thinks carefully">
    Think of text and structure as two different views of the same underlying reality. The graph structure tells you **who interacts with whom** — which papers cite which, which products are bought together, which users follow which accounts. The text tells you **what each node is about** — the content, topic, and semantics. A good representation captures both. The research question is whether to process them separately and combine, or to train them together from the start.

### 23.2.1 Bag-of-Words as a Baseline

The simplest approach to converting text to node features is a **bag-of-words (BoW)** representation. Each node's text is tokenized into words, and the feature vector \( \mathbf{x}_v \in \mathbb{R}^{|\mathcal{V}|} \) counts (or TF-IDF-weights) occurrences of each vocabulary word. BoW is fast, interpretable, and requires no additional model. However, it discards word order entirely, produces very high-dimensional sparse vectors, and captures no semantic similarity: the words "car" and "automobile" appear as completely unrelated dimensions despite being near-synonyms.

TF-IDF (term frequency–inverse document frequency) reweights raw counts:

\[
\text{TF-IDF}(w, v) = \underbrace{\frac{\text{count}(w, v)}{|\text{words in } t_v|}}_{\text{term frequency}} \times \underbrace{\log \frac{|V|}{\text{nodes containing } w}}_{\text{inverse document frequency}}
\]

where \( w \) is a word and \( v \) is a node. High TF-IDF indicates a word that is frequent in this node's text but rare across the graph — a good discriminative signal. Despite its limitations, TF-IDF BoW was the standard node feature for ogbn-arxiv for several years and still provides a strong baseline.

## 23.3 LLM + GNN Integration

Large language models (LLMs) such as BERT, RoBERTa, and GPT produce **contextual embeddings**: vector representations of text that capture semantic meaning, word order, and long-range context. For a text string \( t_v \), a language model encoder produces a dense vector \( \mathbf{e}_v \in \mathbb{R}^d \) where \( d \) is typically 384 to 4096 depending on the model. These vectors exist in a shared semantic space where similar texts produce similar vectors — unlike BoW, "car" and "automobile" would map to nearby points.

The question is how to incorporate \( \mathbf{e}_v \) into a GNN pipeline. Three distinct strategies have emerged, each with different tradeoffs between expressiveness and computational cost.

### 23.3.1 Strategy 1: LLM as Encoder (Feature Extractor)

The simplest integration strategy treats the LLM as a **frozen feature extractor**. You run the LLM over each node's text string once, offline, and store the resulting embedding as the node feature \( \mathbf{x}_v = \mathbf{e}_v \). The GNN then trains on these fixed features in the usual way. No gradients flow back into the LLM during GNN training.

This approach is computationally efficient: the expensive LLM inference happens once per node, and all subsequent GNN training and inference uses the pre-computed embeddings. It separates the two models cleanly, which simplifies debugging and allows using very large LLMs (too large to fit alongside a GNN in GPU memory during training).

Before looking at the code below, note three things: `SentenceTransformer` wraps a pre-trained BERT-family model and pools its final-layer output to produce one embedding per string; `encode` runs inference in batches; the resulting tensor is immediately usable as node features in a PyG data object.

```python
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Step 1: extract text embeddings with a frozen LLM
encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings
node_texts = [paper['title'] + '. ' + paper['abstract'] for paper in dataset]
with torch.no_grad():
    x = torch.tensor(encoder.encode(node_texts, batch_size=256,
                                     show_progress_bar=True))  # shape: [N, 384]

# Step 2: build a GNN that operates on the frozen LLM features
class LLM_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

model = LLM_GCN(in_channels=384, hidden=256, num_classes=40)
# Only model parameters are trained; the LLM is never updated
```

The weakness of this approach is that the LLM's representations are optimized for general language tasks, not for the specific graph classification problem. A paper in a citation network might embed similarly to a paper in a different field if their abstracts use similar vocabulary, even when their structural neighborhoods are completely different. The LLM cannot adjust to account for structural signals.

### 23.3.2 Strategy 2: TAPE — Pseudo-Labels and Explanations

**TAPE** (Text-Attributed graph with Pretrained Embeddings) goes one step further: rather than just using LLM embeddings as features, it uses the LLM to **generate soft pseudo-labels and explanations** for unlabeled nodes, which then serve as additional supervision for the GNN. This exploits the LLM's zero-shot classification ability to inject prior knowledge into semi-supervised training.

The TAPE pipeline has three stages. First, the LLM classifies each unlabeled node's text in isolation using a natural language prompt that includes the label set. Second, the LLM also generates a short explanation of its classification reasoning. Both the class probabilities and the explanation text are encoded and concatenated to the LLM embedding. Third, the GNN trains on labeled nodes using true labels and on unlabeled nodes using the LLM's pseudo-labels with lower weight.

```python
from openai import OpenAI  # or any LLM API

client = OpenAI()

def tape_prompt(abstract, categories):
    return f"""Given this paper abstract:
"{abstract}"

The paper belongs to one of these arxiv categories:
{', '.join(categories)}

Which category is most likely? Respond with just the category name."""

# Generate pseudo-labels for unlabeled nodes
for node in unlabeled_nodes:
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user',
                   'content': tape_prompt(node['abstract'], category_names)}]
    )
    node['pseudo_label'] = response.choices[0].message.content

# Combine true labels and pseudo-labels for GNN training
# (with lower confidence weight on pseudo-labels)
```

TAPE achieves meaningful gains over plain LLM-as-encoder by anchoring the GNN training in the LLM's world knowledge. The explanation text also provides richer signal about the LLM's reasoning, not just its final classification.

### 23.3.3 Strategy 3: LLM as Graph Reasoner

A fundamentally different approach is to ask the LLM to **reason about graph structure directly**, without training a separate GNN at all. The idea is to convert the local neighborhood of a query node into a natural language description and prompt the LLM to make a prediction from that description alone.

For example, for node classification on a citation network, a prompt might look like:

```
Node to classify: "Attention is All You Need" (Vaswani et al., 2017)
Abstract: "We propose a new simple network architecture, the Transformer..."

This paper is cited by:
- "BERT: Pre-training of Deep Bidirectional Transformers" (NLP)
- "GPT-3: Language Models are Few-Shot Learners" (NLP)
- "Graph Attention Networks" (GNN / Graphs)

Based on the paper content and its citation relationships,
predict its arxiv category: cs.AI, cs.CL, cs.LG, cs.CV, or other?
```

The LLM's prediction integrates structural context (who cites this paper) with semantic content (the abstract). This approach requires no training and is immediately applicable to new tasks and datasets. However, it has clear limitations: LLMs struggle with structural reasoning that extends beyond a few hops, the context window constrains how many neighbors can be included, and LLMs may hallucinate structural relationships they were not shown.

!!! mascot-tip "When to Use Which Strategy"
    <img src="../../img/mascot/tip.png" class="mascot-admonition-img" alt="Sage gives a practical tip">
    As a practical guideline: use **LLM-as-encoder** when you want fast, cheap feature extraction and the graph structure carries most of the predictive signal. Use **LLM-as-reasoner** when you want zero-shot generalization without any training and the task can be expressed in a few sentences. Use **joint training** (Section 23.4) when you have the compute budget and want the best possible accuracy on a fixed task. The compute cost roughly scales as: encoder < reasoner ≈ joint fine-tuning << training from scratch.

The following table summarizes the three strategies across key dimensions. All three strategies have been described in prose above, so this table reinforces and organizes that information rather than introducing new content.

| Strategy | LLM Gradients | GNN | Zero-Shot | Compute | Typical Accuracy Gain |
|---|---|---|---|---|---|
| LLM-as-Encoder (frozen) | None | Trained | No | Low | +2–4% over BoW |
| TAPE (pseudo-labels) | None | Trained | Partial | Medium | +4–6% over BoW |
| LLM-as-Reasoner | None | None | Yes | Medium | +0–3% (task-dependent) |
| Joint LLM-GNN | Yes (fine-tuned) | Trained | No | High | +6–10% over BoW |

## 23.4 Graph Instruction Tuning

The strategies above use a **pre-trained, frozen LLM** as a component in a larger pipeline. **Graph instruction tuning** takes the opposite approach: it fine-tunes the LLM itself on graph-related tasks, teaching the model to understand and reason about graph structure as part of its language capabilities.

Instruction tuning is a fine-tuning technique developed in the LLM literature where the model is trained on a collection of (instruction, response) pairs that cover a wide variety of tasks. The model learns to follow natural language instructions, enabling it to generalize to new tasks described at inference time. Graph instruction tuning extends this to graph-structured inputs by creating instruction datasets that include graph descriptions alongside their answers.

An instruction example for node classification might look like this:

```
INSTRUCTION:
You are a graph learning model. A citation graph has the following structure.

Node 1047 has abstract: "We present a novel transformer architecture for molecular property prediction..."
Node 1047's neighbors:
  - Node 83: "Graph neural networks for chemical property prediction" (category: cs.LG)
  - Node 291: "SMILES-based molecular generation using transformers" (category: cs.LG)
  - Node 504: "Attention mechanisms in biochemical sequence models" (category: q-bio.QM)

Predict the arxiv category for Node 1047.

RESPONSE:
cs.LG — The paper applies transformer architecture to molecular graphs, 
a topic in graph machine learning. The majority of its neighbors are in cs.LG,
reinforcing this classification.
```

Two influential systems developed graph instruction tuning frameworks:

**InstructGLM** frames node classification, link prediction, and subgraph description as instruction-following tasks. It creates a diverse instruction dataset by templating graph tasks in multiple natural language formats, then fine-tunes a general-purpose LLM on this dataset. The resulting model handles each task when prompted with the appropriate instruction.

**GraphGPT** uses a two-stage training procedure. In the first stage, it aligns a graph encoder (a standard GNN) with the LLM decoder through a projection layer, using graph-text pairs (papers and their abstracts, knowledge graph entities and their descriptions). In the second stage, it fine-tunes the aligned system on graph instruction-following tasks. The graph encoder feeds structural information into the LLM's context, and the LLM generates natural language outputs.

!!! mascot-encourage "A Field in Motion"
    <img src="../../img/mascot/encouraging.png" class="mascot-admonition-img" alt="Sage encourages students working on hard material">
    Graph instruction tuning is one of the fastest-moving areas in the field. The specific systems described here (InstructGLM, GraphGPT) represent a snapshot of 2023–2024 research. The core insight — that language models can be taught to reason about graph structure through instruction datasets — is likely to persist even as specific architectures evolve. Focus on understanding *why* each design choice was made rather than memorizing model names.

The practical challenge with graph instruction tuning is dataset construction. For a GNN trained on fixed node features, you need a dataset of (graph, label) pairs. For instruction tuning, you need a dataset of (instruction, graph description, response) triples — a much richer and harder-to-collect format. This limits graph instruction tuning to domains where such datasets can be constructed at scale, typically academic citation networks, knowledge bases, or synthetic graphs.

## 23.5 Graph Foundation Models

The phrase "foundation model" entered the AI lexicon around 2021 to describe large models trained on broad data that can be adapted to diverse downstream tasks. GPT-3, BERT, and CLIP are foundation models in the language and vision domains. The natural question for the graph community is: can we build a **graph foundation model** — a single pre-trained model that transfers to any graph task, the way GPT transfers to any text task?

This ambition runs into three fundamental obstacles that do not exist in the language domain:

**Feature space heterogeneity.** A text foundation model processes the same token vocabulary regardless of the task. A graph foundation model must handle node features from completely different domains: 128-dimensional BoW vectors in a citation network, 2048-dimensional image features in a visual scene graph, and scalar physical measurements in a molecular graph. There is no shared "vocabulary" across graph domains.

**Task heterogeneity.** Language foundation models are usually evaluated on classification or generation — structurally similar tasks. Graph tasks range from node classification to link prediction to graph isomorphism — each requires a different output head, loss function, and interpretation of the output.

**Structural diversity.** Different graphs have different numbers of nodes, different degree distributions, and different global topology. A model trained on social networks (scale-free, high clustering) may not transfer to molecular graphs (bounded degree, planar).

Two distinct approaches have emerged to address these obstacles.

### 23.5.1 PRODIGY: In-Context Learning via Prompt Graphs

**PRODIGY** (2023) takes inspiration from in-context learning in language models. In the language setting, you can adapt GPT to a new classification task by providing labeled examples directly in the prompt ("Few-shot prompting"). PRODIGY asks: can we do the same for graphs?

The answer is yes, through the concept of a **prompt graph**. A prompt graph \( G_{\text{prompt}} \) is a small augmented graph that contains:

- **Labeled example nodes** with known labels — these serve as the "few-shot examples"
- **Query nodes** whose labels the model must predict
- **Support edges** connecting example nodes to query nodes through the task graph

The pre-trained GNN processes the prompt graph, and the model's prediction for a query node is guided by the labeled examples in the prompt graph via the message-passing mechanism. No fine-tuning is required: changing the few-shot examples in the prompt graph is sufficient to adapt the model to a new task.

PRODIGY trains the backbone GNN on a large collection of graphs and tasks, teaching it to use prompt context effectively. At inference time on a new graph or task, you simply construct a prompt graph with a handful of labeled examples and run the trained model — no gradient updates required.

\[
p(y_q \mid G, G_{\text{prompt}}) = \text{GNN}(G \cup G_{\text{prompt}})[q]
\]

where \( q \) is a query node, \( G \) is the task graph, and \( G_{\text{prompt}} \) is the augmented prompt graph containing labeled support examples. The union \( G \cup G_{\text{prompt}} \) connects query nodes to support nodes through edges added by the prompt construction procedure.

### 23.5.2 OFA: One for All via Shared Text Space

**OFA** (One for All, 2023) addresses feature space heterogeneity by projecting all graph modalities into a **shared natural language space** using an LLM. Every node, regardless of its original feature representation, is described in natural language:

- An arxiv paper node becomes: *"A paper titled 'Attention is All You Need' in the NLP field, proposing the Transformer architecture for sequence transduction."*
- A product node becomes: *"A laptop with Intel Core i7 processor, 16GB RAM, and 512GB SSD storage, priced at $1,299."*
- A molecule node becomes: *"A carbon atom with two single bonds and one double bond, forming part of a benzene ring."*

By passing these descriptions through a shared LLM encoder, OFA produces embeddings that live in the same high-dimensional semantic space regardless of the original domain. A GNN trained on these unified embeddings can, in principle, transfer across domains — because the features are now comparable. OFA is then fine-tuned jointly across multiple graph datasets and tasks, learning a single parameter set that performs well everywhere.

The key insight is that natural language is the only representation system expressive enough to describe arbitrarily diverse entities. Numbers, types, and labels all differ across domains, but English (or any natural language) can describe anything. By routing all node descriptions through language, OFA creates a common representation channel.

#### Diagram: LLM+GNN Pipeline — Text-to-Prediction


<iframe src="../../sims/ch23-llm-gnn-pipeline/main.html" width="100%" height="502px" scrolling="no"></iframe>
[Run LLM+GNN Pipeline — Text-to-Prediction Fullscreen](../../sims/ch23-llm-gnn-pipeline/main.html)

## 23.6 The ogbn-arxiv Benchmark

To make the strategies concrete, the following table presents representative results on ogbn-arxiv — the 170K-node citation network that has served as the running benchmark throughout this textbook. The task is multi-class node classification into 40 arxiv subject categories. Test accuracy (%) is reported; higher is better.

Before reading the table, recall that all four approaches described above have been explained in prose: BoW+GCN is the baseline without any LLM; LLM-encoder uses frozen BERT embeddings; TAPE adds LLM pseudo-labels; and GraphGPT/OFA jointly train the LLM and GNN.

| Method | LLM | GNN | Test Acc (%) | Year |
|---|---|---|---|---|
| GCN + BoW (baseline) | None | GCN | 71.7 | 2020 |
| GCN + BERT (frozen) | BERT-base | GCN | 73.5 | 2021 |
| TAPE | GPT-3.5 | MLP | 76.5 | 2023 |
| InstructGLM | LLaMA-7B | GNN | 76.9 | 2023 |
| GraphGPT | Vicuna-7B | GNN | 77.2 | 2023 |
| OFA (One-for-All) | LLaMA-7B | GNN | 78.0 | 2023 |

The table reveals a clear pattern: each additional integration between LLM and GNN yields additional accuracy, but with rapidly increasing computational cost. The BoW baseline requires no LLM at all. Frozen BERT features add a one-time offline inference cost. TAPE requires LLM API calls (expensive per-query). Joint training methods (GraphGPT, OFA) require fine-tuning 7B+ parameter models — accessible only with substantial GPU resources.

!!! mascot-warning "API Cost and Latency in Production"
    <img src="../../img/mascot/warning.png" class="mascot-admonition-img" alt="Sage warns about a common pitfall">
    The benchmark table shows accuracy gains from LLM integration — but it does not show inference cost. In production systems, calling a large LLM API for every node at query time is prohibitively expensive and slow. TAPE-style pseudo-labels are generated **offline once** (not at query time), which is workable. LLM-as-reasoner strategies that query the LLM per inference are typically only feasible for low-traffic or high-value decisions. Always benchmark **latency and cost** alongside accuracy when selecting an LLM+GNN strategy.

## 23.7 Common Pitfalls

**Confusing offline and online LLM usage.** LLM-as-encoder computes embeddings once and stores them — this is cheap per-query after the initial computation. LLM-as-reasoner calls the LLM at inference time for every new query — this is expensive and has latency. These are fundamentally different deployment profiles. A system that performs well in a notebook (where you run inference offline) may be completely impractical in production (where you need real-time responses).

**Forgetting that frozen LLM embeddings cannot adapt to graph signals.** When the LLM is frozen, the node embeddings are computed from text alone with no knowledge of the graph structure. Two nodes in completely different neighborhoods may have very similar LLM embeddings if their text is similar. The GNN is then responsible for differentiating them using structural context — but the features it starts with may not encode the right signals for the task. Fine-tuning at least the final few layers of the LLM (even lightly) on the downstream task often improves over fully frozen embeddings.

**Context length as a hard ceiling for LLM-as-reasoner.** When you describe a node's neighborhood in text, the number of tokens grows linearly with the number of neighbors included. For high-degree nodes in a citation network (papers with thousands of citations), you cannot fit even a fraction of the neighborhood into a single context window. Naive truncation biases the prediction toward whichever neighbors happen to appear first. Structured neighbor sampling (analogous to mini-batching in GNNs) is required.

**Feature leakage in graph instruction tuning datasets.** If the instruction tuning dataset includes validation or test graph nodes as labeled examples in the "few-shot prompt" for other nodes, you inadvertently leak test labels into training. This is especially easy to do accidentally when constructing prompt graphs for PRODIGY-style models. Strict dataset splits must be maintained at the prompt-graph level, not just the node level.

**Treating OFA-style models as plug-and-play.** Foundation model papers report impressive zero-shot or few-shot results, but these results typically come from models trained on dozens of carefully curated benchmark graphs. Applying an OFA model to a novel graph domain (say, power grid topology or protein-protein interactions with domain-specific node descriptions) requires careful adaptation of the node description templates. The model's zero-shot ability depends on its training domain coverage.

## 23.8 MicroSim: LLM+GNN Pipeline Explorer

#### Diagram: LLM+GNN Pipeline Explorer (Full Version)


<iframe src="../../sims/ch23-llm-gnn-explorer/main.html" width="100%" height="562px" scrolling="no"></iframe>
[Run LLM+GNN Pipeline Explorer (Full Version) Fullscreen](../../sims/ch23-llm-gnn-explorer/main.html)

## 23.9 Exercises

The following twelve exercises span all six levels of Bloom's taxonomy, from recall to synthesis.

**Remembering**

1. Define "text-attributed graph." Give two real-world examples of graphs where nodes have associated text strings, other than the academic citation network used in this chapter.

2. What are the three LLM+GNN integration strategies described in Section 23.3? For each, state whether the LLM's parameters are updated during GNN training.

**Understanding**

3. Explain the difference between TF-IDF bag-of-words features and LLM-generated embeddings as node features. In what sense do LLM embeddings capture "semantic similarity" that BoW does not?

4. Describe the TAPE pipeline in your own words. Why does TAPE generate *explanations* in addition to pseudo-labels? What information do the explanations add that the pseudo-label probabilities alone do not capture?

5. Why does the LLM-as-reasoner strategy face a "context length ceiling"? Give a specific example of how this limitation would manifest in a high-degree node of a citation network.

**Applying**

6. You are building a product recommendation system where each product node has a 200-word description and the graph represents co-purchase relationships. You have a budget for one offline LLM inference pass (no per-query LLM calls). Which LLM+GNN integration strategy would you choose? Write pseudocode for the key steps.

7. Using the code template in Section 23.3.1, modify the `LLM_GCN` class to use **three** GCN layers instead of two, and add a skip connection that adds the original LLM features \( \mathbf{x}_v \) to the output of the second layer before the third layer.

**Analyzing**

8. Table 23.1 shows that OFA achieves 78.0% test accuracy on ogbn-arxiv compared to 71.7% for BoW+GCN. However, OFA requires fine-tuning a 7B-parameter LLM. Analyze the tradeoffs: under what circumstances would the 6.3% accuracy gain justify the cost? Under what circumstances would you prefer the cheaper baseline?

9. Compare PRODIGY and OFA as approaches to graph foundation models. What is the core challenge each one addresses? Could their approaches be combined? What would a combined system look like?

**Evaluating**

10. A colleague proposes using an LLM-as-reasoner approach for a fraud detection graph where each transaction node has a text description of the transaction details. They argue this approach requires no training and will generalize immediately to new fraud patterns. Evaluate this proposal. What are its strengths? What are at least three specific failure modes the colleague has not addressed?

11. Graph instruction tuning datasets must be constructed from task graphs. Propose a strategy for constructing a graph instruction tuning dataset for a social network recommendation task. What instruction formats would you use? How would you ensure that the dataset tests generalization rather than memorization? What are the hardest data quality issues to avoid?

**Creating**

12. Design a new LLM+GNN integration strategy that is not covered in this chapter. Specifically: (a) describe at what stage and in what direction information flows between the LLM and GNN, (b) describe what signals are shared versus kept separate, (c) identify one concrete graph task where your strategy would outperform all three strategies in Section 23.3, and explain why, (d) identify one concrete task where your strategy would underperform, and explain why.

## 23.10 Further Reading

**PRODIGY: Enabling In-context Learning Over Graphs** — Huang et al. (2023). *arXiv:2305.12600.* The paper that introduced prompt graphs as a mechanism for graph in-context learning. Highly readable; Section 3 on prompt graph construction is the key technical contribution. Relevant to Section 23.5.1.

**Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning (TAPE)** — He et al. (2023). *arXiv:2305.19523.* The TAPE paper. Section 3 describes the pseudo-label generation and explanation encoding pipeline in detail. The ablation study (Table 4) is especially useful for understanding which component contributes most. Relevant to Section 23.3.2.

**GraphGPT: Graph Instruction Tuning for Large Language Models** — Tang et al. (2023). *arXiv:2310.13023.* Describes the two-stage alignment and instruction tuning approach. The graph token projection mechanism (Section 3.2) is the core technical innovation. Relevant to Section 23.4.

**One for All: Towards Training One Graph Model for All Classification Tasks** — Liu et al. (2023). *arXiv:2310.00149.* The OFA paper. The unified node description language in Section 3.1 is the key idea. The cross-dataset generalization experiments (Table 3) are the main empirical result. Relevant to Section 23.5.2.

**Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data** (2025). *arXiv:2510.06377.* A 2025 extension of the foundation model idea to full relational databases, not just graph-structured data. Previews where the field is heading beyond text-attributed graphs.

**Large Language Models on Graphs: A Comprehensive Survey** — Chen et al. (2024). *arXiv:2312.02783.* A comprehensive survey of LLM+GNN integration methods organized by application domain. Useful for finding relevant prior work when applying these methods to a new domain.

**Text-Attributed Graph Representation Learning: Methods, Applications, and Challenges** — Jin et al. (2023). *arXiv:2308.08483.* A survey organized around the three integration strategies (encoder, predictor, alignment). Table 2 provides a comprehensive comparison of 20+ methods across benchmarks.

!!! mascot-celebration "Chapter 23 Complete!"
    <img src="../../img/mascot/celebration.png" class="mascot-admonition-img" alt="Sage celebrates chapter completion">
    You have navigated one of the most rapidly evolving intersections in machine learning — the boundary between language and graph representation. You understand why text-attributed graphs demand a different approach than numerical-feature graphs, the three ways to integrate a language model with a GNN and their tradeoffs, how graph instruction tuning teaches language models to reason about structure, and how PRODIGY and OFA are attempting to build the first true graph foundation models. The next chapter turns to another frontier: advanced self-supervised learning paradigms and uncertainty quantification for GNNs.

[See Annotated References](./references.md)
