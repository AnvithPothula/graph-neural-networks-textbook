# References: Agents and Graphs

1. [Intelligent agent](https://en.wikipedia.org/wiki/Intelligent_agent) - Wikipedia - Covers the foundational definition of intelligent agents, including their perception-action loop and rationality criteria; provides essential background for understanding how LLM-based agents inherit and extend the classical agent framework.

2. [Directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) - Wikipedia - Explains the mathematical structure underlying tool-use DAGs and computation graphs, including topological ordering and reachability; directly relevant to understanding how agent tool plans are represented and executed without cycles.

3. [Knowledge graph](https://en.wikipedia.org/wiki/Knowledge_graph) - Wikipedia - Surveys knowledge graph definitions, prominent examples (Wikidata, Google Knowledge Graph), and query mechanisms; provides the structural context for LLM-on-KG reasoning and agent memory systems built on relational triple stores.

4. Artificial Intelligence: A Modern Approach - Stuart Russell and Peter Norvig - Pearson - The definitive reference for classical agent architectures, planning algorithms, and search methods; Chapters 2–4 (agents and search) and Chapter 10 (classical planning) establish the formal foundations that modern LLM-based graph agents build upon.

5. Language Models are Few-Shot Learners (GPT-3 Technical Report) - Tom Brown et al. - OpenAI/Advances in Neural Information Processing Systems - Introduces the few-shot prompting paradigm that underlies ReAct and Toolformer; understanding in-context learning is essential for grasping why tool-use agents can generalize to new APIs and graph query patterns without weight updates.

6. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) - arXiv - Yao et al. (2022) introduce the ReAct framework that interleaves chain-of-thought reasoning traces with environment actions, establishing the observe-reason-act loop that structures multi-hop graph traversal agents.

7. [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) - arXiv - Schick et al. (2023) show how a language model can learn when and how to call external tools via self-supervised API annotations; the self-supervised training approach is directly relevant to building agents that query knowledge graph APIs autonomously.

8. [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831) - arXiv - Gutierrez et al. (2024) build a hippocampus-inspired retrieval system using a knowledge graph with GNN-based subgraph retrieval, providing a concrete implementation of the agent-memory-as-graph idea with empirical evaluation on multi-hop QA benchmarks.

9. [Graph-of-Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687) - arXiv - Besta et al. (2023) extend chain-of-thought prompting to a graph structure in which reasoning steps can branch, merge, and loop, enabling compositional multi-hop reasoning that linear chains of thought cannot express.

10. [KGQA: Knowledge Graph Question Answering Survey](https://paperswithcode.com/task/knowledge-graph-question-answering) - Papers With Code - Aggregates benchmarks, leaderboards, and linked papers for knowledge graph question answering; an essential starting point for understanding the empirical landscape of LLM-on-KG reasoning, including multi-hop datasets such as WebQuestionsSP and ComplexWebQuestions.
