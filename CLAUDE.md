# Graph Neural Networks Textbook — Project Instructions

## Learning Mascot: Sage the Graph Node

### Mascot File Index

The canonical files for this mascot. When editing any of these, update the others in the same turn so they stay in sync.

| File | Purpose |
|------|---------|
| [`docs/img/mascot/character-sheet.md`](docs/img/mascot/character-sheet.md) | Canonical identity document (name, visual, voice). Source of truth. |
| [`docs/img/mascot/image-prompts.md`](docs/img/mascot/image-prompts.md) | Self-contained AI prompts for regenerating each pose. |
| [`docs/img/mascot/neutral.png`](docs/img/mascot/neutral.png) | Default / general-purpose pose. |
| [`docs/img/mascot/welcome.png`](docs/img/mascot/welcome.png) | Chapter-opening pose. |
| [`docs/img/mascot/thinking.png`](docs/img/mascot/thinking.png) | Key-concept pose. |
| [`docs/img/mascot/tip.png`](docs/img/mascot/tip.png) | Hint / helpful-guidance pose. |
| [`docs/img/mascot/warning.png`](docs/img/mascot/warning.png) | Common-mistake / pitfall pose. |
| [`docs/img/mascot/encouraging.png`](docs/img/mascot/encouraging.png) | Difficult-content / struggle pose. |
| [`docs/img/mascot/celebration.png`](docs/img/mascot/celebration.png) | End-of-chapter / achievement pose. |
| [`docs/css/mascot.css`](docs/css/mascot.css) | Custom admonition styles for the seven pose contexts. |
| [`docs/learning-graph/mascot-test.md`](docs/learning-graph/mascot-test.md) | Rendering test page that exercises every admonition style. |

### Character Overview

- **Name:** Sage
- **Type:** Abstract geometric — a glowing graph node (indigo circle, radiating edges)
- **Catchphrase:** "Let's aggregate some knowledge!"
- **Named after:** GraphSAGE — the nod is intentional

### Voice

- Clear and precise; uses graph metaphors naturally
- Never uses jargon without defining it first
- Encourages students to think about relationships, not just individual items
- Signature phrases: "Let's aggregate some knowledge!", "Your neighbors have insights too!", "Connect the dots and the pattern emerges."

### Mascot Admonition Format

Always place mascot images in the admonition body, never in the title bar:

```markdown
!!! mascot-welcome "Title Here"
    <img src="../../img/mascot/welcome.png" class="mascot-admonition-img" alt="Sage waving welcome">
    Admonition text goes here after the img tag.
```

Image path depth: chapters at `chapters/01-intro/index.md` → use `../../img/mascot/`. Learning graph pages → use `../../img/mascot/`.

### Placement Rules

| Context | Admonition Type | Frequency |
|---------|----------------|-----------|
| General note / sidebar | mascot-neutral | As needed |
| Chapter opening | mascot-welcome | Every chapter |
| Key concept | mascot-thinking | 2–3 per chapter |
| Helpful tip | mascot-tip | As needed |
| Common mistake | mascot-warning | As needed |
| Difficult content | mascot-encourage | Where students may struggle |
| Chapter completion | mascot-celebration | End of each chapter |

**Hard limits:** ≤6 Sage admonitions per chapter. Never back-to-back. Chapter 1 must include Sage's self-introduction naming all six pose-roles.

### Six Pose-Roles (Bloom's Aligned)

1. **Curious (Remember):** Sage looking outward, edges reaching toward neighbors
2. **Explaining (Understand):** Sage with speech bubble describing message flow
3. **Working (Apply):** Sage with a code window, data flowing in via edges
4. **Analyzing (Analyze):** Sage comparing two subgraph configurations
5. **Evaluating (Evaluate):** Sage with one edge marked X and another with a checkmark
6. **Designing (Create):** Sage's edges fanning outward joyfully, node-confetti

## Running Examples

Use these two datasets consistently throughout the textbook:

- **Karate Club graph** (Zachary 1977) — Chapters 1–9: social network, 34 nodes, fully visualizable, always available via `networkx.karate_club_graph()`
- **ogbn-arxiv** (OGB) — Chapters 6–20: academic citation network, 170K nodes, standard OGB benchmark for performance comparisons

## Design Standards (from textbook-design-principles.md)

Every chapter MUST include:
- Real-world motivating example (not a toy problem)
- Intuition section BEFORE equations
- Full mathematical derivation (not just the result)
- At least one MicroSim idea with p5.js implementation notes
- Complete runnable PyTorch Geometric code
- Benchmark results table with 2024–2025 citations
- "Common Pitfalls" subsection
- 12 exercises distributed across all 6 Bloom's levels
- "Further Reading" section with 5–8 annotated papers

## Taxonomy Abbreviations

When generating the learning graph CSV, use these exact abbreviations:
`PREREQ | FOUND | ALGO | EMB | GNN | THEORY | TRANS | KG | HETERO | APP | SCALE | GEN | ADV | TRAIN | TOOLS`

## MkDocs Config Notes

- Site URL: https://AnvithPothula.github.io/graph-neural-networks-textbook/
- Do NOT add `navigation.tabs` — use side navigation only
- Math: arithmatex extension enabled; MathJax JS will be added via book-installer
- After adding each chapter's index.md, update the `nav:` section in mkdocs.yml
