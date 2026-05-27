# CS224W Materials — GNN Textbook Source

This folder contains all organized source materials for generating an intelligent textbook on **Graph Neural Networks and Machine Learning with Graphs**, going beyond Stanford CS224W in depth, coverage, and interactivity.

Built to feed into Dan McCreary's Claude Skills intelligent textbook workflow:
- Repo: https://github.com/dmccreary/claude-skills
- Keep skills up to date: `cd ~/Documents/Projects/claude-skills && git pull`
- Monitor token usage: https://claude.ai/settings/usage
- Preview textbook locally (once built): http://127.0.0.1:8000

---

## Folder Structure

| Folder / File | Contents |
|---|---|
| [`course-description.md`](./course-description.md) | Full course description with Bloom's Taxonomy — **start here** |
| [`concept-list.md`](./concept-list.md) | ~282 GNN concepts for the learning graph generator |
| [`textbook-design-principles.md`](./textbook-design-principles.md) | Why & how this textbook exceeds CS224W — read before generating content |
| [`syllabus/`](./syllabus/) | 2025 and 2021 course schedules with slide links |
| [`lectures/`](./lectures/) | One markdown file per lecture (19 topics) with key concepts, papers, code |
| [`colabs/`](./colabs/) | 6 Colab notebook assignments with direct links |
| [`readings/`](./readings/) | ~60 key papers organized by topic |
| [`youtube/`](./youtube/) | YouTube playlist breakdown (60 videos, 2021 offering) |
| [`reference-books/`](./reference-books/) | Three free reference textbooks with reading guidance |

---

## Textbook Generation Workflow

> ⚠️ **Always pull latest skills first:**
> ```bash
> cd ~/Documents/Projects/claude-skills
> git pull
> ```
> Then check available skills: `./scripts/list-skills.sh`

> 💡 **Read each Claude response before running the next step.** Don't blindly chain prompts — review what was generated and verify quality before proceeding.

---

### Step 0 — Pull Latest Skills and Set Up Project

```bash
# 1. Pull latest claude-skills
cd ~/Documents/Projects/claude-skills
git pull

# 2. Create/navigate to your textbook project directory
cd ~/Documents/Projects/Textbooks/Graph-Neural-Networks-Textbook

# 3. Install skills for this project (creates symlinks to ~/.claude/skills/)
cd ~/Documents/Projects/claude-skills/scripts
./install-claude-skills.sh
```

Then in Claude Code, verify skills are loaded:
```
Hi Claude! What skills do you know about?
```
You should see ~19 skills listed.

---

### Step 1 — Analyze and Refine Course Description

**Input file:** `Materials/course-description.md`

```
Use the course-description-analyzer skill on @Materials/course-description.md
```

**Goal:** Score above **85/100** before proceeding. The analyzer checks:
- Bloom's taxonomy coverage (all 6 levels)
- Clarity of target audience and prerequisites
- Specificity of learning objectives
- Quality of topic enumeration

Iterate on `course-description.md` until score ≥ 85. The file already has a strong foundation — you may need only minor refinements.

---

### Step 2 — Generate Learning Graph

**Input:** The analyzed course description

```
Run the learning-graph-generator skill using @Materials/course-description.md and @Materials/concept-list.md as source material
```

**What gets generated in `docs/learning-graph/`:**
- `concept-list.md` — enumerated concepts (~200, canonical textbook version)
- `learning-graph.csv` — concept dependencies in DAG format
- `quality-metrics.md` — validation report (target quality score ≥ 70/100)
- `learning-graph.json` — vis-network visualization format
- `taxonomy-distribution.md` — distribution across 12+ taxonomy categories

**After generation, run the Python validation scripts:**
```bash
cd docs/learning-graph
python analyze-graph.py learning-graph.csv quality-metrics.md
python csv-to-json.py learning-graph.csv learning-graph.json
python taxonomy-distribution.py learning-graph.csv taxonomy-distribution.md
```

**Review:** Open `quality-metrics.md` and check for:
- Zero circular dependencies (must be a DAG)
- No orphan nodes
- Average 2–4 dependencies per concept
- No single taxonomy category > 30% of concepts

---

### Step 3 — Install Learning Graph Viewer

```
Run the book-installer skill to install the learning graph viewer
```

This installs the interactive vis-network graph explorer into `docs/sims/graph-viewer/`. After running, serve the site and navigate to the Learning Graph section to verify the visualization looks correct.

> **Note:** You may need to adjust the legend colors and taxonomy ordering after installation to match the GNN domain's taxonomy categories.

---

### Step 4 — Generate Book Chapter Structure

```
Run the book-chapter-generator skill
```

**What gets generated:** A `docs/chapters/` directory with one subdirectory per chapter. Each chapter gets an `index.md` with:
- Chapter overview
- List of concepts to cover (from the learning graph)
- Concept dependencies that must have been introduced by earlier chapters

**Review the output carefully:**
- Does the chapter ordering respect concept dependencies? (GNN foundations before GNN theory, etc.)
- Are the 26 chapters balanced in scope? (look at the shell output for concept counts per chapter)
- Do the early chapters cover prerequisites before jumping into GNNs?

You can ask Claude to adjust: *"Move temporal graphs to its own chapter between recommender systems and generative models"*

---

### Step 5 — Generate Chapter Content

Run the content generator chapter by chapter. Do not batch all chapters at once — review each one.

```
Run the chapter-content-generator skill on chapter 1 @docs/chapters/01-*/index.md
```

Repeat for each chapter:
```
Run the chapter-content-generator skill on chapter 2 @docs/chapters/02-*/index.md
```

**For each chapter, verify the generated content includes:**
- [ ] Real-world motivating example at the start
- [ ] Intuitive explanation before equations
- [ ] Full mathematical derivations
- [ ] A MicroSim placeholder/idea
- [ ] Complete runnable PyTorch Geometric code
- [ ] A benchmark results table
- [ ] "Common Pitfalls" section
- [ ] 12 exercises across all Bloom's levels
- [ ] "Further Reading" section with annotated papers

**Reference the lecture files for source material:**
The `Materials/lectures/` folder has detailed notes for each topic. Point Claude at these:
```
Run the chapter-content-generator skill on chapter 6 @docs/chapters/06-*/index.md 
using @Materials/lectures/06-gnn-foundations.md as source material
```

---

### Step 6 — Create MicroSims

For each chapter's key concept, generate an interactive p5.js simulation:

```
Run the microsim-generator skill to create a p5.js simulation of GNN message passing
```

**Priority MicroSims** (one per Part, minimum):
1. Graph property explorer (Chapter 1)
2. WL color refinement step-through (Chapter 2)
3. Biased random walk visualizer with p/q controls (Chapter 3)
4. PageRank power iteration convergence (Chapter 4)
5. GCN message passing — click a node, see messages flow (Chapter 6)
6. GAT attention weights as edge thickness (Chapter 7)
7. TransE geometry in 2D (Chapter 12)
8. GraphRNN graph generation step-by-step (Chapter 21)

After creating each MicroSim, capture a screenshot for the catalog:
```
Capture a screenshot of this MicroSim
```

---

### Step 7 — Generate Supporting Content

Run these after all chapters are complete:

**Glossary (ISO 11179-compliant):**
```
Run the glossary-generator skill
```
Target: ~282 terms (one per concept in the learning graph), each with a precise definition and concrete example.

**FAQ:**
```
Run the faq-generator skill
```
Target: ~70 questions covering common student confusion points across all topics.

**Quizzes:**
```
Run the quiz-generator skill on chapter 6 @docs/chapters/06-*/index.md
```
Repeat per chapter. Target: 40 questions per chapter across all 6 Bloom's levels.

**Reference list:**
```
Run the reference-generator skill
```
This generates a curated `docs/references.md` from the papers in `Materials/readings/papers.md`.

---

### Step 8 — Quality Assurance

**Generate book metrics report:**
```
Run the book-metrics-generator skill
```
This produces word counts, code block counts, concept coverage, MicroSim count, etc.

**Validate MicroSims:**
```
Run the microsim-utils skill to standardize all MicroSims
```

**Diagram/MicroSim report:**
```
Run the diagram-reports-generator skill
```

**Check for:**
- All 282 concepts defined in glossary
- All chapters have at least one MicroSim
- All chapters have quizzes
- No circular concept dependencies
- Reading level appropriate for target audience

---

### Step 9 — Build and Preview

```bash
# In VSCode terminal:
mkdocs serve
```
Go to http://127.0.0.1:8000 and navigate every chapter, check every MicroSim, test every link.

---

### Step 10 — Deploy to GitHub Pages

```bash
# In VSCode terminal:
mkdocs gh-deploy
```

The site will be live at: `https://[your-github-username].github.io/Graph-Neural-Networks-Textbook/`

---

## Quick Reference: Skill Invocations

| Skill | When to use | Invocation |
|---|---|---|
| `course-description-analyzer` | Step 1 | `Use the course-description-analyzer skill on @Materials/course-description.md` |
| `learning-graph-generator` | Step 2 | `Run the learning-graph-generator skill` |
| `book-installer` | Step 3 | `Run the book-installer skill to install the learning graph viewer` |
| `book-chapter-generator` | Step 4 | `Run the book-chapter-generator skill` |
| `chapter-content-generator` | Step 5 | `Run the chapter-content-generator skill on chapter N @docs/chapters/0N-*/index.md` |
| `microsim-generator` | Step 6 | `Run the microsim-generator skill to create a p5.js simulation of [concept]` |
| `glossary-generator` | Step 7 | `Run the glossary-generator skill` |
| `faq-generator` | Step 7 | `Run the faq-generator skill` |
| `quiz-generator` | Step 7 | `Run the quiz-generator skill on chapter N @docs/chapters/0N-*/index.md` |
| `reference-generator` | Step 7 | `Run the reference-generator skill` |
| `book-metrics-generator` | Step 8 | `Run the book-metrics-generator skill` |
| `microsim-utils` | Step 8 | `Run the microsim-utils skill to standardize all MicroSims` |
| `diagram-reports-generator` | Step 8 | `Run the diagram-reports-generator skill` |
| `linkedin-announcement-generator` | After launch | `Run the linkedin-announcement-generator skill` |

---

## Learning Graph CSV Format

When manually editing `learning-graph.csv`, follow this format:
```
ConceptID,ConceptLabel,Dependencies,TaxonomyID
1,Graph,, FOUND
2,Node,1,FOUND
3,Edge,1,FOUND
...
108,Graph Neural Network,91|106|107,GNN
```

- **ConceptID:** Integer 1–282
- **ConceptLabel:** Title Case, max 32 characters
- **Dependencies:** Pipe-delimited ConceptIDs (empty for foundational concepts)
- **TaxonomyID:** 3–5 letter abbreviation from your taxonomy list

Taxonomy abbreviations for this textbook:
`PREREQ` | `FOUND` | `ALGO` | `EMB` | `GNN` | `THEORY` | `TRANS` | `KG` | `HETERO` | `APP` | `SCALE` | `GEN` | `ADV` | `TRAIN` | `TOOLS`

---

## Source URLs

| Source | URL |
|---|---|
| CS224W 2025 | https://web.stanford.edu/class/cs224w/ |
| CS224W 2021 (YouTube-aligned) | http://snap.stanford.edu/class/cs224w-2021/ |
| YouTube Playlist (60 videos) | https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn |
| CS224W Assignment Solutions | https://github.com/AndrewSpano/Stanford-CS224W-ML-with-Graphs |
| Course Notes PDF | https://archives.leni.sh/stanford/CS224w.pdf |
| Claude Skills Repo | https://github.com/dmccreary/claude-skills |
| Claude Skills Docs | https://dmccreary.github.io/claude-skills/ |
| Token Usage Monitor | https://claude.ai/settings/usage |
