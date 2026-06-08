---
title: SIR Epidemic Dynamics on Network Structures
description: Interactive p5.js MicroSim for sir epidemic dynamics on network structures.
image: /sims/ch05-sir-epidemic-network/ch05-sir-epidemic-network.png
og:image: /sims/ch05-sir-epidemic-network/ch05-sir-epidemic-network.png
twitter:image: /sims/ch05-sir-epidemic-network/ch05-sir-epidemic-network.png
social:
   cards: false
quality_score: 0
---

# SIR Epidemic Dynamics on Network Structures

<iframe src="main.html" height="557" width="100%" scrolling="no"></iframe>

[Run the SIR Epidemic Dynamics on Network Structures MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

The SIR model divides a population into Susceptible, Infected, and Recovered compartments. In a well-mixed population the trajectory is determined by the reproduction number \(R_0 = \beta/\gamma\). On an explicit network, structure matters enormously: hubs accelerate early spread, while peripheral nodes may never be reached.

This MicroSim runs both models in parallel. On the left is the ODE trajectory; on the right is the network simulation. Sliders control the infection rate \(\beta\) and recovery rate \(\gamma\). You can pre-load different network topologies (random, scale-free, lattice) to compare epidemic outcomes.

**Learning objective (Bloom's Analyze (Level 4)):** Compare SIR epidemic spread on a well-mixed population (classical ODE model) vs. an explicit network, and see how network heterogeneity — especially hubs — dramatically alters epidemic behavior.

## How to Use

1. **Choose a network** — click a topology preset (Erdős–Rényi, Barabási–Albert, grid).
2. **Set \(\beta\) and \(\gamma\)** — sliders on the bottom control transmission and recovery rates.
3. **Seed infection** — click a node to infect it, or use "Infect Hub" to start at the highest-degree node.
4. **Run simulation** — click "Start" to animate. The left panel shows live SIR curves for the network simulation vs. the ODE prediction.
5. **Compare outcomes** — observe when the ODE overestimates or underestimates peak infection.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch05-sir-epidemic-network/main.html"
        height="557"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
20–30 minutes

### Prerequisites
Differential equations (basic ODE intuition). Graph degree and connectivity (Chapter 1). Optionally: reproduction number \(R_0\).

### Activities

1. Set \(\beta = 0.3\), \(\gamma = 0.1\) (\(R_0 = 3\)) on both a random graph and a scale-free graph of the same size. Compare peak infection size and outbreak duration.
2. Infect a hub vs. a leaf node on the same Barabási–Albert graph. Quantify the difference in final recovered fraction.
3. Find the critical threshold \(\beta/\gamma\) below which the epidemic dies out without reaching the giant component.

### Assessment Question
Derive the basic reproduction number \(R_0\) for the SIR ODE model. Explain why the network-based SIR can show an epidemic even when the ODE \(R_0 < 1\).

## References

1. Kermack & McKendrick (1927). A contribution to the mathematical theory of epidemics. Proc. R. Soc.
2. Pastor-Satorras & Vespignani (2001). Epidemic spreading in scale-free networks. PRL.

---
*Part of Chapter 5: Label Propagation and Semi-Supervised Learning. Return to the [chapter page](../../chapters/05-label-propagation/index.md) or browse all [MicroSims](../index.md).*
