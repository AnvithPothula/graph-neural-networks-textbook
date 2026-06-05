---
title: Traffic Temporal Graph Forecasting
description: Interactive p5.js MicroSim for traffic temporal graph forecasting.
image: /sims/ch22-traffic-temporal/ch22-traffic-temporal.png
og:image: /sims/ch22-traffic-temporal/ch22-traffic-temporal.png
twitter:image: /sims/ch22-traffic-temporal/ch22-traffic-temporal.png
social:
   cards: false
quality_score: 0
---

# Traffic Temporal Graph Forecasting

<iframe src="main.html" height="522" width="100%" scrolling="no"></iframe>

[Run the Traffic Temporal Graph Forecasting MicroSim Fullscreen](./main.html){ .md-button .md-button--primary }
<br/>
[Edit in the p5.js Editor](https://editor.p5js.org/)

## About This MicroSim

Traffic forecasting is a canonical temporal graph task: road sensors form a spatial graph (edges = road adjacency), and each sensor records a time series of speed measurements. Two types of dependency must be captured: spatial (a bottleneck upstream affects downstream sensors) and temporal (rush hour repeats daily).

This MicroSim shows a small road network with sensors. A time slider replays speed data. Click a sensor to see its time series and watch how its neighbors' speeds co-vary. A "spatial only" vs "spatio-temporal" toggle shows why a purely spatial GNN misses the temporal pattern.

**Learning objective (Bloom's Understand (Level 2)):** See how spatial dependencies (neighboring sensors show correlated speed drops) and temporal dependencies (predictable rush-hour patterns) motivate joint spatio-temporal modeling with STGCN and DCRNN.

## How to Use

1. **Scrub time** — drag the time slider to advance or rewind the simulation clock.
2. **Select a sensor** — click any road sensor node to see its historical speed time series.
3. **Toggle spatial / spatio-temporal** — switch between a GCN-only baseline (spatial only) and STGCN (spatio-temporal).
4. **Read prediction** — the forecast for the next 30 minutes is shown for the selected sensor.
5. **Play animation** — click "Play" to animate the speed values over the road network in real time.

## Iframe Embed Code

You can embed this MicroSim in any web page with the following HTML:

```html
<iframe src="https://AnvithPothula.github.io/graph-neural-networks-textbook/sims/ch22-traffic-temporal/main.html"
        height="522"
        width="100%"
        scrolling="no"></iframe>
```

## Lesson Plan

### Grade Level
Undergraduate / Graduate (College Level)

### Duration
10–15 minutes

### Prerequisites
GNN message passing (Chapter 6). Time series basics. Understanding of recurrent models at an introductory level.

### Activities

1. Select a highway sensor. During rush hour, which upstream sensors show speed drops first? Does the downstream sensor lag by how many time steps?
2. Compare the "spatial only" vs. "spatio-temporal" forecast for the same sensor. Which has lower mean absolute error during rush hour?
3. Explain why STGCN uses graph convolution for spatial aggregation and temporal convolution (not recurrence) for time.

### Assessment Question
Define the spatio-temporal graph forecasting task formally. Write the input/output specification for STGCN and explain the spatial graph convolution + temporal gated convolution architecture.

## References

1. Yu et al. (2018). Spatio-Temporal Graph Convolutional Networks. IJCAI.
2. Li et al. (2018). Diffusion Convolutional Recurrent Neural Network. ICLR.

---
*Part of Chapter 22: Temporal and Dynamic Graphs. Return to the [chapter page](../../chapters/22-temporal-graphs/) or browse all [MicroSims](../index.md).*