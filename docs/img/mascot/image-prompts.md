# AI Image Prompts: Sage the Graph Node

Self-contained prompts for generating each pose. Every prompt includes the full base character description so it can be used independently.

## Base Description (embedded in every prompt)

> A flat vector illustration of Sage, a glowing indigo circular graph node — the pedagogical mascot for a Graph Neural Networks textbook. Sage is a simple glowing indigo circle (#3f51b5) with a soft luminous aura and 3–5 thin edges radiating outward toward smaller, dimmer neighbor nodes. Sage has a simple geometric face embedded in the circle: two small white dot eyes and a small curved white smile. The art style is flat vector, modern geometric, clean lines, transparent background. No text in image.

---

## Pose 1: Neutral (General Purpose)

```
A flat vector illustration of Sage, a glowing indigo circular graph node — the pedagogical mascot for a Graph Neural Networks textbook. Sage is a simple glowing indigo circle (#3f51b5) with a soft luminous aura and 3–5 thin edges radiating outward toward smaller, dimmer neighbor nodes. Sage has a simple geometric face: two small white dot eyes and a small curved white smile. The art style is flat vector, modern geometric, clean lines, transparent background. No text in image.

Neutral pose: Sage faces the viewer directly in a calm, relaxed state. The radiating edges extend evenly in all directions with a gentle glow. The expression is friendly and open — a slight inquisitive tilt suggested by the face geometry. Filename: neutral.png. Fully transparent background — no white, black, or checkered fill.
```

---

## Pose 2: Welcome (Chapter Openings)

```
A flat vector illustration of Sage, a glowing indigo circular graph node — the pedagogical mascot for a Graph Neural Networks textbook. Sage is a simple glowing indigo circle (#3f51b5) with a soft luminous aura and 3–5 thin edges radiating outward toward smaller, dimmer neighbor nodes. Sage has a simple geometric face: two small white dot eyes and a small curved white smile. The art style is flat vector, modern geometric, clean lines, transparent background. No text in image.

Welcome pose: Sage's edges fan outward in a warm, expansive arc as if opening arms in greeting. Two edges on the upper sides curve slightly upward like raised hands. The glow intensifies warmly. Expression is wide-eyed and inviting. A small orange spark at each edge tip. Filename: welcome.png. Fully transparent background.
```

---

## Pose 3: Thinking (Key Concepts)

```
A flat vector illustration of Sage, a glowing indigo circular graph node — the pedagogical mascot for a Graph Neural Networks textbook. Sage is a simple glowing indigo circle (#3f51b5) with a soft luminous aura and 3–5 thin edges radiating outward toward smaller, dimmer neighbor nodes. Sage has a simple geometric face: two small white dot eyes and a small curved white smile. The art style is flat vector, modern geometric, clean lines, transparent background. No text in image.

Thinking pose: A small lightbulb icon or thought bubble appears above Sage. The edges are drawn inward slightly, as if Sage is gathering information from neighbors. One edge has a small animated arrow pointing inward. Expression looks slightly upward and thoughtful. Filename: thinking.png. Fully transparent background.
```

---

## Pose 4: Tip (Helpful Hints)

```
A flat vector illustration of Sage, a glowing indigo circular graph node — the pedagogical mascot for a Graph Neural Networks textbook. Sage is a simple glowing indigo circle (#3f51b5) with a soft luminous aura and 3–5 thin edges radiating outward toward smaller, dimmer neighbor nodes. Sage has a simple geometric face: two small white dot eyes and a small curved white smile. The art style is flat vector, modern geometric, clean lines, transparent background. No text in image.

Tip pose: One edge points upward and to the right with a small orange star at its tip, like a finger pointing. The other edges remain in natural positions. Expression is knowing and helpful — eyes slightly narrowed in a wink. A small sparkle near the pointing edge. Filename: tip.png. Fully transparent background.
```

---

## Pose 5: Warning (Common Mistakes)

```
A flat vector illustration of Sage, a glowing indigo circular graph node — the pedagogical mascot for a Graph Neural Networks textbook. Sage is a simple glowing indigo circle (#3f51b5) with a soft luminous aura and 3–5 thin edges radiating outward toward smaller, dimmer neighbor nodes. Sage has a simple geometric face: two small white dot eyes and a small curved white smile. The art style is flat vector, modern geometric, clean lines, transparent background. No text in image.

Warning pose: Two edges extend horizontally in a gentle "stop" gesture. One edge has a small red X at its tip. Expression is concerned but caring — slightly furrowed brow. A small orange exclamation mark floats nearby. The glow shifts slightly redder at the edges. Filename: warning.png. Fully transparent background.
```

---

## Pose 6: Encouraging (Difficult Content)

```
A flat vector illustration of Sage, a glowing indigo circular graph node — the pedagogical mascot for a Graph Neural Networks textbook. Sage is a simple glowing indigo circle (#3f51b5) with a soft luminous aura and 3–5 thin edges radiating outward toward smaller, dimmer neighbor nodes. Sage has a simple geometric face: two small white dot eyes and a small curved white smile. The art style is flat vector, modern geometric, clean lines, transparent background. No text in image.

Encouraging pose: One edge extends upward with a small thumbs-up or checkmark at its tip. The other edges radiate outward supportively. Expression is warm and reassuring — a generous smile. A soft blue glow surrounds the node. Filename: encouraging.png. Fully transparent background.
```

---

## Pose 7: Celebration (Chapter Complete)

```
A flat vector illustration of Sage, a glowing indigo circular graph node — the pedagogical mascot for a Graph Neural Networks textbook. Sage is a simple glowing indigo circle (#3f51b5) with a soft luminous aura and 3–5 thin edges radiating outward toward smaller, dimmer neighbor nodes. Sage has a simple geometric face: two small white dot eyes and a small curved white smile. The art style is flat vector, modern geometric, clean lines, transparent background. No text in image.

Celebration pose: All edges fan outward dramatically like a burst of energy. Small graph-node confetti (tiny circles) scatter around Sage. The glow is bright and golden-orange. Expression is joyful — wide smile. The whole character seems to pulse with energy. Filename: celebration.png. Fully transparent background.
```

---

## Generation Notes

- Generate at 512×512 or 1024×1024 pixels
- Export as PNG with fully transparent background
- After generating, run the trim script to remove excess padding:
  ```bash
  python ~/Documents/Projects/claude-skills/src/image-utils/trim-padding-from-image.py docs/img/mascot/FILENAME.png
  ```
- Place final images in `docs/img/mascot/`
