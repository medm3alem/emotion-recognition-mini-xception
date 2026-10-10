# Real-Time Facial Emotion Recognition — Mini-Xception

A lightweight, browser-based facial expression recognition application powered by **Mini-Xception**, **TensorFlow.js**, and **BlazeFace**.

**[Live demo](https://medm3alem.github.io/emotion-recognition-mini-xception/)** · **[Repository](https://github.com/medm3alem/emotion-recognition-mini-xception)**

## Overview

This project implements a real-time facial expression recognition pipeline:

**Webcam → BlazeFace detection → Grayscale 48 × 48 preprocessing → Mini-Xception → 7 expression scores**

Inference runs in the browser without a Python backend. The interface shows the predicted expression, per-class scores, and processing latency measurements.

## Model

| Property | Value |
|---|---|
| Architecture | Mini-Xception (depthwise separable convolutions and residual connections) |
| Training dataset | FER-2013 |
| Input | `float32` tensor `[1, 48, 48, 1]` |
| Preprocessing | Face crop, grayscale, resize to 48 × 48, divide by 255 |
| Output | 7 class probabilities |
| Parameters | Approximately 120K |
| Deployment | TensorFlow.js GraphModel (`model.json` + weight shard) |

**Output class order (important):** `angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`.

Training-time data augmentation was removed from the inference export. A test input produced identical predictions before and after the removal.

## Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 60.37% |
| Macro F1-score | 0.563 |

These metrics describe the FER-2013 classification evaluation, **not** the end-to-end webcam pipeline. Runtime performance varies by device and browser.

## Latency

The web application uses `performance.now()` to measure processing time. Depending on the deployed version, the interface reports face detection time, Mini-Xception preprocessing/inference time, total processing latency, and processing throughput. Processing throughput is distinct from the actual interface update rate when a delay is inserted between iterations.

## Project structure

```text
emotion-recognition-mini-xception/
├── index.html
├── style.css
├── app.js
├── model/
│   ├── model.json
│   └── group1-shard1of1.bin
└── README.md
```

## Run locally

```bash
git clone https://github.com/medm3alem/emotion-recognition-mini-xception.git
cd emotion-recognition-mini-xception
python -m http.server 8000
```

Open **http://localhost:8000**, then allow webcam access. The web app does not require a Python ML environment, but an internet connection is needed for its JavaScript CDN dependencies.

## Deployment

The static site is designed for **GitHub Pages**. In repository **Settings → Pages**, deploy from branch `main`, folder `/ (root)`.

Demo URL: https://medm3alem.github.io/emotion-recognition-mini-xception/

## Technology stack

- **TensorFlow / Keras:** training and export
- **TensorFlow.js:** browser inference
- **BlazeFace:** face detection
- **JavaScript, HTML, CSS:** application and UI
- **GitHub Pages:** hosting

## Limitations and next steps

- Facial expression categories do not reliably reveal a person's internal emotional state.
- Results may degrade with occlusions, head pose, lighting changes, and camera quality.
- The current pipeline focuses on one face at a time.
- Planned work: compare against **LBP + KNN** and **HOG + SVM** using consistent inputs and latency measurement methodology, and benchmark across devices.

## Author

[medm3alem](https://github.com/medm3alem)
