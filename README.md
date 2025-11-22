# nanoDETR: DEtection TRansformer Implemented From Scratch (PyTorch)

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Vision-Object%20Detection-blue.svg?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Concept-Transformer-red.svg?style=for-the-badge" />
</p>

## 🎬 Video Documentation: A Deep Dive Tutorial

This repository serves as the companion code for a **full, live-coded video tutorial** where the entire DETR architecture is built from first principles.

The video explains the theory behind the Bipartite Matching loss, the mathematics of 2D Positional Encoding, and the construction of the full Encoder-Decoder Transformer from scratch.

➡️ **[WATCH THE FULL VIDEO HERE: https://www.youtube.com/watch?v=YEVVKlyAuQ8]**

➡️ **[ACCESS PAPER HERE: https://arxiv.org/pdf/2005.12872]**

---

## 🌟 Project Motivation

The Detection Transformer (DETR) was a foundational work, pioneering the use of the Transformer architecture for end-to-end object detection. Its key innovation was eliminating the need for complex, manual components like Anchor Boxes and Non-Maximum Suppression (NMS).

The goal of this project, `nanoDETR`, is purely **pedagogical**: to deconstruct and implement every major component of the original DETR paper using basic PyTorch and NumPy, thereby demonstrating a deep, conceptual understanding of the model's structure.

### Key Features Implemented From Scratch:

* **Custom Transformer:** Complete Encoder and Decoder stacks implemented using base PyTorch layers, including both matrix-heavy and abstraction-heavy attention designs.
* **Bipartite Matching Loss:** Implementation of the set-based global loss function using the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`) to enforce unique prediction-to-ground-truth assignment.
* **2D Positional Encodings:** Sinusoidal positional encodings implemented explicitly for injecting spatial information into the flattened image features.
* **Gradient Accumulation:** Training loop configured to support accumulating gradients across batches to simulate larger batch sizes.

---

## 🧠 Implementation Breakdown

The model is split across three main files following a clean modular structure:

### `nanoDETR.py` (The Architecture)

This file contains the core neural network components:

* **Backbone:** Uses a pre-trained `ResNet50` (ImageNet weights) with its final classification layers removed. Features are projected to the Transformer's `hidden_dim` (256) via a 1x1 Convolution.
* **Encoder:** Consists of 6 `EncoderBlock`s, processing the flattened image features and performing self-attention.
* **Decoder:** Consists of 6 `DecoderBlock`s. This is the heart of the prediction, where learnable **Object Queries** attend to the image features (cross-attention) to predict class labels and bounding boxes.

### `main.py` (The Loss & Training Loop)

This is where the magic of end-to-end detection happens.

* **`loss()` function:** This function is built to execute the logic of **Bipartite Matching**:
    1.  Calculates a **Cost Matrix** using the combined classification, L1, and $\text{GIoU}$ matching costs.
    2.  Solves the assignment problem using the Hungarian algorithm (`linear_sum_assignment`).
    3.  Computes the final NLL and bounding box losses only for the optimally matched pairs.
    4.  Uses a reduced weight (0.1) for the "no-object" class in the cross-entropy loss for stability.
* **Data:** Loads the **Pascal VOC 2012** dataset for object detection training.
* **Proof of Concept:** The model is trained for a limited number of epochs ($\sim 10$) to prove the loss function works, resulting in a declining loss curve and a non-zero $\text{mAP}$ (e.g., **$\approx 0.04 \text{mAP}_{50}$**). This is sufficient proof of mechanism correctness.

### `utils.py` (Helper Functions)

Contains critical helper functions:
* `sinusoidal_pos_encode_2d()`: The from-scratch implementation of the positional encoding.
* `plot_pred()`: Visualization logic for drawing predicted bounding boxes.

