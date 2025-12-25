# Liquid Neural Network Comparison: Euler-Based vs. CfC-Based Architectures

This repository contains the implementation of two distinct **Liquid Neural Network (LNN)** architectures designed for high-performance pixel-wise classification, specifically for identifying DDOS attack patterns in spatial data. The project evaluates the trade-offs between traditional numerical ODE solvers and modern analytical closed-form solutions for continuous-time dynamics.

---

## 1. Overview
Liquid Neural Networks are a class of continuous-time RNNs that model hidden states using differential equations, allowing them to adapt to time-varying data. 

* **Model A: Euler-Based LNN** — Uses a discrete numerical ODE solver to approximate the system's state over time.
* **Model B: CfC-Based LNN** — Utilizes a Closed-form Continuous solution for superior computational efficiency and numerical stability.

---

## 2. Model 1: Euler-Based Liquid Neural Network
This model is designed to discover continuous-time dynamics using a simple numerical ODE solver.

### Architecture & Layers
* **Input Layer**: A 2D convolutional layer ($nn.Conv2d$) transforms the RGB input (3 channels) into a 128-channel feature map using a $3 \times 3$ kernel ($padding=1$) and a $tanh$ activation function.
* **Liquid Layer**: Models continuous dynamics via an ODE approximated using the **Euler method**.
  * **The ODE**: 
    $$\frac{dh}{dt} = \frac{1}{\tau} [-h(t) + \tanh(W_{liq}(h+u) + b_{liq})]$$
  * **The Euler Update**:
    $$h(t+\Delta t) = h(t) + \Delta t \cdot \frac{dh}{dt}$$
  * **Parameters**: Fixed time constant $\tau = 1.0$; hidden size = 128.
* **Output Layer**: A $1 \times 1$ convolutional layer maps the hidden state to per-pixel class scores for 10 DDOS classes.



---

## 3. Model 2: Closed-form Continuous (CfC) LNN
The **Closed-form Continuous (CfC)** model represents an evolution in LNN design by replacing the iterative ODE solver with an analytical solution.

### Key Features
* **Analytical Solver**: Bypasses iterative integration, instead using a closed-form gating mechanism to define the hidden state $h(t)$.
* **Zero Integration Error**: Eliminates the numerical drift and approximation errors associated with the Euler method.
* **Robustness**: Inherently more stable and capable of handling irregularly sampled temporal data.



---

## 4. Computational Efficiency & Optimization
The two architectures exhibit distinct performance profiles regarding hardware utilization and throughput.

| Feature | Euler-Based LNN | CfC-Based LNN |
| :--- | :--- | :--- |
| **Solver Type** | Incremental Numerical | Single-step Analytical |
| **Efficiency** | Computationally intensive | Significantly faster |
| **GPU Utilization** | Sequential update bottleneck | Optimized for CUDA 11.7 |
| **Integration Error** | Present | Zero (Analytical) |
| **Total Parameters** | ~150,000 | ~150,000 |



---

## 5. Technical Specifications
* **Input Dimension**: $128 \times 128 \times 3$ (RGB Image)
* **Output Dimension**: $128 \times 128 \times 10$ (Class Scores)
* **Target Hardware**: NVIDIA GPUs (CUDA 11.7+)
* **Framework**: PyTorch
