# overcookedExperiments
# Multi-Agent PPO (MAPPO) in Overcooked-AI 🍳🤝

This repository contains my implementation of **Multi-Agent Proximal Policy Optimization (MAPPO)** with a **Centralized Training and Decentralized Execution (CTDE)** architecture, applied to the [Overcooked-AI environment](https://github.com/HumanCompatibleAI/overcooked_ai).

The goal is to study how MAPPO enables **emergent collaboration** between agents in cooperative cooking tasks, and how training on **multiple layouts** improves **generalization** compared to single-layout training.

---

## 🚀 Features
- **MAPPO implementation** with shared policy and centralized critic
- **CTDE architecture** to address non-stationarity and credit assignment
- **Support for multiple layouts** (e.g., `cramped_room`, `asymmetric_advantages`, `bottleneck`, `forced_coordination`)
- **GIF rollout generation** for qualitative behavior visualization
- **Evaluation pipeline** with mean ± std reward reporting

...
