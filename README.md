# DL & GenAI Project 2026 – Smart MCQ Solver Challenge

## Overview
This repository contains the work completed for the DL & GenAI Project 2026 based on the Kaggle competition: **Smart MCQ Solver Challenge**.  
The objective of the challenge is to build intelligent systems capable of ranking the most probable answers for multiple-choice questions and optimizing performance using the MAP@3 evaluation metric.

## Competition Information
* **Competition:** Smart MCQ Solver Challenge
* **Platform:** Kaggle
* **Notebook:** DL-22f3001900-notebook-t22026-milestone-5
* **Repository:** dl-genai-project-26-t1

---

## Repository Structure
This repository follows a milestone-based workflow.

| Branch | Description |
| :--- | :--- |
| `milestone-1` | Exploratory Data Analysis, Text Processing, TF-IDF Baselines |
| `milestone-2` | Modern Architectures, Context-Aware Embeddings, NLI & Generative QA |
| `milestone-3` | Retrieval-Augmented Generation (RAG) Pipelines & Deep Cross-Encoder Reranking |
| `milestone-4` | Parameter-Efficient Fine-Tuning (PEFT) & AutoModelForMultipleChoice Fine-Tuning |
| `milestone-5` | Competition Metric Mastery, Inference Optimization, Ensembling & Test-Time Augmentation (TTA) |
| `main` | Final integrated project solution |

*Each milestone is developed independently and maintained in its corresponding branch.*

---

## Milestone 5: Competition Metric Mastery, Multi-Model Ensembling & Test-Time Augmentation (TTA)
**Topics covered:**
* **Multi-Model Inference & Probability Calibration:** Deployed fine-tuned sequence classification checkpoints (`microsoft/deberta-v3-small` and `roberta-base`) across multi-choice configurations. Extracted raw logit arrays and converted them to uniform probability distributions via standard multi-class Softmax functions.
* **Simple & Weighted Ensembling Strategies:** Investigated variance reduction across models. Developed a dual-tiered blending pipeline contrasting simple probability ensembling against a performance-biased weighted probability average ($0.70 \times P_{\text{DeBERTa}} + 0.30 \times P_{\text{RoBERTa}}$).
* **Top-K Selection & Competition Output Layouts:** Structured prediction output handlers to parse multi-class probability indices into Kaggle-compliant `Top-3` prediction strings (e.g., `C A E`), prioritizing the most confident choice rankings.
* **Test-Time Augmentation (TTA) Pipelines:** Engineered prompt robustification by introducing prefix instruction injections (`"Answer the following multiple-choice question carefully:"`) at runtime. Evaluated stability shifts and boundary decision updates by averaging standard and instruction-augmented validation passes.
* **Confidence Gain Dynamics:** Tracked predictive behavior transformations before and after model blending. Measured the absolute variation in maximum probability boundaries ($\text{Confidence Gain} = \text{Ensemble Confidence} - \text{DeBERTa Confidence}$) to identify structural safety regions.
* **Downstream Competition Metric Optimization:** Built local evaluation engines to systematically compute Mean Average Precision @ 3 (**MAP@3**) tracking multi-choice rank distributions ($1.0$, $0.5$, $0.333$, or $0.0$) across cross-validation subsets.

**Key outputs:**
* Fully working ensembling pipeline tracking multi-model probabilities for thousands of parallel test instances.
* Generation of an optimized, Kaggle-compliant `submission.csv` tracking exact multi-option token ordering sequences (`id,prediction`).
* Measurable distribution of Test-Time Augmentation (TTA) changes across cross-validation samples.
* Structural analytics tracing confidence shifts, Top-1 predictions flips, and Top-3 ordered ranking alterations between baseline models and the blended ensemble.
* Deterministic locally validated **MAP@3** evaluation reports rounded up to 4 decimal places.

---

## Technologies Used
* Python
* Pandas & NumPy
* PyTorch (`torch`, `torch.nn.functional`)
* Hugging Face `transformers` & `datasets`
* Ensembling (Simple Blending / Weighted Averaging)
* Test-Time Augmentation (TTA)
* Evaluation Metrics (MAP@3 Engine)
* Kaggle Accelerated Environments (GPUs)

---

## Future Work
Upcoming milestones will explore:
* Blending Generative Decoder Large Language Models (LLMs) with Encoder Softmax arrays.
* Post-processing ranking refinements via dynamic calibration techniques.
* Advanced scaling using hyperparameter optimization for ensemble weights.

---

## Author
**Vedant M. Kakde** IIT Madras – BS Degree Program  
Roll Number: 22F3001900
