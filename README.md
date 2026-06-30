# DL & GenAI Project 2026 – Smart MCQ Solver Challenge

## Overview
This repository contains the work completed for the DL & GenAI Project 2026 based on the Kaggle competition: **Smart MCQ Solver Challenge**.  
The objective of the challenge is to build intelligent systems capable of ranking the most probable answers for multiple-choice questions and optimizing performance using the MAP@3 evaluation metric.

## Competition Information
* **Competition:** Smart MCQ Solver Challenge
* **Platform:** Kaggle
* **Notebook:** DL-22f3001900-notebook-t22026-milestone-2
* **Repository:** dl-genai-project-26-t1

---

## Repository Structure
This repository follows a milestone-based workflow.

| Branch | Description |
| :--- | :--- |
| `milestone-1` | Exploratory Data Analysis, Text Processing, TF-IDF Baselines |
| `milestone-2` | Modern Architectures, Context-Aware Embeddings, NLI & Generative QA |
| `milestone-3` | To be updated |
| `main` | Final integrated project solution |

*Each milestone is developed independently and maintained in its corresponding branch.*

---

## Milestone 1: Classical NLP Baselines
**Topics covered:**
* Dataset exploration & Answer distribution analysis
* Text normalization & Stop-word removal
* Vocabulary analysis
* TF-IDF feature extraction & Cosine similarity based ranking
* Majority class baseline & MAP@3 evaluation

**Key outputs:**
* Frequency analysis of answer labels
* Vocabulary statistics
* TF-IDF feature space generation
* Similarity-based answer ranking baseline

---

## Milestone 2: Modern Architectures & Context-Aware Embedding Pipelines
**Topics covered:**
* **Hugging Face Ecosystem:** Transitioned dataset workflows from Pandas to the `datasets` abstraction layer, handling unstructured input formatting and sanitization pipelines.
* **Structural Tokenization:** Explored the structural limits of `bert-base-uncased`, mapping rigid vocabulary dimensions, tracking special control indices (`[SEP]`), and applying static sequence limits via array padding.
* **Attention Mechanics & Hidden States:** Isolated multi-head hidden tensors ($768$ embedding space divided into $12$ distinct heads of size $64$). Tracked exact pooler layer activation vectors (`[CLS]` token dimensions) and layer-normalized sequence-to-sequence attention weights.
* **Semantic Ranking Pipelines:** Engineered shared vector context representations via `all-MiniLM-L6-v2`. Quantified strict validation improvements across the training distribution against the TF-IDF baseline using **MAP@3** evaluations.
* **Zero-Shot Inference Mechanics:** Evaluated predictive probability shifts between competing Softmax objectives (`multi_label=False`) and threshold-independent Sigmoid arrays (`multi_label=True`) via `bart-large-mnli`.
* **Generative Small Language Models (SLMs):** Standardized decoder-only prompt mapping rules using `google/flan-t5-small`, bypassing token duplication anomalies via localized `.generate()` tensor constraints.

**Key outputs:**
* Formatted text token shapes: `[2000, 128]`
* Calculated attention distribution weights for key conceptual terms
* Dense vector semantic retrieval maps over the training collection
* MAP@3 baseline metric jumps from sparse keywords to deep semantic layers
* Clean isolated token predictions (`A`/`B`) from text-to-text generation setups

---

## Technologies Used
* Python
* Pandas & NumPy
* Scikit-learn
* PyTorch (`torch`)
* Hugging Face `transformers` & `datasets`
* Sentence-Transformers
* Kaggle Notebooks

---

## Future Work
Upcoming milestones will explore:
* Fine-tuning encoder-decoder architectures
* Retrieval-Augmented Generation (RAG) methodologies
* Ensemble ranking strategies
* Competition-grade inference optimization pipelines

---

## Author
**Vedant M. Kakde** IIT Madras – BS Degree Program  
Roll Number: 22F3001900
