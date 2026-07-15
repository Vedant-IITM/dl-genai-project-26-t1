# DL & GenAI Project 2026 – Smart MCQ Solver Challenge

## Overview
This repository contains the work completed for the DL & GenAI Project 2026 based on the Kaggle competition: **Smart MCQ Solver Challenge**.  
The objective of the challenge is to build intelligent systems capable of ranking the most probable answers for multiple-choice questions and optimizing performance using the MAP@3 evaluation metric.

## Competition Information
* **Competition:** Smart MCQ Solver Challenge
* **Platform:** Kaggle
* **Notebook:** DL-22f3001900-notebook-t22026-milestone-4
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

## Milestone 3: Retrieval-Augmented Generation (RAG) & Two-Stage Attention Reranking
**Topics covered:**
* **Vector Indexing & Dense Space Retrieval:** Initialized standard index repositories using `faiss.IndexFlatL2` populated by `all-MiniLM-L6-v2` sequence projections. Examined bi-encoder vector limits where exact contextual targets slip lower down the retrieval ranks.
* **Two-Stage Reranking Architectures:** Implemented deep dual-input processing pipelines via a `cross-encoder/ms-marco-MiniLM-L-6-v2` network. Evaluated localized token interaction mechanics via cross-attention models to reposition baseline matches directly to rank `#1`.
* **Structural Context Window Constraints:** Modeled structural physical constraints of context sequence injection windows using `bert-base-uncased` subword processing token structures to scale concatenated multi-chunk documents.
* **Adversarial RAG & Hallucination Vectors:** Proved empirical system reliance on external text parameters ("Garbage In, Garbage Out") by analyzing target metric decay shifts when exposed to unaligned / corrupted knowledge bases (`index 999`).
* **Database Accuracy Hit Optimization:** Standardized retrieval evaluations across indices via automated string sequence presence tracking to evaluate systemic dataset lookup accuracy bounds at $k=5$.
* **End-to-End RAG Execution Optimization:** Engineered an integrated multi-tier inference system (Bi-encoder Retrieval $\rightarrow$ Cross-encoder Attention Sorting $\rightarrow$ Prefix Augmentation $\rightarrow$ Zero-shot Multi-class Softmax Probability Scoring) to hit state-of-the-art **MAP@3** margins.

**Key outputs:**
* Fully working downstream two-stage search framework engine.
* Measurable retrieval target accuracy map jumps from an un-augmented pipeline baseline ($0.384$) up to an explicit context injection configuration ($0.989$).
* Systematic index verification measuring an exact $73.0\%$ contextual inclusion hit rate over multi-row target samples.
* Reached an advanced integrated **MAP@3** validation metric score of **0.975** processing through the comprehensive retrieval system infrastructure.

---

## Milestone 4: Parameter-Efficient Fine-Tuning (PEFT) & Task-Specific MCQ Adaptation
**Topics covered:**
* **AutoModelForMultipleChoice Architecture:** Integrated task-specific classification heads over encoder frameworks. Configured input tensor mappings for multi-option processing, handling batch inputs of shape `[batch_size, num_choices, seq_len]` (e.g., `[16, 5, 128]`).
* **Low-Rank Adaptation (LoRA) Integration:** Implemented parameter-efficient fine-tuning via `PEFT` targeting key attention weight projection layers (`query`, `value`). Tuned LoRA hyperparameters ($r=8$, $\alpha=16$) to drastically restrict active gradient tracking while preventing catastrophic forgetting.
* **Custom Dataset Tokenization Pipelines:** Built dynamic encoding maps translating raw strings into structural token collections. Addressed input limits using custom sequence chunkers, generating isolated attention masks and flat-array structures formatted for Hugging Face `Dataset` structures.
* **PyTorch Trainer Optimization:** Deployed robust training environments using Hugging Face's `Trainer` API. Configured specialized training arguments targeting step-wise execution, logging limits, gradient accumulation steps, and optimized resource allocation.
* **Softmax Probability Inference:** Extracted raw logit distribution outputs from fine-tuned architectures and projected them into standard class probability distributions to measure prediction confidences.

**Key outputs:**
* Successful integration of multiple-choice structural tensors tracking correct dim configurations of inputs (`Q5: 5` choices output).
* Verification of LoRA trainable parameters ($294,912$ active parameters) against frozen baseline weights.
* Complete local step execution running on active Kaggle environments tracking deterministic steps (`Q9: 4` global steps completed).
* Inference probability pipeline extracting Softmax distribution bounds for answer candidate arrays.

---

## Technologies Used
* Python
* Pandas & NumPy
* Scikit-learn
* PyTorch (`torch` & `torch.nn.functional`)
* Hugging Face `transformers` & `datasets`
* PEFT (`peft` with `LoraConfig`, `TaskType.SEQ_CLS`)
* FAISS (`faiss-cpu`)
* Sentence-Transformers (`CrossEncoder` / `SentenceTransformer`)
* Kaggle Notebooks

---

## Future Work
Upcoming milestones will explore:
* Fine-tuning encoder-decoder architectures
* Hyperparameter fine-tuning of retriever contexts
* Ensemble ranking strategies
* Competition-grade inference optimization pipelines

---

## Author
**Vedant M. Kakde** IIT Madras – BS Degree Program  
Roll Number: 22F3001900
