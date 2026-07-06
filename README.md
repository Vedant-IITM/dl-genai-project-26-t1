# Smart MCQ Solver Challenge

A machine learning project developed for the **Smart MCQ Solver Challenge**, comparing three different approaches for automatically solving multiple-choice questions.

The project demonstrates the complete workflow of text preprocessing, feature engineering, deep learning, pretrained transformers, experiment tracking, and Kaggle submission generation.



## Project Overview

The objective of this project is to predict the correct answer option for multiple-choice questions.

Three different modelling approaches were implemented and compared:

* **TF-IDF + Logistic Regression**
* **MLP Neural Network (Built from Scratch using PyTorch)**
* **Fine-tuned DistilBERT Transformer**

Each model was trained and evaluated independently, with experiment tracking performed using **Weights & Biases (W&B)**.



## Features

* Text preprocessing pipeline
* TF-IDF feature extraction
* Classical Machine Learning baseline
* Neural network implemented from scratch
* Fine-tuned pretrained transformer
* Experiment tracking using W&B
* Automatic Kaggle submission generation
* MAP@3 compatible prediction format



## Models Implemented

### 1. TF-IDF + Logistic Regression

* TF-IDF Vectorizer
* Word-level unigram and bigram features
* Logistic Regression classifier
* Fast baseline model



### 2. Multi-Layer Perceptron (Built from Scratch)

Implemented entirely in **PyTorch** without using pretrained neural networks.

Architecture includes:

* Fully connected layers
* ReLU activation
* Dropout regularization
* Adam optimizer
* Cross-Entropy Loss

This model satisfies the **Built from Scratch** requirement.



### 3. DistilBERT (Pretrained)

A pretrained **DistilBERT** transformer was fine-tuned on the MCQ dataset.

This model leverages contextual language representations and satisfies the **Pretrained Model** requirement.



## Experiment Tracking

All experiments were tracked using **Weights & Biases (W&B)**.

Logged metrics include:

* Training Loss
* Validation Loss
* Accuracy
* Weighted F1 Score

This enables comparison between different modelling approaches and reproducibility of experiments.



## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* Weights & Biases
* Kaggle Notebook Environment



## Results

The project compares three modelling approaches with different levels of complexity:

| Model                        | Type                               |
| ---------------------------- | ---------------------------------- |
| TF-IDF + Logistic Regression | Traditional Machine Learning       |
| MLP                          | Deep Learning (Built from Scratch) |
| DistilBERT                   | Pretrained Transformer             |

Performance metrics were monitored using Weights & Biases and evaluated on a validation split before generating Kaggle submissions.



## Future Improvements

* Hyperparameter optimization
* Ensemble learning
* Larger transformer models
* Cross-validation
* Model deployment using Hugging Face
* Interactive inference demo using Gradio



## Author

**Vedant Kakde**

Electronics and Telecommunication Engineering

Machine Learning • Deep Learning • Embedded Systems • AI
