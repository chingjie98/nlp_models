## BBC News Classifier
This project is a text classification system built to classify BBC news articles into predefined categories using machine learning techniques.

---

### Table of Contents
1. [Overview](#overview)
2. [Datasets](#datasets)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Future Work](#future-work)

---

### Overview
The goal of this project is to classify BBC news articles into their respective categories using a Naive Bayes Multinomial classifier. <br>
The embedding technique used is Count Vectorizer, which converts textual data into a bag-of-words representation for model training.

---

### Datasets
Source: BBC News Dataset on Kaggle (https://www.kaggle.com/c/learn-ai-bbc)<br>
The dataset is organized into the following directory structure:

```plaintext
data/
├── raw/               # Raw BBC news articles
├── processed/         # Preprocessed articles for training
└── external/          # Additional datasets (if any)
```
Dataset Information:
- Number of Articles: 2225
- Categories: Business, Sports, Politics, Technology, Entertainment

---

### Installation
#### Prerequisites
Python 3.9+ installed
Virtual environment setup (e.g., Conda, venv)

To clone the repository, use the following command:
```bash
git clone https://github.com/chingjie98/news-classifier.git
```
Navigate to the project directory:
```bash
cd news-classifier
```
Install the dependencies:
```bash
pip install -r requirements.txt
```
---

### Project Structure
```plaintext
news-classifier/
├── config/                   # Configuration files
├── data/                     # Dataset storage
├── notebooks/                # Jupyter notebooks for exploration
├── src/                      # Source code
│   ├── data/                 # Data processing scripts
│   ├── models/               # Model training and evaluation scripts
│   ├── evaluation/           # Metrics and evaluation utilities
│   └── utils/                # Helper functions
├── tests/                    # Unit tests
├── train/                    # Model checkpoints, logs, metrics
├── deployment/               # Model deployment
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

### Model Training
Preprocessing:

- Clean and preprocess the raw text data.
- Convert text into numerical format using Count Vectorizer.
- Split the dataset into training and testing sets.

Command to preprocess:

```bash
python src/data/preprocess.py --input data/raw --output data/processed
```

Training:

- Train a Naive Bayes Multinomial Classifier on the processed data.

Command to train the model:

```bash
python src/models/train.py --config config/config.yaml
```
---
### Evaluation
Metrics:

- Accuracy: X% (update after evaluation)
- Precision, Recall, F1-Score: (List specific metrics here)
- Evaluation Script:

```bash
python src/evaluation/evaluate.py --model train/checkpoints/model.pkl
```
---
### Future Work
- Experiment with other embedding techniques like TF-IDF or word embeddings.
- Implement additional classifiers (e.g., SVM, Logistic Regression).
- Build an API for real-time classification.
- Integrate a pipeline for automated dataset updates and retraining.
