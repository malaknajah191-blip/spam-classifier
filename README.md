# Spam Message Classifier
[Try the live app]([[https://spam-classifier-lc5xubzzypjpy8e9jcbqtw.streamlit.app](https://spam-classifier-lc5xubzzypjpy8e9jcbqtw.streamlit.app/)](https://spam-classifier-lc5xubzzypjpy8e9jcbqtw.streamlit.app/))

A machine learning project that classifies SMS text messages as spam or not spam, using natural language processing and a Naive Bayes classifier. Includes an interactive Streamlit app for live predictions.

## Overview

This project takes a real-world dataset of SMS messages and builds a model that detects spam. It walks through a complete NLP workflow: cleaning raw text, converting it into numerical features with TF-IDF, training and comparing two models, and deploying the best one as an interactive web app.

## Dataset

[SMS Spam Collection](https://www.kaggle.com/code/mehakiftikhar/ml-for-email-spam-detection-nlp-classification) 5,572 real SMS messages in English, labeled as `ham` (legitimate) or `spam`. Originally compiled for academic spam research and widely used in NLP courses and research.

## Features

- Text cleaning: lowercasing, removing punctuation and numbers, removing duplicate messages
- TF-IDF vectorization to convert raw text into numerical features
- Compared two classification models: Naive Bayes and Logistic Regression
- Selected the best-performing model automatically based on F1 score
- Interactive Streamlit app for real-time spam detection with confidence scores

## Model results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Naive Bayes** | **97.67%** | **100%** | **81.25%** | **0.897** |
| Logistic Regression | 96.41% | 97.89% | 72.66% | 0.834 |

Naive Bayes performed best, with perfect precision (it never wrongly flagged a real message as spam) and 81% recall (it correctly caught 81% of actual spam messages).

## Technologies used

- Python 3
- pandas
- scikit-learn (TF-IDF vectorizer, Naive Bayes, Logistic Regression)
- Streamlit (web app)
- joblib (model persistence)




## Project structure
spam-classifier/
├── data/
│   └── spam.csv          
├── train_model.py        
├── app.py                # Streamlit app for interactive predictions
├── requirements.txt
└── README.md

## What I learned

This project helped me practice:
- Cleaning and preprocessing real-world text data
- Converting text into numerical features using TF-IDF
- Comparing classification models using precision, recall, and F1 score (not just accuracy)
- Understanding why accuracy alone is misleading on imbalanced datasets
- Deploying a trained NLP model as an interactive web app

## Possible improvements

- Add more advanced text features (n-grams, message length, presence of links/numbers)
- Try other models (SVM, Random Forest) or a simple neural network
- Add cross-validation for more robust model comparison

## Author

Built by Malak Najah as a personal machine learning practice project.
