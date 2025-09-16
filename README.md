# 💬 Sentiment-Analysis-on-Omicron-Variant-from-Tweets using NLP
## 🌐 Project Overview
This project is designed to analyze the sentiment expressed in textual data—whether it's positive, negative, or neutral. Sentiment Analysis is a common application of Natural Language Processing (NLP) used in fields like marketing, product analysis, customer support, and public opinion mining.
* Natural Language Processing (NLP) in machine learning is a field that enables computers to understand, interpret, and generate human language. Essentially, it's about giving machines the ability to communicate with us in a way that feels natural, whether it's through text or speech. 

The analysis leverages libraries such as NLTK, TextBlob, and scikit-learn, incorporating both rule-based preprocessing and machine learning classification techniques.

## 🔎 Problem Statement
Text data, especially from social media, news, and product reviews, contains valuable insights about user opinions. The objective is to build a model that can:
* Clean and preprocess raw text,
* Extract meaningful features,
* Train a classifier,
* Predict the sentiment polarity of unseen text data.

## 📁 Dataset Overview
The dataset contains textual entries (possibly tweets, reviews, or articles) along with labels indicating sentiment:
* Positive
* Negative
* Neutral
The text often contains HTML content, stopwords, slang, and emotive expressions—necessitating a rigorous preprocessing pipeline.

## 🧹 Text Preprocessing
### Why preprocessing is important:
Raw text is noisy and needs to be cleaned to extract meaningful features for machine learning models. This project applies the following:
### HTML Tag Removal
  Using BeautifulSoup to strip out tags and special HTML entities.
### Lowercasing
  To ensure uniformity (e.g., Happy and happy are treated the same).
### Stopword Removal
  Common words like the, is, and are removed using NLTK’s stopword list.
### Punctuation Removal
  Non-alphanumeric characters are stripped to simplify tokenization.
### Stemming and Lemmatization
  * Stemming (via PorterStemmer or LancasterStemmer) reduces words to base form (e.g., "running" → "run").
  * Lemmatization uses vocabulary and morphological analysis (e.g., "better" → "good").
### Tokenization
  Sentences are split into words using word_tokenize.
### Unicode Normalization
  Special characters and symbols are normalized using Python’s unicodedata.

## 🧰 NLP Libraries Used
* NLTK (Natural Language Toolkit): Core preprocessing (stopwords, stemmers, tokenizers)
* TextBlob: High-level NLP API for sentiment scoring
* BeautifulSoup: HTML cleaning
* WordCloud: Visualizes word frequency
* Scikit-learn: Machine learning pipeline and metrics

## 🎯 Sentiment Scoring with TextBlob
### TextBlob provides:
* Polarity: Float value in [-1.0, 1.0], where negative is negative sentiment, and positive is positive sentiment.
* Subjectivity: Float value in [0.0, 1.0], where 0 is factual and 1 is opinion-based.
### The score is used to classify text into:
* Positive (polarity > 0)
* Neutral (polarity = 0)
* Negative (polarity < 0)

## 📈 Feature Engineering & Classification
The cleaned and scored data is used to train classification models. The pipeline includes:
* TF-IDF Vectorization: Converts text into numerical vectors representing importance of words across documents.
* Model Training using algorithms like:
 * Logistic Regression
 * Naive Bayes
 * Support Vector Machines (SVM)

## 📊 Evaluation Metrics
  To evaluate the performance of classification models:
### ✅ Accuracy
  Proportion of correct predictions over total predictions.
### 📉 Precision, Recall, F1-Score
  * Precision: TP / (TP + FP) — how many predicted positives are actually positive.
  * Recall: TP / (TP + FN) — how many actual positives were correctly predicted.
  * F1-Score: Harmonic mean of precision and recall.

## 🔄 Confusion Matrix
Shows breakdown of predictions vs actual labels:
* TP: Correctly predicted positive
* TN: Correctly predicted negative
* FP: False positive
* FN: False negative
Visualized using seaborn heatmaps.

## 🌐 Visualizations
* WordClouds: Illustrate most frequent words across sentiments.
* Bar Charts: Display sentiment distribution.
* Histograms: Show polarity and subjectivity distribution.
* Heatmaps: For confusion matrices.

## 🧪 Experiments
Various models were trained and evaluated with different configurations of:
* Vectorization (TF-IDF, CountVectorizer)
* N-grams (unigrams, bigrams)
* Stemming vs. Lemmatization
* Model types and hyperparameters

## 📝 Summary
This project demonstrates an end-to-end pipeline for sentiment analysis using traditional NLP and ML methods. It includes thorough data cleaning, text feature engineering, model building, and evaluation. Visualizations help understand the sentiment patterns in the dataset. This setup can be extended to real-world applications like tweet analysis, movie reviews, and more.



















