# 💬 Financial Tweets Sentiment Analysis

### 👥 Project Members
- Gaspar Pereira  
- Iris Moreira  
- Rafael Borges  
- Rita Wang  

---

## 🧪 Experiments Workflow

<p align="center">
  <img width="800" alt="Experiments Workflow" src="https://github.com/user-attachments/assets/82f28c37-1cbf-4c4c-9fb8-05774878c9c5" />
</p>

---

## 📊 Results

### 🧹 Preprocessing + Feature Extraction

| Preprocessing                                                                 | Feature Extractor                                                                 | Classifier          | F1-macro (avg) |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------|----------------|
| Remove stop words, Stemming, Regex, Lowercase, Remove punctuation           | TF-IDF <br><sub>max_df: 0.8 \| min_df: 6 <br> ngram_range: (1, 2)</sub>         |                     | 71.75          |
| Remove stop words, Lemmatization, Regex, Lowercase, Remove punctuation      | Glove Twitter 200                                                                | Logistic Regression | 62.68          |
| -                                                                            | Encoder (*zcharaf*)                                                              |                     | 92.60          |
|                                                                              | Encoder (*HugMaik*)                                                              |                     | 87.62          |
|                                                                              | Encoder (*RashidNLP*)                                                            |                     | **95.58**      |

**Table 1**: Evaluation of the top performing preprocessing methods with respective feature extractors

---

### 🤖 Classifier Comparison (with RashidNLP embedding)

| Preproc | Embedder   | Classifier        | F1-macro (avg) |
|---------|------------|-------------------|----------------|
| -       | RashidNLP  | Logistic Regression <br><sub>(penalty = l1 \| C = 0.1 \| solver = liblinear)</sub>         | 97.18          |
|         |            | KNN <br><sub>(neighbors = 15 \| weights = distance)</sub>                                   | 96.78          |
|         |            | Random Forest <br><sub>(n_estimators = 100 \| max_depth = 10 \| min_samples_leaf = 2)</sub> | 96.91          |
|         |            | SVM <br><sub>(C = 10 \| kernel = rbf \| gamma = scale \| probability = True)</sub>          | **97.20** *    |
|         |            | Encoder Head                                                                                 | 86.59          |
|         |            | Llama-3.1-SuperNova-Lite                                                                      | 71.87          |

**Table 2**: Evaluation of the top performing classification models

---

## 🏁 Final Result on Test Set

- **F1-macro**: 🎯 **0.98**

<p align="center">
  <img width="700" alt="Final Results" src="https://github.com/user-attachments/assets/dd3dc757-b54f-420a-82cf-d1995eca3ea7" />
</p>
