# Financial Tweets Sentiment Analysis

### Project Members
- Gaspar Pereira  
- Iris Moreira  
- Rafael Borges  
- Rita Wang  

---

## Experiments Workflow

<p align="center">
  <img width="800" alt="Experiments Workflow" src="https://github.com/user-attachments/assets/82f28c37-1cbf-4c4c-9fb8-05774878c9c5" />
</p>

---

## Results

We evaluated various combinations of preprocessing techniques with different text embedders:
- TF-IDF
- Pre-trained and fine-tuned GloVe
- Pre-trained and fine-tuned transformer-based encoders (e.g., BERT)
  

| Preprocessing                                                                 | Feature Extractor                                                                 | Classifier          | F1-macro (avg) |
|------------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------|----------------|
| Remove stop words, Stemming, Regex, Lowercase, Remove punctuation           | TF-IDF <br><sub>max_df: 0.8 \| min_df: 6 <br> ngram_range: (1, 2)</sub>         |  Logistic Regression                   | 71.75          |
| Remove stop words, Lemmatization, Regex, Lowercase, Remove punctuation      | Glove Twitter 200                                                                | "" | 62.68          |
| -                                                                            | Transformer Encoder <br>(*zcharaf*)                                                              | ""                     | 92.60          |
| -                                                                             | Transformer Encoder (*HugMaik*)                                                              | ""                     | 87.62          |
| -                                                                             | Transformer Encoder (*RashidNLP*)                                                            | ""                     | **95.58** *     |


---

We selected the top-performing embedder and tested it with various classifiers <br>
Additionally, we explored zero-shot classification using Llama.



| Preproc | Embedder   | Classifier        | F1-macro (avg) |
|---------|------------|-------------------|----------------|
| -       | RashidNLP  | Logistic Regression <br><sub>(penalty = l1 \| C = 0.1 \| solver = liblinear)</sub>         | 97.18          |
|         | ""           | KNN <br><sub>(neighbors = 15 \| weights = distance)</sub>                                   | 96.78          |
|         | ""           | Random Forest <br><sub>(n_estimators = 100 \| max_depth = 10 \| min_samples_leaf = 2)</sub> | 96.91          |
|         | ""           | SVM <br><sub>(C = 10 \| kernel = rbf \| gamma = scale \| probability = True)</sub>          | **97.20** *    |
|         | ""           | Encoder Head                                                                                 | 86.59          |
|         | ""           | Llama-3.1-SuperNova-Lite                                                                      | 71.87          |


---

Final model configuration:
- Encoder with RashidNLP (fine-tuned DeBERTa)
- Classification with SVM 

## Final Result on Test Set

- **F1-macro**: **0.98**

<p align="center">
  <img width="700" alt="Final Results" src="https://github.com/user-attachments/assets/dd3dc757-b54f-420a-82cf-d1995eca3ea7" />
</p>
