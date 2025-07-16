# Financial Tweets Sentiment Analysis
### Text Mining 
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
| Remove stop words, Stemming, Regex, Lowercase, Remove punctuation           | TF-IDF <br><sub>max_df: 0.8 \| min_df: 6 <br> ngram_range: (1, 2)</sub>         |  Logistic Regression (LR)                   | 71.75          |
| Remove stop words, Lemmatization, Regex, Lowercase, Remove punctuation      | Glove Twitter 200                                                                | LR | 62.68          |
| -                                                                            | Transformer Encoder <br>(*zcharaf*)                                                              | LR                     | 92.60          |
| -                                                                             | Transformer Encoder (*HugMaik*)                                                              | LR                     | 87.62          |
| -                                                                             | Transformer Encoder (*RashidNLP*)                                                            | LR                     | **95.58** *     |


---

We selected the top-performing embedder and tested it with various classifiers <br>
Additionally, we explored zero-shot classification using Llama.



| Preproc | Embedder   | Classifier        | F1-macro (avg) |
|---------|------------|-------------------|----------------|
| -       | RashidNLP (RNLP) | Logistic Regression <br><sub>(penalty = l1 \| C = 0.1 \| solver = liblinear)</sub>         | 97.18          |
|         | RNLP           | KNN <br><sub>(neighbors = 15 \| weights = distance)</sub>                                   | 96.78          |
|         | RNLP           | Random Forest <br><sub>(n_estimators = 100 \| max_depth = 10 \| min_samples_leaf = 2)</sub> | 96.91          |
|         | RNLP           | SVM <br><sub>(C = 10 \| kernel = rbf \| gamma = scale \| probability = True)</sub>          | **97.20** *    |
|         | RNLP           | Encoder Head                                                                                 | 86.59          |
|         | -          | Llama-3.1-SuperNova-Lite <br><sub>zero-shot classification</sub>                                                                    | 71.87          |


---

Final model configuration:
- Encoder with RashidNLP (fine-tuned DeBERTa)
- Classification with SVM 

## Final Result on Test Set

- **F1-macro**: **0.98**
Best result on the course  
