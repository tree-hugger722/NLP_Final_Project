# NLP_Final_Project

Emma Neil ('23) and Yufeng Wu ('24)

This repository contains the code for a final project for Prof. Katie Keith's Natural Language Processing Course in Spring 2023 at Williams College.

We are examining the effectiveness of Hinton et al.'s method from in ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531) (2015) to compress a large machine learning model into a much smaller one using the distillation loss technique. We attempt to distill a sentiment analysis model which is trained on Twitter data.

## Models
"Teacher" Model : [Twitter-roBERTa-base for Sentiment Analysis](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- Trained on : [TweetEval](https://github.com/cardiffnlp/tweeteval)

"Student" Model:
- Trained on: [TweetEval](https://github.com/cardiffnlp/tweeteval)
- Input data embedded using pre-trained [GloVE vector embeddings](https://nlp.stanford.edu/projects/glove/) (2B tweets, 27B tokens, 1.2M vocab, uncased, 50d)
- Distilled from Teacher model

Baseline Models:
- Logistic Regression with Bag-of-Words input
- Logistic Regression with input data embedded using pre-trained [GloVE vector embeddings](https://nlp.stanford.edu/projects/glove/) (2B tweets, 27B tokens, 1.2M vocab, uncased, 50d)
- Student Model, trained only on the "hard labels" (true sentiment output) with same GloVE-embedded input as above

## Pipeline
1) Load dataset

Run all cells in `get_data.ipynb` to load test.csv, train.csv, and validation.csv into the `data/` folder.

2) Pre-process data

Run all cells in `preprocess_data.ipynb` to create new CSVs with soft labels to the training set and hard labels to all sets. Each new dataset is  called `<train/validation/test>_preprocessed.csv` and is located in the `data/` folder.

3) Download GloVE embeddings

Download the GloVE embeddings pre-trained on Twitter data from the [Stanford website](https://nlp.stanford.edu/projects/glove/). Make sure to add the 50d .txt file to the repository.

4) Run baseline BoW model

Run all cells in `baseline_BOW.ipynb` and observe train, validation, and test results in the Jupyter Notebook.

9) Run other baselines and student model + evaluate
