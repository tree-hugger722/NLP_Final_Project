# NLP_Final_Project
Emma Neil ('23) and Yufeng Wu ('24)

This repository contains the code for a final project for Prof. Katie Keith's Natural Language Processing Course in Spring 2023 at Williams College.

We are examining the effectiveness of Hinton et al.'s method to compress a large machine learning model into a much smaller one using the distillation loss discussed in the ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531) (2015). We attempt to distill a sentiment analysis model which is trained on Twitter data.

### Models
"Teacher" Model : [Twitter-Roberta-Base-Sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment?text=Apple%27s+new+product+is+not+meeting+the+expectation)
- Trained on : [Tweet-Eval](https://github.com/cardiffnlp/tweeteval)

"Student" Model:
- Trained on: [MTEB Tweet_Sentiment_Extraction](https://huggingface.co/datasets/mteb/tweet_sentiment_extraction/viewer/mteb--tweet_sentiment_extraction/train?p=274)
- Input data embedded using pre-trained [GloVE vector embeddings](https://nlp.stanford.edu/projects/glove/) (2B tweets, 27B tokens, 1.2M vocab, uncased)
- Distilled from Teacher model

Baseline Models:
- Logistic Regression with Bag-of-Words input
- Logistic Regression with input data embedded using pre-trained [GloVE vector embeddings](https://nlp.stanford.edu/projects/glove/) (2B tweets, 27B tokens, 1.2M vocab, uncased)
- Student Model, trained only on the "hard labels" (true sentiment output) with same GloVE-embedded input as above

### Navigating the Code Base
`dataset_twitter` : directory containing test, train, and dev CSVs

`baseline_sentence_embedding` : code to run each models

`data_to_csvs` : processes input data and splits training set into a train.csv and dev.csv; also adds soft labels (collected by running the teacher model on the input data) to train.csv.