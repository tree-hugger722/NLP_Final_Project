from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Seethal/sentiment_analysis_generic_dataset")

model = AutoModelForSequenceClassification.from_pretrained("Seethal/sentiment_analysis_generic_dataset")