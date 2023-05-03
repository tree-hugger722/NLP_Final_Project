import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("abhishek/autonlp-bbc-news-classification-37229289")

model = AutoModelForSequenceClassification.from_pretrained("abhishek/autonlp-bbc-news-classification-37229289")

text_test = "ha"

inputs = tokenizer(text_test, return_tensors="pt")

print(inputs)

with torch.no_grad():
    logits = model(**inputs).logits

print(logits)
predicted_class_id = logits.argmax().item()
print(predicted_class_id)
print(model.config.id2label[predicted_class_id])