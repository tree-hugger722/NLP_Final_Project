import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")

model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")

inputs = tokenizer("Hello, my dog is stupid. sad", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

print(logits)
predicted_class_id = logits.argmax().item()
print(predicted_class_id)
print(model.config.id2label[predicted_class_id])

