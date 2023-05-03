import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Seethal/sentiment_analysis_generic_dataset")

model = AutoModelForSequenceClassification.from_pretrained("Seethal/sentiment_analysis_generic_dataset")

inputs = tokenizer("This is a table.", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

print(logits)
predicted_class_id = logits.argmax().item()
print(predicted_class_id)
print(model.config.id2label[predicted_class_id])