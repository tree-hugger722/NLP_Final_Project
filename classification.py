import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("abhishek/autonlp-bbc-news-classification-37229289")

model = AutoModelForSequenceClassification.from_pretrained("abhishek/autonlp-bbc-news-classification-37229289")

text_test = '''
A man widely seen as the godfather of artificial intelligence (AI) has quit his job, warning about the growing dangers from developments in the field.

Geoffrey Hinton, 75, announced his resignation from Google in a statement to the New York Times, saying he now regretted his work.

He told the BBC some of the dangers of AI chatbots were "quite scary".

"Right now, they're not more intelligent than us, as far as I can tell. But I think they soon may be."

Dr Hinton also accepted that his age had played into his decision to leave the tech giant, telling the BBC: "I'm 75, so it's time to retire."

Dr Hinton's pioneering research on neural networks and deep learning has paved the way for current AI systems like ChatGPT.

In artificial intelligence, neural networks are systems that are similar to the human brain in the way they learn and process information. They enable AIs to learn from experience, as a person would. This is called deep learning.

The British-Canadian cognitive psychologist and computer scientist told the BBC that chatbots could soon overtake the level of information that a human brain holds.
'''

inputs = tokenizer(text_test, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

print(logits)
predicted_class_id = logits.argmax().item()
print(predicted_class_id)
print(model.config.id2label[predicted_class_id])