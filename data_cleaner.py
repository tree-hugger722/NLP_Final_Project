import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

'''
Citation:

We asked ChatGPT with the following prompt:

Write a python program that does the following:
open a csv file, process the second column of it called text, pass it to the following sentiment analysis pipeline, then finally append it to the csv file of the outputted sentiment.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

inputs = tokenizer("Hello, my dog is stupid. sad", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

print(logits)
predicted_class_id = logits.argmax().item()
print(predicted_class_id)
print(model.config.id2label[predicted_class_id])
'''
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def get_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

def add_sentiment(input_file: str, output_file: str) -> None:
    # Read input CSV file

    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        header = next(reader)
        header.append("sentiment")
        writer.writerow(header)
        
        # Process and write rows
        for row in reader:
            text = row[1]
            print(len(text))
            sentiment = get_sentiment(text)
            row.append(sentiment)
            writer.writerow(row)

# TODO: truncate input to a max size

add_sentiment("dataset_bbc/train.csv", "dataset_bbc/train_w_sentiment.csv")
add_sentiment("dataset_bbc/test.csv", "dataset_bbc/test_w_sentiment.csv")