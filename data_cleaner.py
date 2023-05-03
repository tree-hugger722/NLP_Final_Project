import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict

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
sentiment_dict = defaultdict(int)

def get_sentiment(text: str):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length = 514, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    sentiment_dict[model.config.id2label[predicted_class_id]] += 1
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

            max_lengths = [500, 1000, 1500, 1750, 2000, len(text)]
            num_tries = len(max_lengths)-1
            while num_tries > 0:
                try:
                    text = truncate(text, max_lengths[num_tries])
                    print("Length: ", len(text))
                    sentiment = get_sentiment(text)
                    break
                except:
                    num_tries -= 1

            if sentiment: # in case we don't get sentiment after trying
                row.append(sentiment)
                writer.writerow(row)
            

def truncate(output_string: str, max_length) -> list:
    return output_string[0:max_length]

add_sentiment("dataset_bbc/train.csv", "dataset_bbc/train_w_sentiment.csv")
print(sentiment_dict)
# add_sentiment("dataset_bbc/test.csv", "dataset_bbc/test_w_sentiment.csv")