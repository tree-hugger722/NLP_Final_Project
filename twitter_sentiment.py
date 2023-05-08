import pandas as pd

def load_csv():
    train_df = pd.read_json("dataset_twitter/train.jsonl", lines=True)
    test_df = pd.read_json("dataset_twitter/test.jsonl", lines=True)

    train_csv = train_df.to_csv("dataset_twitter/train.csv")
    test_csv = test_df.to_csv("dataset_twitter/test.csv")

if __name__ == "__main__":
    load_csv()