import csv
import urllib.request

import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
)

from processing import process_csv, rmv_tags, demoji_tweet, rmv_stopwords, stem_tweet
import matplotlib.pyplot as plt


# Preprocess text (username and link placeholders)
def preprocess(text, lan="en", preprocess=True):
    if preprocess:
        text = rmv_tags(text)
        text = demoji_tweet(text, lan)
        text = rmv_stopwords(text, lan)
        text = stem_tweet(text, lan)

    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)

    return " ".join(new_text)


# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task = "sentiment"
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode("utf-8").split("\n")
    csvreader = csv.reader(html, delimiter="\t")
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)

# Tag 1 is negative, 0 positive
correct = 0
data_en = pd.read_csv("dataset/data_en_processed.csv").replace(np.nan, "", regex=True)

raw_results_n = {}
raw_results_p = {}

negative_author_ratios = []
positive_author_ratios = []
for author in tqdm(data_en["author_id"].unique()):
    data_author = data_en[data_en["author_id"] == author]
    true_y = data_author["tag"].iloc[0]
    author_negative = 0
    author_positive = 0

    for idx, row in data_author.iterrows():
        text = row["tweet"]
        encoded_input = tokenizer(text, return_tensors="pt")
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        # print(f"For tweet {text} with tag {row['tag']} predicted {labels[ranking[0]]} with score {scores[ranking[0]]}")
        if labels[ranking[0]] == "negative":  # predicted negative and it was negative
            author_negative += 1
        else:  # predicted positive/neutral and it was positive
            author_positive += 1
    raw_results_n[author] = author_negative
    raw_results_p[author] = author_positive
    if true_y == 0:
        negative_author_ratios.append(
            author_negative / (author_negative + author_positive)
        )
    else:
        positive_author_ratios.append(
            author_negative / (author_negative + author_positive)
        )


def evaluate_ratio(ratio):
    correct = 0
    for author in data_en["author_id"].unique():
        data_author = data_en[data_en["author_id"] == author]
        true_y = data_author["tag"].iloc[0]
        counter_negative = raw_results_n[author]
        counter_positive = raw_results_p[author]
        if counter_negative > ratio * counter_positive:
            author_y = 1
        else:
            author_y = 0

        if true_y == author_y:
            correct += 1
    return correct / len(data_en["author_id"].unique())


for ratio in np.arange(0, 4.2, 0.2):
    print(f"Accuracy {evaluate_ratio(ratio):.2f} with sensitivity ratio {ratio:.2f}")


df = pd.DataFrame(
    {
        "Negative authors": negative_author_ratios,
        "Positive authors": positive_author_ratios,
    }
)
ax = df.plot.kde()
ax.get_figure().savefig("negative_ratios.png")
