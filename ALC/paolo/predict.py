from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import pandas as pd
from tqdm import tqdm

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []


    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)

#Tag 1 is negative, 0 positive
correct = 0
data_en = pd.read_csv("dataset/data_en.csv")

for author in tqdm(data_en["author_id"].unique()):
    data_author = data_en[data_en["author_id"]==author]
    true_y = data_author["tag"].iloc[0]
    author_negative = 0
    author_positive = 0

    for idx,row in tqdm(data_author.iterrows(), total = len(data_author)):
        text = row["tweet"]
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        # print(f"For tweet {text} with tag {row['tag']} predicted {labels[ranking[0]]} with score {scores[ranking[0]]}")
        if(labels[ranking[0]]=="negative"):#predicted negative and it was negative
            author_negative+=1
        elif(labels[ranking[0]]!="negative"): #predicted positive/neutral and it was positive
            author_positive+=1


    if(author_negative>0.8*author_positive):
        author_y = 1
    else:
        author_y = 0

    if(true_y==author_y):
        correct+=1

print(correct/len(data_en["author_id"].unique()))