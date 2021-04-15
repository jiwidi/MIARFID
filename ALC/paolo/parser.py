import xml.etree.ElementTree as ET
import pandas as pd
import os



root_folder = 'dataset/pan21-author-profiling-training-2021-03-14/en/'


truths={}
with open(root_folder+"truth.txt") as f:
    for line in f:
        author, tag = line.split(":::")
        truths[author]=tag

data_en = []
for file in os.listdir(root_folder):
    if file[-3:]=="xml":
        root = ET.parse(root_folder+file).getroot()
        assert root[0].tag=="documents"
        for child in root[0]:
            assert child.tag=="document"
            data = [file[:-4], child.text, tag] # [userid, tweettext,tag ]
            data_en.append(data)

root_folder = 'dataset/pan21-author-profiling-training-2021-03-14/en/'
data_es = []
for file in os.listdir(root_folder):
    if file[-3:]=="xml":
        root = ET.parse(root_folder+file).getroot()
        assert root[0].tag=="documents"
        for child in root[0]:
            assert child.tag=="document"
            data = [file[:-4], child.text] # [userid, tweettext]
            data_es.append(data)


data_en = pd.DataFrame(data_en, columns=['author_id','tweet','tag'])
data_en.to_csv("dataset/data_en.csv",index=False)
print(data_en)


