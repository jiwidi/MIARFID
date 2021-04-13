import xml.etree.ElementTree as ET
import pandas as pd
import os



root_folder = 'pan21-author-profiling-training-2021-03-14/en/'



data_en = []
for file in os.listdir(root_folder):
    if file[-3:]=="xml":
        root = ET.parse(root_folder+file).getroot()
        assert root[0].tag=="documents"
        for child in root[0]:
            assert child.tag=="document"
            data = [file[:-4], child.text] # [userid, tweettext]
            data_en.append(data)

root_folder = 'pan21-author-profiling-training-2021-03-14/en/'
data_es = []
for file in os.listdir(root_folder):
    if file[-3:]=="xml":
        root = ET.parse(root_folder+file).getroot()
        assert root[0].tag=="documents"
        for child in root[0]:
            assert child.tag=="document"
            data = [file[:-4], child.text] # [userid, tweettext]
            data_es.append(data)


