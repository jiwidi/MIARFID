import os
import xml.etree.ElementTree as ET

import pandas as pd

#root_folder = "dataset/pan21-author-profiling-test-without-gold-for-participants/en/"
#
#print("Processing en")
#data_en = []
#for file in os.listdir(root_folder):
#    print(file)
#    if file[-3:] == "xml":
#        root = ET.parse(root_folder + file).getroot()
#        assert root[0].tag == "documents"
#        for child in root[0]:
#            assert child.tag == "document"
#            data = [
#                file[:-4],
#                child.text
#            ]  # [userid, tweettext]
#            data_en.append(data)

print('Processing es')
root_folder = "dataset/pan21-author-profiling-test-without-gold-for-participants/es/"

data_es = []
i = 0
for file in os.listdir(root_folder):
    print(file, i)
    i += 1
    if file[-3:] == "xml":
        root = ET.parse(root_folder + file).getroot()
        assert root[0].tag == "documents"
        for child in root[0]:
            assert child.tag == "document"
            data = [
                file[:-4],
                child.text
            ]  # [userid, tweettext]
            data_es.append(data)


print("Saving to files")
#data_en = pd.DataFrame(data_en, columns=["author_id", "tweet"])
#data_en.to_csv("dataset/data_en_test.csv", index=False)
#
data_es = pd.DataFrame(data_es, columns=["author_id", "tweet"])
data_es.to_csv("dataset/data_es_test.csv", index=False)
#
#print("Done")