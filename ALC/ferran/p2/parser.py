
from xml.dom import minidom
import pandas as pd

files = ["data/TASS2017_T1_test.xml","data/TASS2017_T1_training.xml","data/TASS2017_T1_development.xml" ]


for file in files:
    xmldoc = minidom.parse(file)
    tweets = xmldoc.getElementsByTagName('tweet')
    # outfile = open('training.txt', 'w')
    ids=[]
    texts=[]
    target=[]
    for t in tweets:
        for node in t.childNodes:
            # print (t.childNodes)
            if node.nodeName == 'tweetid':
                ids.append(node.firstChild.data)
            elif node.nodeName == 'content':
                texts.append(node.firstChild.data.replace("\n", " "))

            if(file!="data/TASS2017_T1_test.xml"):
                if node.nodeName == 'sentiment':
                    for h in node.childNodes:
                        if h.nodeName == 'polarity':
                            if h.firstChild.nodeName == 'value':
                                target.append(h.firstChild.firstChild.data)
            else:
                target.append(None)
    df = pd.DataFrame(list(zip(ids, texts,target)),
               columns =['id', 'text', 'target'])
    df.to_csv(file.replace(".xml","_parsed.csv"),index=False)