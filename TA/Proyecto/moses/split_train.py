import pandas as pd 


print("Procesando train EN")
train_en = open('Corpus/europarl-v7.train.en', 'w')
dev_en = open('Corpus/europarl-v7.dev.en', 'w')

i=0
with open('Corpus/europarl-v7.es-en-train-red.en') as f:
    for line in f.readlines():
        if i<45000:
            train_en.write(line)
        else:
            dev_en.write(line)
        
        i+=1

train_en.close()
dev_en.close()


print("Procesando train ES")
dev_es = open('Corpus/europarl-v7.dev.es', 'w')
train_es = open('Corpus/europarl-v7.train.es', 'w')
i=0
with open('Corpus/europarl-v7.es-en-train-red.es') as f:
    for line in f.readlines():
        if i<45000:
            train_es.write(line)
        else:
            dev_es.write(line)
        i+=1

train_es.close()
dev_es.close()

        