echo "Loading env variables"
source ../exports.txt

"Downloading data and extracting"
wget http://www.prhlt.upv.es/~fcn/Students/ta/Corpus.tgz --no-check-certificate

tar zxvf Corpus.tgz

echo "Cleaning corpus"

#TRAINING
##EN
$MOSES/scripts/tokenizer/tokenizer.perl -l en < Corpus/europarl-v7.es-en-train-red.en > Corpus/train.tok.en
$MOSES/scripts/tokenizer/lowercase.perl < Corpus/train.tok.en > Corpus/train.tok.lc.en
##ES
$MOSES/scripts/tokenizer/tokenizer.perl -l en < Corpus/europarl-v7.es-en-train-red.es > Corpus/train.tok.es
$MOSES/scripts/tokenizer/lowercase.perl < Corpus/train.tok.es > Corpus/train.tok.lc.es

clean-corpus-n.perl Corpus/train.tok.lc es en Corpus/training.clean 1 60

#Test
##EN
$MOSES/scripts/tokenizer/tokenizer.perl -l en < Corpus/europarl-v7.es-en-test.en > Corpus/test.tok.en
$MOSES/scripts/tokenizer/lowercase.perl < Corpus/test.tok.en > Corpus/test.tok.lc.en
##ES
$MOSES/scripts/tokenizer/tokenizer.perl -l en < Corpus/europarl-v7.es-en-test.es > Corpus/test.tok.es
$MOSES/scripts/tokenizer/lowercase.perl < Corpus/test.tok.es > Corpus/test.tok.lc.es

clean-corpus-n.perl Corpus/test.tok.lc es en Corpus/test.clean 1 60

mkdir data; cd data
mkdir dataset

mv ../Corpus/training.clean.es dataset/tr-full.src
mv ../Corpus/training.clean.en dataset/tr-full.tgt

mv ../Corpus/test.clean.es dataset/test.src
mv ../Corpus/test.clean.en dataset/test.tgt

cd ..

python3 split_train.py