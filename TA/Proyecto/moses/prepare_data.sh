echo "Loading env variables"
source ../exports.txt

"Downloading data and extracting"
wget http://www.prhlt.upv.es/~fcn/Students/ta/Corpus.tgz --no-check-certificate

tar zxvf Corpus.tgz 

echo "Splitting train data in train/dev 45k/5k"
python3 split_train.py

echo "Cleaning corpus"
mkdir train
clean-corpus-n.perl Corpus/europarl-v7.train es en train/training.clean 1 60

mkdir dev
clean-corpus-n.perl Corpus/europarl-v7.dev es en dev/dev.clean 1 60

mkdir test
cp Corpus/europarl-v7.es-en-test.en test/test.en
cp Corpus/europarl-v7.es-en-test.es test/test.es 

echo "Tokenizing corpus"
$MOSES/scripts/tokenizer/tokenizer.perl -l en < train/training.clean.en > train/training.clean.tok.en
# $MOSES/scripts/tokenizer/tokenizer.perl -l en < dev/dev.clean.es > dev/dev.clean.tok.es

