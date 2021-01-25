echo "Loading env variables"
source ../exports.txt

"Downloading data and extracting"
wget http://www.prhlt.upv.es/~fcn/Students/ta/Corpus.tgz --no-check-certificate

tar zxvf Corpus.tgz

echo "Splitting train data in train/dev 45k/5k"
python3 split_train.py

echo "Cleaning corpus"

mkdir data; cd data
mkdir dataset
clean-corpus-n.perl Corpus/europarl-v7.train es en dataset/training.clean 1 60

mv dataset/training.clean.es dataset/tr.src
mv dataset/training.clean.en dataset/tr-aux.tgt

clean-corpus-n.perl Corpus/europarl-v7.dev es en dataset/dev.clean 1 60

mv dataset/dev.clean.es dataset/dev.src
mv dataset/dev.clean.en dataset/dev.tgt

cp Corpus/europarl-v7.es-en-test.es dataset/test.src
cp Corpus/europarl-v7.es-en-test.en dataset/test.tgt

echo "Tokenizing corpus"
$MOSES/scripts/tokenizer/tokenizer.perl -l en < dataset/tr-aux.tgt > dataset/tr.tgt
# $MOSES/scripts/tokenizer/tokenizer.perl -l en < dataset/dev.clean.es > dataset/dev.clean.tok.src

