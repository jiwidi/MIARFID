mkdir Data; mkdir Data/EuTrans

cp ../moses/Corpus/europarl-v7.train.en Data/EuTrans/training.en
cp ../moses/Corpus/europarl-v7.train.es Data/EuTrans/training.es

cp ../moses/Corpus/europarl-v7.dev.en Data/EuTrans/development.en
cp ../moses/Corpus/europarl-v7.dev.es Data/EuTrans/development.es

cp ../moses/Corpus/europarl-v7.es-en-test.en Data/EuTrans/test.en
cp ../moses/Corpus/europarl-v7.es-en-test.es Data/EuTrans/test.es

# cp ../moses/train/training.e? Data/EuTrans/

cp "$NMT"/nmt-keras/config.py .