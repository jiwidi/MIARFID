cd train; mkdir lm

ngram-count -order $NGRAMA -unk -interpolate -kndiscount -text training.clean.en -lm lm/europarl-3.lm

export LM=$PWD/lm/europarl-$NGRAMA.lm


cd ../