
# ngram-count -order $NGRAMA -unk -interpolate -kndiscount -text training.clean.en -lm lm/europarl-$NGRAMA.lm


docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/srilm/lm/bin/i686-m64/ngram-count -order $NGRAMA -unk -interpolate -$SUAVIZADO \
-text /data/dataset/tr.tgt -lm /data/model-$NGRAMA.lm

# export LM=$PWD/lm/europarl-$NGRAMA.lm