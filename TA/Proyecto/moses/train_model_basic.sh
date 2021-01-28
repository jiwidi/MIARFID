docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/train-model.perl -root-dir /data/alignment \
-mgiza -mgiza-cpus 12 -cores 12 \
-corpus /data/dataset/tr -f src -e tgt \
-alignment grow-diag-final-and -reordering msd-bidirectional-fe \
-lm 0:$NGRAMA:/data/model-$NGRAMA.lm \
-external-bin-dir /opt/moses/mgiza/mgizapp/bin/

mv data/alignment/model/moses.ini data/alignment/model/moses-$NGRAMA.ini