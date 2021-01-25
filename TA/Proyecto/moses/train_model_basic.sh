docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/train-model.perl -mgiza -mgiza-cpus 10 --first-step 1 \
-root-dir /data/alignment -corpus /data/dataset/tr -f src -e tgt \
-alignment grow-diag-final-and -reordering msd-bidirectional-fe -lm 0:5:/data/model-$NGRAMA.lm \
-external-bin-dir /opt/moses/mgiza/mgizapp/bin/

mv data/alignment/model/moses.ini data/alignment/model/moses-$NGRAMA.ini