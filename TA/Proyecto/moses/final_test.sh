#FINAL TEST FILE
export MERT=10
export NGRAMA=3
echo "Preparing data NGRAMA" "$NGRAMA" "MERT" "$MERT"

echo "Training LM"
docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/srilm/lm/bin/i686-m64/ngram-count -order $NGRAMA -unk -interpolate -kndiscount \
-text /data/final/tr.tgt -lm /data/model-$NGRAMA.lm > moses.output 2>&1

echo "Training basic model"
docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/train-model.perl -root-dir /data/alignment \
-mgiza -mgiza-cpus 12 -cores 12 \
-corpus /data/final/tr -f src -e tgt \
-alignment grow-diag-final-and -reordering msd-bidirectional-fe \
-lm 0:$NGRAMA:/data/model-$NGRAMA.lm \
-external-bin-dir /opt/moses/mgiza/mgizapp/bin/ > moses.output 2>&1

mv data/alignment/model/moses.ini data/alignment/model/moses-$NGRAMA.ini

echo "Adjusting weights"
docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/mert-moses.pl \
/data/final/dev.src /data/final/dev.tgt \
/opt/moses/bin/moses /data/alignment/model/moses-$NGRAMA.ini \
-threads=12 \
--maximum-iterations=$MERT \
--working-dir /data/mert \
--mertdir /opt/moses/bin/ \
--decoder-flags "-threads 12" > moses.output 2>&1

mv data/mert/moses.ini data/mert/moses-lm$NGRAMA-mert$MERT.ini

echo "Predict"
docker container run -i --rm -v ${PWD}/data/:/data moses \
/opt/moses/bin/moses -threads 10 -f /data/mert/moses-lm$NGRAMA-mert$MERT.ini < /mnt/kingston/github/MIARFID/TA/Proyecto/moses/data/final/europarl-v7.es-en-test-hidden.en \
> /mnt/kingston/github/MIARFID/TA/Proyecto/nmt-keras/trained_models/JAIMEFERRANDO_MOSES.es

docker container run -i --rm -v ${PWD}/data/:/data moses \
/opt/moses/bin/moses -threads 10 -f /data/mert/moses-lm$NGRAMA-mert$MERT.ini < /mnt/kingston/github/MIARFID/TA/Proyecto/moses/data/final/test.src \
> /mnt/kingston/github/MIARFID/TA/Proyecto/nmt-keras/trained_models/data.hyp

docker container run -i --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/generic/multi-bleu.perl -lc data/final/test.tgt < /mnt/kingston/github/MIARFID/TA/Proyecto/nmt-keras/trained_models/data.hyp | tail -n 1

# python sample_ensemble.py --models trained_model/epoch_5 --dataset datasets/Dataset_Europarl_enes.pkl --text europarl-v7.es-en-test-hidden.en --dest /data/trained_models/JAIMEFERRANDO.es