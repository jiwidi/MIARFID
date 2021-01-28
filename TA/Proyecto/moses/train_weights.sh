docker container run -it --rm -v ${PWD}/data/:/data moses \
/opt/moses/scripts/training/mert-moses.pl \
/data/dataset/dev.src /data/dataset/dev.tgt \
/opt/moses/bin/moses /data/alignment/model/moses-$NGRAMA.ini \
-threads=12 \
--maximum-iterations=$MERT \
--working-dir /data/mert \
--mertdir /opt/moses/bin/ \
--decoder-flags "-threads 12"

mv data/mert/moses.ini data/mert/moses-lm$NGRAMA-mert$MERT.ini