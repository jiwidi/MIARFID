$MOSES/scripts/training/mert-moses.pl \
dev/dev.clean.es dev/dev.clean.en \
$MOSES/bin/moses train/work/model/moses.ini \
--maximum-iterations=$MERT \
--mertdir $MOSES/bin/

mv mert-work/moses.ini mert-work/moses.europarl.ini