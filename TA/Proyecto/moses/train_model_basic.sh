export CPU=1

cd train

$SCRIPTS_ROOTDIR/training/train-model.perl -root-dir work \
-mgiza -mgiza-cpus $CPU \
-corpus training.clean -f es -e en \
-alignment grow-diag-final-and -reordering msd-bidirectional-fe \
-lm 0:3:$LM -external-bin-dir $GIZA

cd ..