TRAIN="data/dihana.entrenamiento1.txt"
TEST="data/dihana.prueba1.txt"

mkdir ej2

for N in {3..4}
do
  echo "Good-Turing $N"

  ngram-count -order $N -lm ej2/modelo$N -text $TRAIN

  ngram -order $N -lm ej2/modelo$N -ppl $TEST

  echo Witten-Bell $N

  ngram-count -order $N -lm ej2/modeloWB$N -wbdiscount -text $TRAIN

  ngram -order $N -lm ej2/modeloWB$N -ppl $TEST

  echo Modified Kneser-Ney $N

  ngram-count -order $N -lm ej2/modeloMKN$N -kndiscount -text $TRAIN

  ngram -order $N -lm ej2/modeloMKN$N -ppl $TEST

  echo Unmodified Kneser-Ney $N

  ngram-count -order $N -lm ej2/modeloKN$N -ukndiscount -text $TRAIN

  ngram -order $N -lm ej2/modeloKN$N -ppl $TEST

done
