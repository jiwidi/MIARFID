TRAIN="data/dihana.entrenamiento1.txt"
TEST="data/dihana.prueba1.txt"

mkdir ej3

for N in {3..4}
do
  
  echo Witten-Bell $N non interpolated

  ngram-count -order $N -lm ej3/modeloWB$N -wbdiscount -text $TRAIN

  ngram -order $N -lm ej3/modeloWB$N -ppl $TEST

  echo Modified Kneser-Ney $N non interpolated

  ngram-count -order $N -lm ej3/modeloMKN$N -kndiscount -text $TRAIN

  ngram -order $N -lm ej3/modeloMKN$N -ppl $TEST

done

for N in {3..4}
do
  
  echo Witten-Bell $N interpolated

  ngram-count -order $N -lm ej3/modeloWB$N -wbdiscount -interpolate -text $TRAIN

  ngram -order $N -lm ej3/modeloWB$N -ppl $TEST

  echo Modified Kneser-Ney $N interpolated

  ngram-count -order $N -lm ej3/modeloMKN$N -kndiscount -interpolate -text $TRAIN

  ngram -order $N -lm ej3/modeloMKN$N -ppl $TEST

done
