TRAIN="data/dihana.entrenamiento1.txt"
TEST="data/dihana.prueba1.txt"
mkdir ej1
for N in {1..5}
do
  ngram-count -order $N -lm ej1/modelo$N -text $TRAIN
  echo "Evaluando ej1/modelo$N"
  ngram -order $N -lm ej1/modelo$N -ppl $TEST
done
