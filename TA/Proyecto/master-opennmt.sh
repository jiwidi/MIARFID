cd opennmt

echo "Building vocab"
onmt_build_vocab -config opennmt.yaml -n_sample 10000

echo "Training"
onmt_train -config opennmt.yaml

echo "Predicting"
onmt_translate -model toy-ende/run/model_step_1000.pt -src ../nmt-keras/Data/Europarl/test.en -output toy-ende/pred_1000.txt -gpu 0 -verbose

echo "Eval"
perl multi-bleu.perl ../nmt-keras/Data/Europarl/test.es < toy-ende/pred_10000.txtpas

