cd moses

if [ ! -f "Corpus.tgz" ]; then
    echo "Preparing data"
   sh prepare_data.sh > moses.output 2>&1
fi

export SUAVIZADO=kndiscount
echo "Experimentos NGRAMA"
export MERT=5
for NGRAMA in 2 3 4 5 6; do
    export NGRAMA=$NGRAMA
    echo "Preparing data NGRAMA" "$NGRAMA" "MERT" "$MERT"

    echo "Training LM"
    sh train_lm.sh > moses.output 2>&1

    echo "Training basic model"
    sh train_model_basic.sh > moses.output 2>&1

    echo "Adjusting weights"
    sh train_weights.sh > moses.output 2>&1

    echo "Predict"
    sh generate_predictions.sh > moses.output 2>&1

    echo "Evaluate"
    echo "Experiment NGRAMA" "$NGRAMA" "MERT" "$MERT" >> resultados.txt
    sh evaluate.sh >> resultados.txt
done

export NGRAMA=3
echo "Experimentos MERT"
for MERT in 5 7 10; do
    export MERT=$MERT
    echo "Preparing data NGRAMA" "$NGRAMA" "MERT" "$MERT"

    echo "Training LM"
    sh train_lm.sh > moses.output 2>&1

    echo "Training basic model"
    sh train_model_basic.sh > moses.output 2>&1

    echo "Adjusting weights"
    sh train_weights.sh > moses.output 2>&1

    echo "Predict"
    sh generate_predictions.sh > moses.output 2>&1

    echo "Evaluate"
    echo "Experiment NGRAMA" "$NGRAMA" "MERT" "$MERT" >> resultados.txt
    sh evaluate.sh >> resultados.txt
done






# export NGRAMA=3
# export MERT=10


# echo "Preparing data NGRAMA" "$NGRAMA" "MERT" "$MERT " "$SUAVIZADO"

# #Wb
# echo "Training LM"
# docker container run -it --rm -v ${PWD}/data/:/data moses \
# /opt/srilm/lm/bin/i686-m64/ngram-count -order $NGRAMA -unk -interpolate -wbdiscount \
# -text /data/dataset/tr.tgt -lm /data/model-$NGRAMA.lm

# echo "Training basic model"
# sh train_model_basic.sh > moses.output 2>&1

# echo "Adjusting weights"
# sh train_weights.sh > moses.output 2>&1

# echo "Predict"
# sh generate_predictions.sh > moses.output 2>&1

# echo "Evaluate"
# echo "Experiment NGRAMA" "$NGRAMA" "MERT" "$MERT" "WB" >> resultados.txt
# sh evaluate.sh >> resultados.txt

# #KN
# echo "Training LM"
# docker container run -it --rm -v ${PWD}/data/:/data moses \
# /opt/srilm/lm/bin/i686-m64/ngram-count -order $NGRAMA -unk -interpolate -kndiscount \
# -text /data/dataset/tr.tgt -lm /data/model-$NGRAMA.lm

# echo "Training basic model"
# sh train_model_basic.sh > moses.output 2>&1

# echo "Adjusting weights"
# sh train_weights.sh > moses.output 2>&1

# echo "Predict"
# sh generate_predictions.sh > moses.output 2>&1

# echo "Evaluate"
# echo "Experiment NGRAMA" "$NGRAMA" "MERT" "$MERT" "KN" >> resultados.txt
# sh evaluate.sh >> resultados.txt

# #Good turing
# export SUAVIZADO=''
# echo "Preparing data NGRAMA" "$NGRAMA" "MERT" "$MERT " "$SUAVIZADO"

# echo "Training LM"
# docker container run -it --rm -v ${PWD}/data/:/data moses \
# /opt/srilm/lm/bin/i686-m64/ngram-count -order $NGRAMA -unk -interpolate \
# -text /data/dataset/tr.tgt -lm /data/model-$NGRAMA.lm

# echo "Training basic model"
# sh train_model_basic.sh > moses.output 2>&1

# echo "Adjusting weights"
# sh train_weights.sh > moses.output 2>&1

# echo "Predict"
# sh generate_predictions.sh > moses.output 2>&1

# echo "Evaluate"
# echo "Experiment NGRAMA" "$NGRAMA" "MERT" "$MERT" "GOOD TURING" >> resultados.txt
# sh evaluate.sh >> resultados.txt






