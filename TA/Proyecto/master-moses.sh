cd moses

if [ ! -f "Corpus.tgz" ]; then
    echo "Preparing data"
   sh prepare_data.sh > moses.output 2>&1
fi


echo "Experimentos NGRAMA"
export MERT=5
for NGRAMA in 2 3 4 5; do
    export NGRAMA=$NGRAMA
    echo "Preparing data NGRAMA" "$NGRAMA" "MERT" "$MERT"

    echo "Training LM"
    sh train_lm.sh > moses.output 2>&1

    echo "Training basic model"
    sh train_model_basic.sh > moses.output 2>&1

    echo "Adjusting weights"
    sh train_weights.sh > moses.output 2>&1
done

export NGRAMA=3
echo "Experimentos MERT"
for MERT in 5 7 10; do
    echo "Preparing data NGRAMA" "$NGRAMA" "MERT" "$MERT"

    echo "Training LM"
    sh train_lm.sh > moses.output 2>&1

    echo "Training basic model"
    sh train_model_basic.sh > moses.output 2>&1

    echo "Adjusting weights"
    sh train_weights.sh > moses.output 2>&1
done