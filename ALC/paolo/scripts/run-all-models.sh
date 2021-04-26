for vec in {0..1}
do
    for model in {0..8}
    do
        python train_models.py --lan en --vec ${vec} --model ${model}
    done
done

for vec in {0..1}
do
    for model in {0..8}
    do
        python train_models.py --lan es --vec ${vec} --model ${model}
    done
done