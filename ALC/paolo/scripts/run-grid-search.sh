for vec in {0..1}
do
    for model in {0..8}
    do
        python grid_search.py --lan en --vec ${vec} --model ${model}
    done
done

for vec in {0..1}
do
    for model in {0..8}
    do
        python grid_search.py --lan es --vec ${vec} --model ${model}
    done
done