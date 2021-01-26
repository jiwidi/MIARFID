for file in *.src; do
    mv "$file" "$(basename "$file" .src).aux"
done

for file in *.tgt; do
    mv "$file" "$(basename "$file" .tgt).src"
done

for file in *.aux; do
    mv "$file" "$(basename "$file" .aux).tgt"
done