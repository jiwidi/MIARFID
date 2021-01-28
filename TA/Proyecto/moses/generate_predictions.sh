docker container run -i --rm -v ${PWD}/data/:/data moses \
/opt/moses/bin/moses -threads 10 -f /data/mert/moses-lm$NGRAMA-mert$MERT.ini < data/dataset/test.src \
> data/test.hyp