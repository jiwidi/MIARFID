cd nmt-keras/

sh prepare_data.sh

# nvidia-docker container run -it --rm -v \
# -v $PWD:/data \
# nmt-keras-ta
# # -v $ruta/config.py:/opt/nmt-keras/config.py \
# nmt-keras-ta

# nvidia-docker container run -it --rm \
# -v $PWD:/data \
# -v $PWD/config.py:/opt/nmt-keras/config.py \
# nmt-keras-ta /bin/bash
rm -rf trained_models/

nvidia-docker container run -it --rm \
    -v $PWD:/data \
    -v $PWD/config.py:/opt/nmt-keras/config.py \
    nmt-keras-ta /opt/miniconda/bin/python main.py



