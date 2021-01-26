FROM nvcr.io/nvidia/tensorflow:20.12-tf1-py3

##################
## Requirements ##
##################
RUN apt-get update && apt-get install -y git wget && \
rm -rf /var/lib/apt/lists/*
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64

###############
## NMT-Keras ##
###############
RUN git clone https://github.com/PRHLT/nmt-keras_practicas-TA.git /opt/nmt-keras \
&& cp /opt/nmt-keras/docker/config.py /opt/nmt-keras/config.py
WORKDIR /opt/nmt-keras
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
bash /tmp/miniconda.sh -b -p /opt/miniconda && export PATH="/opt/miniconda/bin:$PATH" && \
conda config --set always_yes yes --set changeps1 no && conda init && \
conda install --file /opt/nmt-keras/req-travis-conda.txt && conda install mkl mkl-service pip && \
pip install -r /opt/nmt-keras/req-travis-pip.txt && \
pip install nvidia-pyindex && \
#pip install astor && pip install opt-einsum gast==0.2.2 wrapt tensorflow-estimator==1.15.1 && \
#pip install -i https://pypi.ngc.nvidia.com nvidia-tensorflow && \
pip install https://developer.download.nvidia.com/compute/redist/nvidia-tensorflow/nvidia_tensorflow-1.15.4%2Bnv20.12-cp38-cp38-linux_x86_64.whl && \

export PATH=$PATH:$HOME/.local/bin
	#pip install https://developer.download.nvidia.com/compute/redist/nvidia-tensorflow/nvidia_tensorflow-1.15.4%2Bnv20.12-cp38-cp38-linux_x86_64.whl
ENV PYTHONPATH="PYTHONPATH=${PYTHONPATH}:/opt/nmt-keras/keras:\
/opt/nmt-keras/coco-caption:/opt/nmt-keras/multimodal_keras_wrapper"

##########################
## Default: train model ##
##########################
CMD export PATH="/opt/miniconda/bin:$PATH" && \
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 && \
python main.py
