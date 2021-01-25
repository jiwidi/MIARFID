
# python "$NMT"/nmt-keras/main.py -c config.py

rm trained_model

ln -s \
trained_models/EuTrans_esen_Transformer_model_size_256_ff_size_1024_num_heads_8_encoder_blocks_1_decoder_blocks_1_deepout_linear_Adam_0.001 \
trained_model

python "$NMT"/nmt-keras/sample_ensemble.py --models trained_model/epoch_5 --dataset datasets/Dataset_EuTrans_esen.pkl --text Data/EuTrans/test.es --dest hyp.test.en --changes MODEL_TYPE='Transformer' MODEL_SIZE=64


"$NMT"/nmt-keras/utils/multi-bleu.perl -lc Data/EuTrans/test.en < hyp.test.en