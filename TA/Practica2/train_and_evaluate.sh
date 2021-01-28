python "$NMT"/nmt-keras/main.py -c config.py

rm trained_model

ln -s trained_models/EuTrans_esen_AttentionRNNEncoderDecoder_\
src_emb_128_bidir_True_enc_LSTM_64_dec_ConditionalLSTM_64_deepout_\
linear_trg_emb_128_Adagrad_0.001 trained_model

python "$NMT"/nmt-keras/sample_ensemble.py \
--models trained_model/epoch_5 \
--dataset datasets/Dataset_EuTrans_esen.pkl \
--text Data/EuTrans/test.es \
--dest hyp.test.en

"$NMT"/nmt-keras/utils/multi-bleu.perl -lc Data/EuTrans/test.en < hyp.test.en