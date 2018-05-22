'''
config file for training encoders 
'''
# python train_nli.py --nlipath dataset/STS_NLI/ \
#                     --datatype sts_nli \
#                     --outputmodelname blstm.all.400.pickle \
#                     --n_epochs 20 \
#                     --encoder_type BLSTMEncoder \
#                     --enc_lstm_dim 700 \
#                     --dpout_model 0.2 \
#                     --dpout_fc 0.2 \
#                     --nonlinear_fc 0 \
#                     --n_enc_layers 1 \
#                     --fc_dim 128 \
#                     --pool_type max \
#                     --gpu_id 2 \
#                     --batch_size 128 \
#                     --test_batch_size 32 \
#                     --test

python find_label.py --nlipath dataset/STS_NLI/ \
                     --datatype sts_nli \
                     --outputmodelname blstm.all.400.pickle \
                     --n_epochs 20 \
                     --encoder_type BLSTMEncoder \
                     --enc_lstm_dim 700 \
                     --dpout_model 0.2 \
                     --dpout_fc 0.2 \
                     --nonlinear_fc 0 \
                     --n_enc_layers 1 \
                     --fc_dim 128 \
                     --pool_type max \
                     --gpu_id 2 \
                     --batch_size 64 \
                     --test_batch_size 32 