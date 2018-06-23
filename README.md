# Bi-directional Entailment Estimation (BDEE)
### Overview
-----------------------
We aim to build a system for Machine Translation Evaluationn using Bi-directinal Entailment. We first build an Entailment system by training on NLI corpus like SNLI and MNLI. Post training we use the Entailment system to evaluate the candidate traslation with respect to reference translation in WMT14 dataset for metric task. We provide the code used for training Entailment system and evaluating on WMT14 dataset.

### Code Setup
-----------------------
$ bash setup.sh

The above script performs following:
1. Calls get_data.bash to download the required datasets like Glove word vector file, SNLI and MNLI dataset. SNLI and MNLI are further preprocessed to separate the train, dev and test set into 
	1. s1.train, s2.train, labels.train
	2. s1.dev, s2.dev, labels.dev
	3. s1.test, s2.test, labels.test

2. Creates the required directory structure required during training and later during evaluation step. The directories created and their significance is as below - 
	1. dataset/ - All the dataset like SNLI, MNLI, COMB, SICK after preprocessing are kept in this folder. This is required by our training code.
	2. encoderdir/ - This containes the pickle file of the encoder of the trained model. The format for saving '[model_name].[dataset].]pkl', where the model name is the name that we give during training.  
	3. glove_vocab/ - This keeps the saved glove vectors for different datasets used during training and evaluation in pickle format. A dictionary for all the words found in a dataset(SNLI, MNLI, WMT14) is formed and the corresponding glove vector for all the words are loaded when we run any model on the dataset for the first time. These word vectors are then saved in this directory with the convention '[dataset_name].pkl'. On successive run the word vector are loaded from these pickle files thus saving loading time of glove vectors.  
	4. logs/ - This directory stores the '[model_name].desc' files during evaluation. The .desc files contains the pair of sentences in WMT14 dataset and the corresponding forward, backward Entailment scores and cosine scores.   
	5. savedir/ - It contains '[model_name].[dataset].]pkl' files. These files stores the entire trained models i.e. encoder along with classifier and softmax layer, in pickle format.
	6. score/ - It contains the two files system and segment scores generated for during evalation for each Entailment model. 
	7. tuned_score/ - It contains the two files system and segment scores generated for during evalation for each Entailment model post tuning.
	8. testlogdir/ - It contains the predicted labels on the test set of Entailment dataset(SNLI/MNLI) after training is completed.


3. The SNLI and MNLI dataset are combined into COMB dataset.

4. Preproces the SICK dataset.


### Training the model
-----------------------
$ bash train.sh

train.sh calls the main python script for training with all required arguments. For training an Entailment model open this file and provide the required parameters and then run the script.

The hyperparameters that can be provided are-
	--nlipath dataset/SNLI/ \
	--datatype snli \
	--outputmodelname lstm.100.pickle \
	--n_epochs 1 \
	--encoder_type LSTMEncoder \
	--enc_lstm_dim 100 \
	--dpout_model 0.2 \
	--dpout_fc 0.2 \
	--nonlinear_fc 0 \
	--n_enc_layers 1 \
	--fc_dim 128 \
	--pool_type max \
	--gpu_id 1 \
	--batch_size 64 \
	--test_batch_size 32 



 
### Evaluating the model
------------------------
$ bash test_wmt.sh [model_name].pickle [encoder_dim] [own_name] [gpu_id]

example: bash test_wmt.sh lstm.100.pickle 100 lstm.100 1



### Tuning the system
-----------------------
$ bash tuning.sh [model_name]

example: bash tuning.sh lstm.100
