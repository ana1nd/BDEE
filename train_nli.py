# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import time
import argparse
import time

import numpy as np

import torch
import torch as th
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import NLINet

start_time = time.time()
GLOVE_PATH = "glove_dir/glove.840B.300d.txt"


parser = argparse.ArgumentParser(description='NLI training')

# paths
parser.add_argument("--nlipath", type=str, default='dataset/MultiNLI_original/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--datatype", type=str, default='snli', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--encoderdir",type=str,default='encoderdir/',help="Directory to save encoder")
parser.add_argument("--testlogdir",type=str,default='testlogdir/',help="Directory to save encoder")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--encodermodelname",type=str,default='encoder.pickle',help="Directory to save encoder")
parser.add_argument("--logdir",type=str,default='logs/',help="Directory to save encoder")

#for WMT
parser.add_argument("--refpath",type=str,default='../data/wmt_data/references/newstest2014-ref.cs-en',help="Directory to save encoder")
parser.add_argument("--hyppath",type=str,default='../data/wmt_data/system-outputs/newstest2014/cs-en/newstest2014.cu-moses.3383.cs-en',help="Directory to save encoder")
parser.add_argument("--lp",type=str,default='cs-en',help="Directory to save encoder")


# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--test_batch_size", type=int, default=16)

parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-7, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--test", action='store_true', help="Call if you want to only test on the best checkpoint.")

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=100, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=2, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=1, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")


params, _ = parser.parse_known_args()

model_name = params.outputmodelname
model_name = model_name[:model_name.rfind('.')]


print ('Available devices ', th.cuda.device_count())
print ('Current cuda device ', th.cuda.current_device())
th.cuda.set_device(params.gpu_id)
print ('Current cuda device ', th.cuda.current_device())
print (' Yes/No ', th.cuda.is_available())

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
train, valid, test = get_nli(params.nlipath)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], GLOVE_PATH,params.datatype)

temp_s1 = test['s1'] #dont delete this line. It comes handy in just testing
temp_s2 = test['s2']
temp_label = test['label']

for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])

params.word_emb_dim = 300
#sys.exit()

"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}

# model
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder','GRUEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
nli_net = NLINet(config_nli_model)
print(nli_net)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

# cuda by default
#torch.backends.cudnn.enabled=False
nli_net.cuda()
loss_fn.cuda()


"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]


    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size

        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data[0])
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    entail_forward = []
    entail_backward = []
    #print len(s1)
    for i in range(0, len(s1), params.test_batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.test_batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.test_batch_size], word_vec)
        #print i
        s1_batch, s2_batch = Variable(s1_batch.cuda(),volatile=True), Variable(s2_batch.cuda(),volatile=True)
        #print s1_len
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.test_batch_size])).cuda()

        # model forward
        output_forward = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        output_backward = nli_net((s2_batch, s2_len), (s1_batch, s1_len))

        soft = torch.nn.Softmax()
        softmax_output_forward = soft(output_forward)
        softmax_output_backward = soft(output_backward)

        sz_forward = softmax_output_forward.size()[0]
        sz_backward = softmax_output_backward.size()[0]
        
        
        for j in range (0,sz_forward,1):
            entail_forward.append(softmax_output_forward[j][0])
            entail_backward.append(softmax_output_backward[j][0])

        pred = output_forward.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net, os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc,entail_forward,entail_backward


if os.path.isfile(os.path.join(params.outputdir,params.outputmodelname)):
    print "Model is already present, using model =",params.outputmodelname
    nli_net = torch.load(os.path.join(params.outputdir, params.outputmodelname))
"""
Train model on Natural Language Inference task
"""
epoch = 1

dotest = params.test
print "To Test? =", dotest
if dotest == False:
    while not stop_training and epoch <= params.n_epochs:
        train_acc = trainepoch(epoch)
        print "Saving Model"
        torch.save(nli_net, os.path.join(params.outputdir,params.outputmodelname))
        print "Saved Model"
        eval_acc = evaluate(epoch, 'valid')
        epoch += 1

print("Done Training")
print("Evaluate Now")
# Run best model on test set.
del nli_net
nli_net = torch.load(os.path.join(params.outputdir, params.outputmodelname))

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(epoch, 'valid')
#evaluate(1e6, 'valid', True)
#eval_acc,entail_forward = evaluate(0,'test',True)
print "Writing in testlog file for this model on Test file"
eval_acc,entail_forward,entail_backward = evaluate(0, 'test', True)

test_log_file = os.path.join(params.testlogdir) + model_name + ".test.log"

f1 = open(test_log_file,'w+')
sz = len(entail_forward)

for i in range(0,sz,1):
    e1_score = entail_forward[i].data.cpu().numpy()[0]
    e2_score = entail_backward[i].data.cpu().numpy()[0]
    #e2_score = e1_score
    
    if temp_label[i] == 0:
        label = "entailment"
    else:
        label = "non-entailment"
    
    curr_line = [str(e1_score),str(e2_score),label,temp_s1[i],temp_s2[i]]
    f1.write('\t'.join(curr_line[0:]) + '\n')
    
print "Done: Writing in testlog file for this model on Test file"
print "****************************************************"

# Save encoder instead of full model
print("Save Encoder")
torch.save(nli_net.encoder,os.path.join(params.encoderdir, params.outputmodelname))
#torch.save(nli_net.encoder,os.path.join(params.outputdir, "anand"))
print("--%s --" % params.outputmodelname)
print("--- %s seconds ---" % (time.time() - start_time))
