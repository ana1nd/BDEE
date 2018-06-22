# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
import pickle
import unicodedata
import re

def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths


def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_glove_wmt(word_dict, glove_path,glove_vocab,file_name):
    # create word_vec with glove vectors
    word_vec = {}
    path = os.path.join(glove_vocab,file_name) + '.pkl'
    print os.path.isfile(path),path
    if os.path.isfile(path):
        print " File already present"
        word_vec = load_obj(path)
    else:
        print "GloVe file not present: make & save"
        with open(glove_path) as f:
            lineN = 0
            for line in f:
                word, vec = line.split(' ',1)
                if lineN % 100000 == 0 :
                    print lineN,word
                lineN += 1
                vec = vec.split(' ')
                vec = [float(i) for i in vec]
                vec = np.array(vec)
                if word in word_dict:
                    word_vec[word] = vec #np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with glove vectors'.format(len(word_vec), len(word_dict)))
        save_obj(word_vec,path)
        print "Saved the embeddings of wmt"
    return word_vec 

def get_glove(word_dict, glove_path,datatype):
    # create word_vec with glove vectors
    word_vec = {}
    file_name = 'word_vec_' + datatype + '.pkl'
    print os.path.isfile(file_name)
    if os.path.isfile(file_name):
        print file_name," File already present"
        word_vec = load_obj(file_name)
    else:
        print " GloVe file not present: make & save"
        with open(glove_path) as f:
            lineN = 0
            for line in f:
                #word, vec = line.split(' ', 1)
                word, vec = line.split(' ',1)
                if lineN % 100000 == 0 :
                    print lineN,word
                lineN += 1
                vec = vec.split(' ')
                vec = [float(i) for i in vec]
                vec = np.array(vec)
                if word in word_dict:
                    word_vec[word] = vec #np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))
        save_obj(word_vec,file_name)
    return word_vec

def build_vocab(sentences, glove_path,datatype):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path,datatype)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec

def build_vocab_wmt(sentences, glove_path,glove_vocab,glove_vocab_filename):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove_wmt(word_dict, glove_path,glove_vocab,glove_vocab_filename)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def get_wmt(ref_path,hyp_path):
    s1 = {}
    s2 = {}
    first, second = list(), list()
    for data_type in ['test']:
        s1[data_type], s2[data_type] = {}, {}
        s1[data_type]['path'] = os.path.join(ref_path)
        s2[data_type]['path'] = os.path.join(hyp_path)

        s1[data_type]['sent'] = [normalizeString(line.rstrip()) for line in open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [normalizeString(line.rstrip()) for line in open(s2[data_type]['path'], 'r')]

        first = [normalizeString(line.rstrip()) for line in open(s1[data_type]['path'], 'r')]
        second = [normalizeString(line.rstrip()) for line in open(s2[data_type]['path'], 'r')]

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) 
        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(data_type.upper(), len(s1[data_type]['sent']), data_type))

    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent']}
    return test, first, second

def unicodeToAscii(s):

    temp = []
    for s_ in s.split():
        temp.append(unicodedata.normalize('NFKD',unicode(s_,'utf8')).encode('ascii','ignore'))
    s = ' '.join(temp)
    return s


def normalizeString(s):

    s = unicodeToAscii(s.lower().strip())
    s = unicodeToAscii(s.lower().strip())

    s = re.sub(r" &apos;t ", r"t ", s)
    s = re.sub(r" &apos;m ", r" am ", s)
    s = re.sub(r" &apos;ve ", r" have ", s)
    s = re.sub(r" &apos;re ", r" are ", s)

    s = re.sub(r" &apos;ll ", r" will ", s)    
    s = re.sub(r" &apos;", r" ", s)
    #s = re.sub(r" &apos;", r"", s)
    s = re.sub(r" &quot; ", r" ", s)
    s = re.sub(r"&quot; ", r"", s)

    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)

    return s



# def normalizeString22(s):
    
#     s = unicodeToAscii(s.lower().strip())
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    
#     stopwords = ['apos','quot']
#     querywords = s.split()

#     resultwords  = [word for word in querywords if word.lower() not in stopwords]
#     s = ' '.join(resultwords)    
    
#     return s

def get_sts(hyp_path):
    s1 = {}
    s2 = {}
    
    first, second, gs = list(), list(), list()
    for data_type in ['test']:
        s1[data_type], s2[data_type] = {}, {}
        s1[data_type]['path'] = os.path.join(hyp_path)
        s2[data_type]['path'] = os.path.join(hyp_path)


        s1[data_type]['sent'] = [normalizeString(line.rstrip().split('\t')[0]) for line in open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [normalizeString(line.rstrip().split('\t')[1]) for line in open(s2[data_type]['path'], 'r')]
        temp = [float(line.rstrip().split('\t')[2]) for line in open(hyp_path,'r')]

        first, second, gs = s1[data_type]['sent'], s2[data_type]['sent'], temp

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) 
        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(data_type.upper(), len(s1[data_type]['sent']), data_type))

    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent']}
    return test, first, second, gs

def get_sts_nli(data_path):
    s1 = {}
    s2 = {}
    target = {}
    first, second, gs = list(), list(), list()
    #dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}
    #dico_label = {'entailment': 0,  'non-entailment':1}

    for data_type in ['train']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        #target[data_type]['path'] = os.path.join(data_path,'labels.' + data_type)

        s1[data_type]['sent'] = [line.rstrip() for line in open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [line.rstrip() for line in open(s2[data_type]['path'], 'r')]
        gs = [float(line.strip()) for line in open(os.path.join(data_path,'gs.' + data_type),'r')]
        #target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')] for line in open(target[data_type]['path'], 'r')])
        first, second = s1[data_type]['sent'], s2[data_type]['sent']
        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent']}
   
    return train, first, second, gs

def get_nli(data_path):
    s1 = {}
    s2 = {}
    target = {}

    #dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}
    dico_label = {'entailment': 0,  'non-entailment':1}

    for data_type in ['train', 'dev', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        target[data_type]['path'] = os.path.join(data_path,
                                                 'labels.' + data_type)

        s1[data_type]['sent'] = [line.rstrip() for line in
                                 open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [line.rstrip() for line in
                                 open(s2[data_type]['path'], 'r')]
        target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]
                for line in open(target[data_type]['path'], 'r')])

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
            len(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': target['train']['data']}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],
           'label': target['dev']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': target['test']['data']}
    return train, dev, test
