
from nltk.corpus import wordnet as wn
from nltk.corpus import semcor
import numpy as np
import collections
import os
import pickle
from nltk import corpus

def getContextEmb(sentence,center,window_size,embedding_dict,emb_size):
    # Input introductions
    # sentence: an array of tokens of untagged sentence. 
    # center: position of the center word
    # window_size: size of context window
    # embedding_Dict: embedding dictionary used to calculate context
    ################################################################
    start_pos = max([0,center-window_size])
    end_pos = min([len(sentence),(center+window_size)+1])
    context_tokens = sentence[start_pos:end_pos]
    output_embedding = np.zeros(emb_size)
    for word in context_tokens:
        try:
            output_embedding+=embedding_dict[word]
        except:
            output_embedding+=np.random.uniform(1,-1,emb_size)
    return output_embedding

def buildSemEmb(tagged_sents,emb_size,embedding_dict,window_size=4):
    def getContextEmb(sentence,center,window_size,embedding_dict,emb_size):
    # Input introductions
    # sentence: an array of tokens of untagged sentence. 
    # center: position of the center word
    # window_size: size of context window
    # embedding_Dict: embedding dictionary used to calculate context
    ################################################################
        start_pos = max([0,center-window_size])
        end_pos = min([len(sentence),(center+window_size)+1])
        context_tokens = sentence[start_pos:end_pos]
        output_embedding = np.zeros(emb_size)
        for word in context_tokens:
            try:
                output_embedding+=embedding_dict[word]
            except:
                output_embedding+=np.random.uniform(1,-1,emb_size)
        return output_embedding

    output_dict = collections.defaultdict(lambda: np.zeros(emb_size))
    count_dict = collections.defaultdict(lambda: 0)
    for sentence in tagged_sents:
        #print(sentence)
        for idx,chunk in enumerate(sentence):
            if(type(chunk))==list:
                continue
            else:
                #Use try except handling since some of the label is broken
                try:
                    sense_index = chunk.label().synset().name()
                except:
                    continue
                context_emb = getContextEmb(sentence,idx,window_size,embedding_dict,emb_size)
                output_dict[sense_index]+=context_emb
                count_dict[sense_index]+=1
    # Averaging
    for key in output_dict.keys():
        output_dict[key] /= count_dict[key]
    return output_dict

def load_glove_embeddings(glove_directory,emsize=50,voc_size=50000):
    #get directory name glove.6B or other training corpus size
    if glove_directory[-1] =='/':
        dirname = glove_directory.split('/')[-2]
    else:
        dirname = glove_directory.split('/')[-1]
    if emsize in [50,100,300]:
        f = open(os.path.join(glove_directory,'%s.%sd.txt'%(dirname,emsize)))
    else:
        print('Please select from 50, 100 or 300')
        return
    loaded_embeddings = collections.defaultdict()
    for i, line in enumerate(f):
        if i >= voc_size: 
            break
        s = line.split()
        loaded_embeddings[s[0]] = np.asarray(s[1:],dtype='float64')
    return loaded_embeddings


import sys
import re


class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)