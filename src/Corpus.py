import numpy as np

import json
from seg_sentence import SegSentence
from Segmenter import Segmenter

from collections import defaultdict,OrderedDict
class Corpus(object):

    UNK = '<UNK>'
    START = '<s>'
    STOP = '</s>'

    @staticmethod
    def vocab_init(fname,verbose = True):

        unigrams_freq = defaultdict(int)
        bigrams_freq = defaultdict(int)
        label_freq = defaultdict(int)

        sentences = SegSentence.load_sentence_file(fname)
        for i,sentence in enumerate(sentences):
            pre_unigram = Corpus.START
            for (unigram,label) in sentence:
                unigrams_freq[unigram] += 1
                bigrams_freq [pre_unigram+unigram] += 1
                bigrams_freq[unigram + pre_unigram] +=1
                pre_unigram = unigram
                label_freq[label] += 1
            bigrams_freq[pre_unigram+Corpus.STOP] += 1
            bigrams_freq[Corpus.STOP+pre_unigram] += 1
        
        bigrams = [Corpus.UNK] + sorted(bigrams_freq)
        unigrams = [
            Corpus.START,
            Corpus.STOP,
            Corpus.UNK,
        ] + sorted(unigrams_freq)

        bdict = OrderedDict((b,i) for (i,b) in enumerate(bigrams))
        udict = OrderedDict((u,i) for (i,u) in enumerate(unigrams))

        labels = sorted(label_freq)
        ldict = OrderedDict((l,i) for (i,l) in enumerate(labels))

        return {
            'origin_sentences':sentences,
            'bdict':bdict,
            'bigrams_freq':bigrams_freq,
            'unigrams_freq':unigrams_freq,
            'udict':udict,
            'ldict':ldict,
            'bigrams':bigrams,
            'unigrams':unigrams,
        }

    def __init__(self,vocabfile):
        if vocabfile is not None:
            data = Corpus.vocab_init(vocabfile)

            self.origin_sentences = data['origin_sentences']
            self.bdict = data['bdict']
            self.udict = data['udict']
            self.bigrams_freq = data['bigrams_freq']
            self.unigrams_freq = data['unigrams_freq']
            self.bigrams = data['bigrams']
            self.unigrams = data['unigrams']
            self.ldict = data['ldict']

            self.bigrams_freq_list = []
            for word in self.bdict.keys():
                if word in self.bigrams_freq:
                    self.bigrams_freq_list.append(self.bigrams_freq[word])
                else:
                    self.bigrams_freq_list.append(0)

            self.unigrams_freq_list = []
            for word in self.udict.keys():
                if word in self.unigrams_freq:
                    self.unigrams_freq_list.append(self.unigrams_freq[word])
                else:
                    self.unigrams_freq_list.append(0)
    @staticmethod
    def from_dict(data):
        new = Corpus(None)
        new.bdict = data['bdict']
        new.udict = data['udict']
        new.ldict = data['ldict']
        new.bigrams = data['bigrams']
        new.unigrams = data['unigrams']
        new.bigrams_freq_list = data['bigrams_freq_list']
        new.unigrams_freq_list = data['unigrams_freq_list']
        return new

    def as_dict(self):
        return {
        'bdict':self.bdict,
        'udict':self.udict,
        'ldict':self.ldict,
        'bigrams':self.bigrams,
        'unigrams':self.unigrams,
        'bigrams_freq_list':self.bigrams_freq_list,
        'unigrams_freq_list':self.unigrams_freq_list,
    }

    def save_json(self,filename):
        with open (filename,'w') as fh:
            json.dump(self.as_dict(),fh,encoding='utf-8')

    @staticmethod
    def load_json(filename):
        with open (filename) as fh:
            data = json.load(fh,object_pairs_hook=OrderedDict,encoding='utf-8')
        return Corpus.from_dict(data)

    def total_bigrams(self):
        return len(self.bdict)

    def total_unigrams(self):
        return len(self.udict)

    def total_labels(self):
        return len(self.ldict)

    def total_span_nums(self):
        if self.total_labels() == 2 : # Using 'Append','Not-append' Label system
            return 4
        elif self.total_labels() == 4 : # Using 'B,M,S,E' label system
            return 4

    def sentence_sequence(self,seg_sentence):

        fwd_bi = seg_sentence.fwd_bigram()
        uni = seg_sentence.unigram()
        fwd_bigram_id = [
            self.bdict[b.decode('utf-8')]
            if b.decode('utf-8') in self.bdict else self.bdict[Corpus.UNK]
            for b in fwd_bi
        ]
        unigram_id = [
            self.udict[u.decode('utf-8')]
            if u.decode('utf-8') in self.udict else self.udict[Corpus.UNK]
            for u in uni
         ]
        fwd_bi = np.array(fwd_bigram_id).astype('int32')
        u = np.array(unigram_id).astype('int32')

        return fwd_bi,u

    def gold_data(self,sentence):
        seg_sentence = SegSentence(sentence)
        fwd_bigrams,unigrams = self.sentence_sequence(seg_sentence)

        features = Segmenter.training_data(seg_sentence)

        return {
            'segSentence':seg_sentence,
            'fwd_bigrams': fwd_bigrams,
            'unigrams': unigrams,
            'features':features,
        }
        
    def gold_data_from_file(self,fname):

        sentences = SegSentence.load_sentence_file(fname)
        result = []
        for sentence in sentences:
            sentence_data = self.gold_data(sentence)
            result.append(sentence_data)
        return result



    





    
            
