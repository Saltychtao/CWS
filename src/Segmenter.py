
import time
import dynet
import numpy as np
from seg_sentence import SegSentence
from seg_sentence import FScore


class Segmenter(object):
    def __init__(self,n,seg_sentence):
        self.todo = 0
        self.n = n
        self.i = 0
        self.labels = []
        self.gold_sentence = seg_sentence.sentence[1:-1]

    ''' def l_features(self):
        """
        Return a list of features of each index.
        (pre-s1-span,s1-span,curIndex)
        return_type : [(lefts,rights)]
        """
        lefts = []
        rights = []

        #pre-s1-span
        lefts.append(1)
        if len(self.stack) < 1:
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            rights.append(s0_left-1)

        #s1-span
        if len(self.stack) < 1:
            lefts.append(1)
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            lefts.append(s0_left)
            s0_right = self.stack[-1][1] +1
            rights.append(s0_right)

        # curIndex
        lefts.append(self.i)
        rights.append(self.i)
        #post-span

        lefts.append(self.i+1)
        rights.append(self.n)

        return tuple(lefts),tuple(rights)
    
    def can_append(self):
        return not (self.i == 0)
    '''
    def wrap_result(self):
        ### TODO



     @staticmethod
    def training_data(seg_sentence):
        l_features = []
        n = len(seg_sentence.sentence) - 2
        state = Segmenter(n,seg_sentence)

        for step in range(n):
            if not state.can_append():
                state.no_append() 
            else:
                action = state.l_oracle()
                features = state.l_features()
                state.take_action(action)
                l_features.append((features,Segmenter.l_action_index(action)))

        return l_features 

    @staticmethod
    def exploration(data,fm,network,alpha=1.0,beta=0):
        dynet.renew_cg()
        network.prep_params()

        label_data = {}

        segSentence = data['segSentence']

        n = segSentence.n0
        state = Segmenter(n,segSentence)
        state.seg_sentence = [c for (c, l) in segSentence.sentence[1:-1]]

        fwd_bigrams = data['fwd_bigrams']
        unigrams = data['unigrams']
        fwd,back = network.evaluate_recurrent(fwd_bigrams,unigrams,test=True)

        for step in range(n):
            features = state.l_features()
            if not state.can_append():
                action = 'NA'
                correct_action = 'NA'
            else:

                correct_action = state.l_oracle()

                r = np.random.random()
                if r < beta:
                    action = correct_action
                else:
                    left,right = features
                    scores = network.evaluate_labels(
                        fwd,
                        back,
                        left,
                        right,
                        test=True,
                    ).npvalue()

                    # sample from distribution
                    exp = np.exp(scores * alpha)
                    softmax = exp / (exp.sum())
                    r = np.random.random()

                    if r <= softmax[0]:
                        action = 'AP'
                    else:
                        action = 'NA'
            
            label_data[features] = Segmenter.l_action_index(correct_action)
            state.take_action(action)
        
        predicted = state.wrap_result()
        accuracy = predicted.compare(segSentence)

        example = {
            'fwd_bigrams':fwd_bigrams,
            'unigrams':unigrams,
            'label_data':label_data
        }

        return example,accuracy


    def l_oracle(self):
        return self.gold_sentence[self.i][1]
 

    def select_combine(self,scores)
        pos = np.argmax([score for  (index,score) in scores])
        self.cur_layer.combine(scores[pos][0])
        
    @staticmethod
    def segment(seg_sentence,fm,network):
        dynet.renew_cg()
        network.prep_params()

        n = seg_sentence.n0
        state = Segmenter(n,seg_sentence)

        fwd,back = network.evaluate_recurrent(fwd_b,u,test=True)

        state.cur_layer = state.prep_layer()
        for step in range(n):
            ap_scores = []

            for i in range(1,len(state.cur_layer)+1):
                span0 = state.cur_layer[i]
                span1 = state.cur_layer[i + 1]
                left,right = state.l_features(span0,span1)
                scores = network.evaluate_labels(
                    fwd,
                    back,
                    left,
                    right,
                    test=True
                ).npvalue()
                action_index = np.argmax(scores)
                action = Segmenter.l_action(action_index)
                if action == 'combine':
                    ap_scores.append((i,scores[0]))
            if ap_scores == []:
                break
            state.select_combine(ap_scores)

        predicted = self.wrap_result()
        return predicted    

    @staticmethod
    def evaluate_corpus(sentences,fm,network):
        accuracy = FScore()
        for sentence in sentences:
            seg_sentence = SegSentence(sentence)
            predicted = Segmenter.segment(seg_sentence,fm,network)
            local_accuracy = predicted.compare(seg_sentence)
            accuracy += local_accuracy

        return accuracy

    @staticmethod
    def write_predicted(fname,test_sentences,fm,network):
        """
        Input sentences being used only to carry sentences.
        """
        start_time = time.time()

        f = open(fname,'w+')
        for sentence in test_sentences:
            seg_sentence = SegSentence(sentence)
            predicted = Segmenter.segment(seg_sentence,fm,network)
            f.write(predicted.pretty() + '\n')
        f.close()

        current_time = time.time()
        runmins = (current_time - start_time) / 60
        print(' Elapsed time: {:.2f}m'.format(runmins))

        

        
class Layer(object):
    def __init__(self,gold_sentence)
        self.layer = [(i+1,i+1) for i in range(len(gold_sentence)) ]
        self.sentence = gold_sentence

    def __len__(self):
        return len(self.layer)

    def combine(self,pos):
        layer = []
        for i in range(len(self.layer)):
            if i is not pos:
                layer.append(self.layer[i])
            else:
                span0 = self.layer[i]
                span1 = self.layer[i+1]
                layer.append(span0[0],span1[1])
                i += 1
        self.layer = layer



