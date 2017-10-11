import time
import dynet
import numpy as np
from Corpus import PosCorpus
from seg_sentence import PosSentence
from seg_sentence import Accuracy

class Tagger(object):
    def __init__(self,n,pos_sentence):
        self.stack = []
        self.n = n
        self.i = 1
        self.tags = []
        self.gold_sentence = pos_sentence.sentence

    def features(self):

        lefts = []
        rights = []

        # lefts.append(1)
        # if len(self.stack) < 1:
        #     rights.append(0)
        # else:
        #     s0_left = self.stack[-1][0] + 1
        #     rights.append(s0_left -1)
        #
        # if len(self.stack) < 1:
        #     lefts.append(1)
        #     rights.append(0)
        # else:
        #     s0_left = self.stack[-1][0] + 1
        #     lefts.append(s0_left)
        #     s0_right = self.stack[-1][1]+ 1
        #     rights.append(s0_right)

        # if self.i == 0:
        #     lefts.append(1)
        #     rights.append(0)
        # else:
        #     lefts.append(self.i-1)
        #     rights.append(self.i - 1)


        lefts.append(self.i)
        rights.append(self.i)

        # lefts.append(self.i+1)
        # rights.append(self.n)

        return tuple(lefts),tuple(rights)

    def wrap_result(self):
        return PosSentence([('<S>','<S>')] + [(c,pred_t) for (pred_t,(c,gold_t)) in zip(self.tags,self.gold_sentence)] + [('</S>','</S>')])

    @staticmethod
    def training_data(fm,pos_sentence):
        data = []
        n = pos_sentence.n0
        state = Tagger(n,pos_sentence)

        for step in range(n):
            action = state.oracle()
            features = state.features()
            state.take_action(action)
            data.append((features,fm.label_index(action)))

        return data

    @staticmethod
    def exploration(data,fm,network,alpha):
        dynet.renew_cg()
        network.prep_params()

        tag_data = {}

        pos_sentence = data['posSentence']

        n = pos_sentence.n0
        state = Tagger(n,pos_sentence)

        w = data['words']

        fwd,back = network.evaluate_recurrent(w,test=False)

        for step in range(n):
            features = state.features()
            correct_action = state.oracle()
            tag_data[features] = fm.tag_index(correct_action)
            r = np.random.random()
            if r < alpha:
                action = correct_action
            else:
                left,right = features
                scores = network.evaluate_tags(
                    fwd,
                    back,
                    left,
                    right,
                    test=True
                ).npvalue()
                action_index = np.argmax(scores)
                action = fm.tag_action(action_index)

            state.take_action(action)

        predicted = state.wrap_result()
        assert len(predicted.tags) == (len(state.tags) + 2)
        accuracy = predicted.compare(pos_sentence)

        example = {
            'w':w,
            'tag_data':tag_data
        }

        return example,accuracy

    def oracle(self):
        return self.gold_sentence[self.i][1]

    def finished(self):
        return self. i == self.n+1

    def take_action(self,action):
        self. i += 1
        self.tags.append(action)

    @staticmethod
    def tag(pos_sentence,fm,network):

        dynet.renew_cg()
        network.prep_params()

        n = pos_sentence.n0
        state = Tagger(n,pos_sentence)

        w = fm.sentence_sequence(pos_sentence)

        fwd,back = network.evaluate_recurrent(w,test=True)

        for step in range(n):
            left,right = state.features()
            scores = network.evaluate_tags(
                fwd,
                back,
                left,
                right,
                test=True,
            ).npvalue()
            action_index = np.argmax(scores)
            action = fm.tag_action(action_index)

            state.take_action(action)

        if not state.finished():
            raise RuntimeError ('Bad ending state!')

        predicted = state.wrap_result()
        return predicted

    @staticmethod
    def evaluate_corpus(sentences,fm,network):
        accuracy = Accuracy()
        for sentence in sentences:
            pos_sentence = PosSentence(sentence)
            predicted = Tagger.tag(pos_sentence,fm,network)
            local_accuracy = predicted.compare(pos_sentence)
            accuracy += local_accuracy
        return accuracy

    @staticmethod
    def write_predicted(fname,test_sentences,fm,network):
        start_time = time.time()

        accuracy = Accuracy()
        f = open(fname,'w')
        for sentence in test_sentences:
            pos_sentence = PosSentence(sentence)
            predicted = Tagger.tag(pos_sentence,fm,network)
            local_accuracy = predicted.compare(pos_sentence)
            accuracy += local_accuracy
            f.write(predicted.pretty() + '\n')
        f.close()

        print(accuracy)

        current_time = time.time()
        runmins = (current_time - start_time) /60
        print(' Elapsed time : {:.2f}m'.format(runmins))