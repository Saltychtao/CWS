
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

    def l_features(self,span0,span1):
        """
        Return a list of features of each index.
        (pre-s1-span,s1-span,curIndex)
        return_type : [(lefts,rights)]
        """
        lefts = []
        rights = []

        #pre-s1-span
        lefts.append(1)
        rights.append(span0[0]-1)

        #s0-span
        lefts.append(span0[0])
        rights.append(span0[1])

        # s1-span
        lefts.append(span1[0])
        rights.append(span1[1])

        #post-span
        lefts.append(span1[1]+1)
        rights.append(self.n)

        return tuple(lefts),tuple(rights)
    
    def wrap_result(self):
        ### TODO
        np_indexs = self.cur_layer.np_indexs()
        self.labels = ['AP' for i in range(self.n)]
        for index in np_indexs:
            self.labels[index-1] = 'NA'

        return SegSentence(
            [('S','NA')] +
            [(c,pred_l) for ((c,gold_l),pred_l) in zip(self.gold_sentence,self.labels)] +
            [('/S','NA')]
        )

    @staticmethod
    def exploration(data,fm,network,drop_prob,unk_params):
        # dynet.renew_cg()
        network.prep_params()

        segSentence = data['segSentence']

        n = segSentence.n0
        state = Segmenter(n,segSentence)
        state.seg_sentence = [c for (c, l) in segSentence.sentence[1:-1]]

        state.prep_layer()

        ### Random UNK
        for (i,uni) in enumerate(data['unigrams']):
            if uni <= 2:
                continue

            u_freq = fm.unigrams_freq_list[uni]
            drop_prob = unk_params / (unk_params + u_freq)
            r = np.random.random()
            if r < drop_prob :
                 data['unigrams'][i] = 0

        for (i,bi) in enumerate(data['fwd_bigrams']):
            if bi <= 2:
                continue

            b_freq = fm.bigrams_freq_list[bi]
            drop_prob = unk_params / (unk_params + b_freq)
            r = np.random.random()
            if r < drop_prob:
                data['fwd_bigrams'][i] = 0
        
        fwd_bigrams = data['fwd_bigrams']
        unigrams = data['unigrams']
        network.evaluate_recurrent(fwd_bigrams, unigrams, test=False)

        errors = []
        state_cnt = 0
        cached_scores = {}
        for step in range(n):
            ap_scores = []
            # ap_value = []
            for i in range(len(state.cur_layer)-1):
                left_span = state.cur_layer[i]
                right_span = state.cur_layer[i + 1]
                #left,right = state.l_features(span0,span1)
                if not (left_span, right_span) in cached_scores:
                    scores = network.evaluate_labels(
                        left_span,
                        right_span,
                        test=False,
                    )
                    cached_scores[(left_span, right_span)] = scores
                else:
                    scores = cached_scores[(left_span, right_span)]
                #ap_value.append(scores.npvalue())
                ap_scores.append(scores)


            loss1 = state.get_loss_1(ap_scores)
            loss2 = state.get_loss_2(ap_scores)

            errors.extend(loss1)
            errors.extend(loss2)

            state_cnt += 1

            combined_spans = state.select_combine(ap_scores)
            if combined_spans is None:
                break
            else:
                left, right = combined_spans
                network.update_encoding(left, right)
        
        predicted = state.wrap_result()
        accuracy = predicted.compare(segSentence)

        result = {
            'state_cnt':state_cnt,
            'loss':errors,
        }
        #if len(errors) == 0:
        #    return dynet.scalarInput(0),accuracy
        return errors,accuracy


    def l_oracle(self):
        return self.gold_sentence[self.i][1]
 
    def label_index(self,label):
        if label == 'AP':
            return 0
        elif label == 'NA':
            return 1

    def get_loss_1(self,scores):

        assert(len(scores) == len(self.cur_layer) - 1)

        errors = []
        values = [s.npvalue() for s in scores]
        for i in range(len(self.cur_layer)-1):
            span0 = self.cur_layer[i]
            span1 = self.cur_layer[i+1]
            index = span1[0]
            probs = dynet.softmax(scores[i])
            #print(probs.value())

            #probs_value = probs.npvalue()
            correct = self.label_index(self.gold_sentence[index-1][1])
            tmp = dynet.pick(probs, correct)
            # print(tmp.value())
            # if abs(tmp.value()) < 1e-50:
            #    continue
            loss = -dynet.log(tmp)
            #loss =  -dynet.log(dynet.pick(probs,correct))
            errors.append(loss)

        return errors


    def get_loss_2(self,scores):
        
        assert(len(scores) == len(self.cur_layer) -1 )

        errors = []
        pos,combine_scores = self.select_position(scores)


        if len(pos) == 0:
            return errors
        else:
            probs = dynet.softmax(dynet.concatenate(combine_scores))
            candidates = self.check_oracle(pos)
            ap_values = [s.value() for s in combine_scores]
            if len(candidates) == 0 :
                index = np.argmin(ap_values)
                tmp = dynet.pick(probs, index)
                # if abs(tmp.value()) > 1e-50:
                loss = - dynet.log(tmp)
                errors.append(loss)
            else:
                index = pos.index(candidates[0])
                tmp = dynet.pick(probs, index)
                # if abs(tmp.value()) > 1e-50:
                loss = - dynet.log(tmp)
                errors.append(loss)
            return errors

    def check_oracle(self,pos):
        candidates = []
        for p in pos:
            index = p[1][0]
            if self.gold_sentence[index - 1][1] == 'AP':
                candidates.append(p)
        return candidates

    def select_position(self,scores):
        
        assert(len(scores) == len(self.cur_layer) - 1)
        pos = []
        combine_scores = []
        values = [ s.npvalue() for s in scores]
        for i in range(len(values)):
            if values[i][0] >= values[i][1]:
                pos.append((self.cur_layer.span[i], self.cur_layer.span[i + 1]))
                combine_scores.append(scores[i][0])

        return pos,combine_scores

    def combine(self,pos):
        self.cur_layer.combine(pos)

    def select_combine(self,scores):
        pos,combine_scores = self.select_position(scores)
        if len(combine_scores) != 0 :
            values = [s.value for s in combine_scores]
            index = np.argmax(values)
            self.combine(pos[index])
            return pos[index]
        else:
            return None

    def prep_layer(self):
        self.cur_layer = Layer(self.gold_sentence)

    @staticmethod
    def segment(seg_sentence,fm,network):
        dynet.renew_cg()
        network.prep_params()

        n = seg_sentence.n0
        state = Segmenter(n,seg_sentence)

        fwd_b,u = fm.sentence_sequence(seg_sentence)

        network.evaluate_recurrent(fwd_b, u, test=True)

        state.prep_layer()
        cached_scores = {}
        for step in range(n):
            ap_scores = []

            for i in range(len(state.cur_layer) - 1):
                left_span = state.cur_layer[i]
                right_span = state.cur_layer[i + 1]
                #left,right = state.l_features(span0,span1)
                if not (left_span, right_span) in cached_scores:
                    scores = network.evaluate_labels(
                        left_span,
                        right_span,
                        test=True
                    )
                    cached_scores[(left_span, right_span)] = scores
                else:
                    scores = cached_scores[(left_span, right_span)]

                ap_scores.append(scores)

            combined_spans = state.select_combine(ap_scores)
            if combined_spans is None:
                break
            else:
                left, right = combined_spans
                network.update_encoding(left, right)

        predicted = state.wrap_result()
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
    def __init__(self,gold_sentence):
        self.span = [(i + 1, i + 1) for i in range(len(gold_sentence))]
        self.sentence = gold_sentence

    def __len__(self):
        return len(self.span)

    def combine(self,pos):
        (left_span, right_span) = pos
        layer = []
        i = 0
        while i < len(self.span):
            if self.span[i][0] == left_span[0] and self.span[i][1] == left_span[1]:
                layer.append((left_span[0], right_span[1]))
                i +=2

            else:
                layer.append(self.span[i])
                i += 1
        self.span = layer
        # layer = []
        # i = 0
        # while i < len(self.span):
        #     if self.span[i][1] + 1 != pos:
        #         layer.append(self.span[i])
        #         i += 1
        #     else:
        #         span0 = self.span[i]
        #         span1 = self.span[i + 1]
        #         layer.append((span0[0],span1[1]))
        #         i += 2
        # self.span = layer

    def np_indexs(self):

        result= []
        for span in self.span:
            result.append(span[0])
        return result

    def __getitem__(self, item):
        return self.span[item]
