class SegSentence(object):
    def __init__(self,sentence,mode = 'Append'):
        self.sentence = sentence   ## (character,label) tuples ,include START AND STOP
        self.mode = mode
        self.n0 =  len(sentence)-2
        self.n = len(sentence)

    def fwd_bigram(self):
        self.fwd_bigram = []
        for i in range(self.n-1):
            self.fwd_bigram.append(self.sentence[i][0]+self.sentence[i+1][0])
        self.fwd_bigram = [('SS')] + self.fwd_bigram + [('/S/S')]
        return self.fwd_bigram

    # def back_bigram(self):
    #     self.back_bigram = []
    #     for i in range(self.n-1):
    #         self.back_bigram.append(self.sentence[i+1][0] + self.sentence[i][0])
    #     return self.back_bigram

    def unigram(self):
        self.unigram = []
        for i in range(self.n):
            self.unigram.append(self.sentence[i][0])
        return self.unigram
    
    # @staticmethod
    # def load_sentence_file(fname,mode = "Append"):
    #     sentences = []
    #     sentence = [SegSentence.sentence_begin(mode)]
    #     with open(fname) as f:
    #         while f is not None:
    #             line = f.readline()
    #             if line == '':
    #                 break
    #             if line  == '\n':
    #                 sentence.append(
    #                     SegSentence.sentence_end(mode)
    #                 )
    #                 sentences.append(sentence)
    #                 sentence = [SegSentence.sentence_begin(mode)]
    #
    #             else :
    #                 character_label = line.split('\t')
    #                 character = character_label[0]
    #                 label = character_label[1].split('\n')[0]
    #                 sentence.append(
    #                     (character,label)
    #                 )
    #     # f.close()
    #     return sentences

    # @staticmethod
    # def sentence_begin(mode='Append'):
    #     if mode == 'BMSE':
    #         return ('<S>','S')
    #     elif mode == 'Append':
    #         return ('<S>','NA')

    # @staticmethod
    # def sentence_end(mode='Append'):
    #     if mode == 'BMSE':
    #         return ('</S>','S')
    #     elif mode == "Append":
    #         return ('</S>','NA')

    def __str__(self):
        str = ''
        for i in range(1,self.n0+1):
            if self.sentence[i][1] == 'NA':
               str += (' ' + self.sentence[i][0])
            elif self.sentence[i][1] == 'AP':
                str += self.sentence[i][0]

        self._str = str
        return str

    def get_spans(self):

        result = set()
        if self.n0 == 1:
            result.add((1,1,))
            return result
        left = 1
        right = 2
        while right <= self.n0+1:
            if self.sentence[right][1] == 'NA' :
                result.add((left,right-1))
                left = right
                right += 1
            elif self.sentence[right][1] ==  'AP':
                right += 1

        return result

    def get_words_length(self):
        self.words_length = 0
        for i in range(1,self.n0+1):
            if self.sentence[i][1] == 'NA':
                self.words_length += 1
        return self.words_length

    def compare(self,other):
        gold_counts = other.get_words_length()
        pred_counts = self.get_words_length()

        gold_spans = other.get_spans()
        pred_spans = self.get_spans()

        n_right_words = 0
        for ps in gold_spans:
            if ps in pred_spans:
                n_right_words += 1

        return FScore(correct=n_right_words,predcount=pred_counts,goldcount=gold_counts)


class PosSentence(object):
    def __init__(self,sentence):
        self.sentence = self.ct2wt(sentence)
        self.n = len(self.sentence)
        self.n0 = self.n-2
        self.words = self.get_words()
        self.tags = self.get_tags()

    @staticmethod
    def ct2wt(sentence):
        wt_list = []
        word = sentence[0][0]
        tag = sentence[0][1]
        for (c,t) in sentence[1:]:
            if t == 'NotStart':
                word += c
            else:
                wt_list.append((word,tag))
                word = c
                tag = t
        wt_list.append((word,tag))
        return wt_list

    def get_words(self):
        self.words = []
        for (w,_) in self.sentence:
            self.words.append(w)
        return self.words

    def get_tags(self):
        self.tags = []
        for (_,t) in self.sentence:
            self.tags.append(t)
        return self.tags

    def pretty(self):
        s = ''
        for (w,t) in self.sentence:
            s += '(' + w + ' ' + t + ')'
        return s

    def compare(self,other):
        lenA = len(self.tags)
        lenB = len(other.tags)
        assert (lenA == lenB)
        correct_cnt = 0
        for (t1,t2) in zip(self.tags,other.tags):
            if t1 == t2:
                correct_cnt += 1

        return Accuracy(correct_cnt,lenA)

class Accuracy(object):
    def __init__(self,correct = 0,count = 0):
        self.correct = correct
        self.count = count

    def precision(self):
        if self.count > 0:
            return (100.0*self.correct) / self.count
        else:
            return 0

    def __str__(self):
        precision = self.precision()
        return '(Accuracy = {:0.2f})'.format(precision)

    def __iadd__(self,other):
        self.correct += other.correct
        self.count += other.count
        return self

    def __cmp__(self,other):
        return cmp(self.precision(),other.precision())

    def __add__(self,other):
        return Accuracy(self.correct + other.correct,
                        self.count + other.count)
class Sentence(object):
    def __init__(self,sentence):
        self.sentence = sentence
        self.n = len(sentence)
        self.n0 = self.n - 2
        self.seg_sentence = SegSentence([(c,l) for (c,_,l) in sentence])
        self.pos_sentence = PosSentence([(c,t) for (c,t,_) in sentence])



    @staticmethod
    def sentence_begin():
        return ('<S>','<S>','NA')

    @staticmethod
    def sentence_end():
        return ('</S>','</S>','NA')

    @staticmethod
    def load_sentence_file(fname, mode):
        sentences = []
        sentence = [Sentence.sentence_begin()]
        with open(fname) as f:
            while f is not None:
                line = f.readline()
                if line == '':
                    break
                if line == '\n':
                    sentence.append(
                        Sentence.sentence_end()
                    )
                    sentences.append(sentence)
                    sentence = [Sentence.sentence_begin()]

                else:
                    ctl_strs = line.split()
                    c = ctl_strs[0]
                    t = ctl_strs[1]
                    l = ctl_strs[2]
                    sentence.append(
                        (c,t,l)
                    )
        sentence.append(Sentence.sentence_end())
        sentences.append(sentence)
        result = []
        if mode == 'seg':
            for s in sentence:
                result.append(
                    [(c,l) for (c,t,l) in s]
                )
        elif mode == 'pos-tagging':
            for s in sentences:
                result.append(
                    [(c,t) for (c,t,l) in s]
                )
        elif mode == 's&p':
            result = sentences

        return result




class FScore(object):
    
    def __init__(self,correct = 0,predcount =0,goldcount = 0):
        self.correct = correct
        self.predcount = predcount
        self.goldcount =goldcount

    def precision(self):
        if self.predcount > 0:
            return (100.0*self.correct) / self.predcount
        else :
            return 0.0

    def recall(self):
        if self.goldcount > 0:
            return (100.0 * self.correct) / self.goldcount
        else :
            return 0.0

    def fscore(self):
        precision = self.precision()
        recall = self.recall()
        if (precision + recall ) > 0:
            return (2*precision *recall) /(precision + recall)
        else:
            return 0.0

    def __str__(self):
        precision = self.precision()
        recall = self.recall()
        fscore = self.fscore()
        return '(P = {:0.2f}, R= {:0.2f}, F= {:0.2f})'.format(
            precision,
            recall,
            fscore
        )

    def __iadd__(self,other):
        self.correct += other.correct
        self.predcount += other.predcount
        self.goldcount += other.goldcount
        return self
    
    def __cmp__(self,other):
        return cmp(self.fscore,other.fscore)
    
    def __add__(self,other):
        return FScore(self.correct + other.correct,
                        self.predcount + other.predcount,
                        self.goldcount + other.goldcount)

        
    
