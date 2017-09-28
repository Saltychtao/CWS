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
    
    @staticmethod
    def load_sentence_file(fname,mode = "Append"):
        sentences = []
        sentence = [SegSentence.sentence_begin(mode)]
        with open(fname) as f:
            while f is not None:
                line = f.readline()
                if line == '':
                    break
                if line  == '\n':
                    sentence.append(
                        SegSentence.sentence_end(mode)
                    )
                    sentences.append(sentence)
                    sentence = [SegSentence.sentence_begin(mode)]

                else :
                    character_label = line.split('\t')
                    character = character_label[0]
                    label = character_label[1].split('\n')[0]
                    sentence.append(
                        (character,label)
                    )
        # f.close()
        return sentences

    @staticmethod
    def sentence_begin(mode):
        if mode == 'BMSE':
            return ('<S>','S')
        elif mode == 'Append':
            return ('<S>','NA')

    @staticmethod
    def sentence_end(mode):
        if mode == 'BMSE':
            return ('</S>','S')
        elif mode == "Append":
            return ('</S>','NA')

    def pretty(self):
        str = ''
        for i in range(1,self.n0+1):
            if self.sentence[i][1] == 'NA':
               str += (' ' + self.sentence[i][0])
            elif self.sentence[i][1] == 'AP':
                str += self.sentence[i][0]

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
    def __init__(self, sentence):
        self.sentence = sentence
        self.n0 = len(sentence) - 2
        self.n = len(sentence)

    def words(self):
        self.words = []
        for (w, l) in self.sentence:
            self.words.append(w)

        return self.words

    @staticmethod
    def sentence_begin():
        return ('S', 'S')

    @staticmethod
    def sentence_end():
        return ('\S', '\S')

    @staticmethod
    def load_sentence_file(fname):
        sentences = []
        sentence = [PosSentence.sentence_begin()]
        word = ''
        tag = ''
        with open(fname, 'r') as f:
            while f is not None:
                line = f.readline()
                if line == '':
                    break
                elif line == '\n':
                    sentence.append(PosSentence.sentence_end())
                    sentences.append(sentence)
                    sentence = [PosSentence.sentence_end()]

                else:
                    strs = line.split()
                    if len(strs) == 3:
                        sentence.append((word, tag))
                        word = strs[0]
                        tag = strs[1]
                    else:
                        word += strs[0]
        return sentences



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

        
    
