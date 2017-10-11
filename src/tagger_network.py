
from __future__ import print_function

import time
import numpy as np
import dynet
import random
import sys

from seg_sentence import Sentence
from seg_sentence import Accuracy
from Tagger import Tagger


class LSTM(object):
    def __init__(self, input_dims, output_dims, model):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.model = model

        self.W_i = model.add_parameters(
            (output_dims, input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_i = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )
        self.W_f = model.add_parameters(
            (output_dims, input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_f = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )
        self.W_c = model.add_parameters(
            (output_dims,input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_c = model.add_parameters(
                (output_dims,),
                init = dynet.ConstInitializer(0)
        )
        self.W_o = model.add_parameters(
            (output_dims, input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_o = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )
        self.c0 = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )

        self.W_params = [self.W_i, self.W_f, self.W_c, self.W_o]
        self.b_params = [self.b_i, self.b_f, self.b_c, self.b_o]
        self.params = self.W_params + self.b_params + [self.c0]

    class State(object):
        def __init__(self, lstm):
            self.lstm = lstm

            self.outputs = []

            self.c = dynet.parameter(self.lstm.c0)
            self.h = dynet.tanh(self.c)

            self.W_i = dynet.parameter(self.lstm.W_i)
            self.b_i = dynet.parameter(self.lstm.b_i)

            self.W_f = dynet.parameter(self.lstm.W_f)
            self.b_f = dynet.parameter(self.lstm.b_f)

            self.W_c = dynet.parameter(self.lstm.W_c)
            self.b_c = dynet.parameter(self.lstm.b_c)

            self.W_o = dynet.parameter(self.lstm.W_o)
            self.b_o = dynet.parameter(self.lstm.b_o)

        def add_input(self, input_vec):

            x = dynet.concatenate([input_vec, self.h])

            i = dynet.logistic(self.W_i * x + self.b_i)
            f = dynet.logistic(self.W_f * x + self.b_f)
            g = dynet.tanh(self.W_c * x + self.b_c)
            o = dynet.logistic(self.W_o * x + self.b_o)

            c = dynet.cmult(f, self.c) + dynet.cmult(i, g)
            h = dynet.cmult(o, dynet.tanh(c))

            self.c = c
            self.h = h
            self.outputs.append(h)

            return self

        def output(self):
            return self.outputs[-1]

    def initial_state(self):
        return LSTM.State(self)


class Network(object):
    def __init__(
            self,
            lstm_units,
            hidden_units,
            words_size,
            words_dims,
            tag_size,
            span_nums ,
            droprate=0
    ):
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.droprate = droprate
        self.words_size = words_size
        self.words_dims = words_dims
        self.tag_size = tag_size
        self.span_nums = span_nums

        self.model = dynet.Model()
        self.trainer = dynet.AdadeltaTrainer(self.model, eps=1e-7, rho=0.99)
        random.seed(1)

        self.activation = dynet.rectify

        self.words_embed = self.model.add_lookup_parameters(
            (self.words_size,self.words_dims),
            dynet.UniformInitializer(0.01)
        )
        self.fwd_lstm1 = LSTM(self.words_dims,
                              self.lstm_units,self.model)
        self.back_lstm1 =LSTM(self.words_dims,
                              self.lstm_units,self.model)
        self.fwd_lstm2 = LSTM(2*self.lstm_units,
                              self.lstm_units,self.model)
        self.back_lstm2 = LSTM(2*self.lstm_units,
                               self.lstm_units,self.model)

        self.hidden_W = self.model.add_parameters(
            (self.hidden_units,2 * self.span_nums * self.lstm_units),
            dynet.UniformInitializer(0.01)
        )
        self.hidden_b = self.model.add_parameters(
            (self.hidden_units,),
            dynet.ConstInitializer(0)
        )
        self.out_W = self.model.add_parameters(
            (self.tag_size,self.hidden_units),
            dynet.UniformInitializer(0.01)
        )
        self.out_b = self.model.add_parameters(
            (self.tag_size,),
            dynet.ConstInitializer(0)
        )





    def init_params(self):

        self.words_embed.init_from_array(
            np.random.uniform(-0.01,0.01,self.words_embed.shape()),
        )

    def prep_params(self):

        self.W1 = dynet.parameter(self.hidden_W)
        self.b1 = dynet.parameter(self.hidden_b)

        self.W2 = dynet.parameter(self.out_W)
        self.b2 = dynet.parameter(self.out_b)


    def evaluate_recurrent(self,words,test=False):
        fwd1 = self.fwd_lstm1.initial_state()
        back1 = self.back_lstm1.initial_state()

        fwd2 = self.fwd_lstm2.initial_state()
        back2 = self.back_lstm2.initial_state()

        sentence = []
        for w in words:
            wordvec = dynet.lookup(self.words_embed,w)
            sentence.append(wordvec)

        fwd1_out = []
        for vec in sentence:
            fwd1 = fwd1.add_input(vec)
            fwd_vec = fwd1.output()
            fwd1_out.append(fwd_vec)

        back1_out = []
        for vec in reversed(sentence):
            back1 = back1.add_input(vec)
            back_vec = back1.output()
            back1_out.append(back_vec)

        lstm2_input = []
        for (f,b) in zip(fwd1_out,reversed(back1_out)):
            lstm2_input.append(dynet.concatenate([f,b]))

        fwd2_out = []
        for vec in lstm2_input:
            if self.droprate > 0 and not test :
                vec = dynet.dropout(vec,self.droprate)
            fwd2 = fwd2.add_input(vec)
            fwd_vec = fwd2.output()
            fwd2_out.append(fwd_vec)

        back2_out = []
        for vec in reversed(lstm2_input):
            if self.droprate > 0 and not test:
                vec = dynet.dropout(vec, self.droprate)
            back2 = back2.add_input(vec)
            back_vec = back2.output()
            back2_out.append(back_vec)

        return fwd2_out,back2_out[::-1]



    def evaluate_tags(self,fwd_out,back_out,lefts,rights,test=False):

        fwd_span_out = []
        for left_index, right_index in zip(lefts,rights):
            fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index -1 ])
        fwd_span_vec = dynet.concatenate(fwd_span_out)

        back_span_out = []
        for left_index,right_index in zip(lefts,rights):
            back_span_out.append(back_out[left_index] - back_out[right_index + 1])
        back_span_vec = dynet.concatenate(back_span_out)

        hidden_input = dynet.concatenate([fwd_span_vec,back_span_vec])

        if self.droprate > 0 and not test:
            hidden_input = dynet.dropout(hidden_input,self.droprate)

        hidden_output = self.activation(self.W1 * hidden_input + self.b1)

        scores = (self.W2 * hidden_output + self.b2)

        return scores
            
        

    def save(self,filename):

        self.model.save(filename)

        with open (filename,'a') as f:
            f.write('\n')
            f.write('words_size = {}\n'.format(self.words_size))
            f.write('words_dims = {}\n'.format(self.words_dims))
            f.write('lstm_units = {}\n'.format(self.lstm_units))
            f.write('hidden_units = {}\n'.format(self.hidden_units))
            f.write('tag_size = {}\n'.format(self.tag_size))
            f.write('span_nums = {}\n'.format(self.span_nums))

    @staticmethod
    def load(filename):
        """
        Loads file created by save() method
        """

        with open(filename) as f:
            f.readline()
            f.readline()
            words_size = int(f.readline().split()[-1])
            words_dims = int(f.readline().split()[-1])
            lstm_units = int(f.readline().split()[-1])
            hidden_units = int(f.readline().split()[-1])
            tag_size = int(f.readline().split()[-1])
            span_nums = int(f.readline().split()[-1])

        network = Network(
            lstm_units=lstm_units,
            hidden_units=hidden_units,
            words_size=words_size,
            words_dims=words_dims,
            tag_size=tag_size,
            span_nums=span_nums,
        )
        network.model.load(filename)

        return network

    @staticmethod
    def train(
        corpus,
        words_dims,
        lstm_units,
        hidden_units,
        epochs,
        batch_size,
        train_data_file,
        dev_data_file,
        model_save_file,
        droprate,
        unk_params,
        alpha,
    ):

        start_time = time.time()

        fm = corpus
        words_size = corpus.total_words()
        tag_size = corpus.total_tags()
        span_nums = corpus.total_span_nums()

        network = Network(
            words_size=words_size,
            words_dims=words_dims,
            lstm_units = lstm_units,
            hidden_units = hidden_units,
            span_nums=span_nums,
            tag_size=tag_size,
            droprate=droprate,
        )

        #network.init_params()


        print('Hidden units : {} ,per LSTM units : {}'.format(
            hidden_units,
            lstm_units,
        ))

        print('Embeddings: words = {}'.format(
            (words_size,words_dims)
        ))

        print('Dropout rate : {}'.format(droprate))
        print('Parameters initialized in [-0.01,0.01]')
        print ('Random UNKing parameter z = {}'.format(unk_params))

        training_data = corpus.gold_data_from_file(train_data_file)
        num_batched = -(-len(training_data) // batch_size)
        print('Loaded {} training sentences ({} batches of size {})!'.format(
            len(training_data),
            num_batched,
            batch_size,)
        )

        parse_every = -(-num_batched // 4)

        dev_sentences = Sentence.load_sentence_file(dev_data_file,mode='pos-tagging')
        print('Loaded {} validation sentences!'.format(len(dev_sentences)))


        best_acc =Accuracy()
        for epoch in xrange(1,epochs + 1):
            print ('............ epoch {} ............'.format(epoch))

            total_cost = 0.0
            total_states = 0
            training_acc = Accuracy()

            #np.random.shuffle(training_data)

            for b in xrange(num_batched):
                batch = training_data[(b * batch_size) : (b + 1) * batch_size]

                explore = [
                    Tagger.exploration(
                        example,
                        fm,
                        network,
                        alpha=alpha,
                    ) for example in batch
                ]
                for (_,acc ) in explore:
                    training_acc += acc

                batch = [example for (example, _ ) in explore]

                dynet.renew_cg()
                network.prep_params()

                errors = []
                for example in batch:
                    ## random UNKing ##
                    # for (i,w) in enumerate(example['w']):
                    #     if w <= 2:
                    #         continue
                    #
                    #     w_freq = fm.words_freq_list[w]
                    #     drop_prob = unk_params / (unk_params + w_freq)
                    #     r = np.random.random()
                    #     if r < drop_prob :
                    #         example['w'][i] = 0


                    fwd,back = network.evaluate_recurrent(
                        example['w'],
                        test=False
                    )

                    for (left,right),correct in example['tag_data'].items():
                        # correct = example['label_data'][(left,right)]
                        scores = network.evaluate_tags(fwd,back,left,right)

                        probs = dynet.softmax(scores)
                        loss = -dynet.log(dynet.pick(probs,correct))
                        errors.append(loss)
                    total_states += len(example['tag_data'])

                batch_error = dynet.esum(errors)
                total_cost += batch_error.scalar_value()
                batch_error.backward()
                network.trainer.update()

                mean_cost = total_cost/total_states

                print(
                    '\rBatch {}  Mean Cost {:.4f}  [Train: {}]'.format(
                        b,
                        mean_cost,
                        training_acc,
                    ),
                    end='',
                )
                sys.stdout.flush()

                if ((b+1) % parse_every) == 0 or b == (num_batched - 1):
                    dev_acc =Tagger.evaluate_corpus(
                        dev_sentences,
                        fm,
                        network,
                    )
                    print (' [Val: {}]'.format(dev_acc))

                    if dev_acc.precision() >best_acc.precision():
                        best_acc = dev_acc
                        network.save(model_save_file)
                        print('    [saved model : {}]'.format(model_save_file))

            current_time = time.time()
            runmins = (current_time - start_time)/60
            print(' Elapsed time: {:.2f}m'.format(runmins))

        return network
