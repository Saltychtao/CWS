# TODO implement tree lstm class mainly modify the self.c and self.h
from __future__ import print_function

import time
import numpy as np
import dynet
import random
import sys

from seg_sentence import SegSentence
from seg_sentence import FScore
from Segmenter import Segmenter


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
            # print(input_vec.dim())
            x = dynet.concatenate([input_vec, self.h])
            # print(self.h.dim())
            #print(self.W_i.dim())
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


class TreeLSTM(object):
    def __init__(self, input_dims, output_dims, model):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.model = model

        self.W_i = model.add_parameters(
            (output_dims, input_dims),
            init=dynet.UniformInitializer(0.01)
        )
        self.b_i = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0)
        )
        self.W_f = model.add_parameters(
            (output_dims, input_dims),
            init=dynet.UniformInitializer(0.01)
        )
        self.b_f = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )
        self.W_u = model.add_parameters(
            (output_dims, input_dims),
            init=dynet.UniformInitializer(0.01)
        )
        self.b_u = model.add_parameters(
            (output_dims)
        )
        self.W_o = model.add_parameters(
            (output_dims, input_dims),
            init=dynet.UniformInitializer(0.01),

        )
        self.b_o = model.add_parameters(
            (output_dims),
        )
        self.c0 = model.add_parameters(
            (output_dims),
            init=dynet.ConstInitializer(0)
        )

        self.U_i = model.add_parameters(
            (output_dims, output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.U_f_left = model.add_parameters(
            (output_dims, output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.U_f_right = model.add_parameters(
            (output_dims, output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.U_o = model.add_parameters(
            (output_dims, output_dims),
            init=dynet.UniformInitializer(0.01),
        )

        self.U_u = model.add_parameters(
            (output_dims, output_dims),
            init=dynet.UniformInitializer(0.01),
        )

        self.W_params = [self.W_i, self.W_f, self.W_o, self.W_u]
        self.b_params = [self.b_i, self.b_f, self.b_o, self.b_u]
        self.U_params = [self.U_i, self.U_f_left, self.U_f_right, self.U_o, self.U_u]
        self.params = self.W_params + self.b_params + self.U_params + [self.c0]

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

            self.W_u = dynet.parameter(self.lstm.W_u)
            self.b_u = dynet.parameter(self.lstm.b_u)

            self.W_o = dynet.parameter(self.lstm.W_o)
            self.b_o = dynet.parameter(self.lstm.b_o)

            self.U_i = dynet.parameter(self.lstm.U_i)
            self.U_f_left = dynet.parameter(self.lstm.U_f_left)
            self.U_f_right = dynet.parameter(self.lstm.U_f_right)
            self.U_o = dynet.parameter(self.lstm.U_o)
            self.U_u = dynet.parameter(self.lstm.U_u)

        def add_input(self, left_node=None, right_node=None, input_vec=None):

            if input_vec is None:
                left_h = left_node.h
                left_c = left_node.c
                right_h = right_node.h
                right_c = right_node.c
                i = dynet.logistic(self.U_i * left_h + self.U_i * right_h + self.b_i)
                f_left = dynet.logistic(self.U_f_left * left_h + self.U_f_left * right_h + self.b_f)
                f_right = dynet.logistic(self.U_f_right * left_h + self.U_f_right * right_h + self.b_f)
                o = dynet.logistic(self.U_o * left_h + self.U_o * right_h + self.b_o)
                u = dynet.tanh(self.U_u * left_h + self.U_u * right_h + self.b_u)

                c = dynet.cmult(f_left, left_c) + dynet.cmult(f_right, right_c)
                h = dynet.cmult(o, dynet.tanh(c))

            else:
                x = input_vec
                left_h = self.h
                right_h = self.h
                left_c = self.c
                right_c = self.c

                i = dynet.logistic(self.W_i * x + self.U_i * left_h + self.U_i * right_h + self.b_i)
                f_left = dynet.logistic(self.W_f * x + self.U_f_left * left_h + self.U_f_left * right_h + self.b_f)
                f_right = dynet.logistic(self.W_f * x + self.U_f_right * left_h + self.U_f_right * right_h + self.b_f)
                o = dynet.logistic(self.W_o * x + self.U_o * left_h + self.U_o * right_h + self.b_o)
                u = dynet.tanh(self.W_u * x + self.U_u * left_h + self.U_u * right_h + self.b_u)

                c = dynet.cmult(i, u) + dynet.cmult(f_left, left_c) + dynet.cmult(f_right, right_c)
                h = dynet.cmult(o, dynet.tanh(c))

            self.c = c
            self.h = h
            self.outputs.append(h)

            return self

        def output(self):
            return self.outputs[-1]

    def initial_state(self):
        return TreeLSTM.State(self)

class Network(object):
    def __init__(
            self,
            bigrams_size,
            unigrams_size,
            bigrams_dims,
            unigrams_dims,
            lstm_units,
            hidden_units,
            label_size,
            span_nums,
            droprate=0,  # self.embed2lstm_W = dynet.parameter(self.p_embed2lstm_W)
            # self.embed2lstm_b = dynet.parameter(self.p_embed2lstm_b)
    ):

        self.bigrams_size = bigrams_size
        self.bigrams_dims = bigrams_dims
        self.unigrams_dims = unigrams_dims
        self.unigrams_size = unigrams_size
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.span_nums =span_nums
        self.droprate = droprate
        self.label_size = label_size

        self.model = dynet.Model()
        self.trainer = dynet.AdadeltaTrainer(self.model, eps=1e-7, rho=0.99)
        #self.trainer = dynet.AdagradTrainer(self.model)
        random.seed(1)

        self.activation = dynet.rectify


        self.bigram_embed = self.model.add_lookup_parameters(
            (self.bigrams_size, self.bigrams_dims),
        )
        self.unigram_embed = self.model.add_lookup_parameters(
            (self.unigrams_size, self.unigrams_dims),
        )
        self.fwd_lstm1 = LSTM(self.bigrams_dims + self.unigrams_dims,
                             self.lstm_units, self.model)
        self.back_lstm1 = LSTM(self.bigrams_dims + self.unigrams_dims,
                              self.lstm_units, self.model)

        # self.fwd_lstm2 = LSTM(2*self.lstm_units,self.lstm_units,self.model)
        # self.back_lstm2 = LSTM(2 * self.lstm_units, self.lstm_units, self.model)

        self.upward_lstm = TreeLSTM(self.lstm_units, self.lstm_units, self.model)

        # self.p_upward_W = self.model.add_parameters(
        #     (self.lstm_units, 2 * self.lstm_units),
        #     dynet.UniformInitializer(0.01)
        # )
        #
        # self.p_upward_b = self.model.add_parameters(
        #     (self.lstm_units),
        #     dynet.ConstInitializer(0)
        # )

        #self.update_lstm = LSTM(2*self.lstm_units,self.lstm_units,self.model)


        self.p_hidden_W = self.model.add_parameters(
            (self.hidden_units, 2 * self.lstm_units),
            dynet.UniformInitializer(0.01)
        )
        self.p_hidden_b = self.model.add_parameters(
            (self.hidden_units,),
            dynet.ConstInitializer(0)
        )
        self.p_output_W = self.model.add_parameters(
            (self.label_size, self.hidden_units),
            dynet.ConstInitializer(0)
        )
        self.p_output_b = self.model.add_parameters(
            (self.label_size,),
            dynet.ConstInitializer(0)
        )



    def init_params(self):


        self.bigram_embed.init_from_array(
            np.random.uniform(-0.01, 0.01, self.bigram_embed.shape())
        )
        self.unigram_embed.init_from_array(
            np.random.uniform(-0.01, 0.01, self.unigram_embed.shape())
        )

    def prep_params(self):
        self.hidden_W = dynet.parameter(self.p_hidden_W)
        self.hidden_b = dynet.parameter(self.p_hidden_b)
        self.output_W = dynet.parameter(self.p_output_W)
        self.output_b = dynet.parameter(self.p_output_b)

        # self.upward_W = dynet.parameter(self.p_upward_W)
        #self.upward_b = dynet.parameter(self.p_upward_b)
        # self.update = self.update_lstm.initial_state()
        self.encoding = {}
        self.lstm_states = {}

    def prep_states(self, n0):
        for i in range(1, n0 + 1):
            self.lstm_states[(i, i)] = self.upward_lstm.initial_state()

    def evaluate_recurrent(self, fwd_bigrams,unigrams, test=False):
        fwd1 = self.fwd_lstm1.initial_state()
        back1 = self.back_lstm1.initial_state()

        # fwd2 = self.fwd_lstm2.initial_state()
        #back2 = self.back_lstm2.initial_state()

        fwd_input = []
        for i in range(len(unigrams)):
            bivec = dynet.lookup(self.bigram_embed,fwd_bigrams[i])
            univec = dynet.lookup(self.unigram_embed,unigrams[i])
            vec = dynet.concatenate([bivec,univec])
         #   fwd_input.append(dynet.tanh(self.embed2lstm_W*vec))
            fwd_input.append(vec)
            
        back_input = []
        for i in range(len(unigrams)):
            bivec = dynet.lookup(self.bigram_embed,fwd_bigrams[i+1])
            univec = dynet.lookup(self.unigram_embed,unigrams[i])
            vec = dynet.concatenate([bivec,univec])
           # back_input.append(dynet.tanh(self.embed2lstm_W*vec))
            back_input.append(vec)
            
        fwd1_out = []
        for vec in fwd_input:
            fwd1 = fwd1.add_input(vec)
            fwd_vec = fwd1.output()
            fwd1_out.append(fwd_vec)

        back1_out = []
        for vec in reversed(back_input):
            back = back1.add_input(vec)
            back1_vec = back.output()
            back1_out.append(back1_vec)

        # lstm2_input = []
        # for (f,b) in zip(fwd1_out,reversed(back1_out)):
        #     lstm2_input.append(dynet.concatenate([f,b]))
        #
        # fwd2_out = []
        # for vec in lstm2_input:
        #     if self.droprate > 0 and not test:
        #         vec = dynet.dropout(vec,self.droprate)
        #     fwd2 = fwd2.add_input(vec)
        #     fwd_vec = fwd2.output()
        #     fwd2_out.append(fwd_vec)
        #
        # back2_out = []
        # for vec in reversed(lstm2_input):
        #     if self.droprate > 0 and not test:
        #         vec = dynet.dropout(vec,self.droprate)
        #     back2 = back2.add_input(vec)
        #     back_vec = back2.output()
        #     back2_out.append(back_vec)

        # fwd_out = [dynet.concatenate([f1,f2]) for (f1,f2) in zip(fwd1_out,fwd2_out)]
        # back_out = [dynet.concatenate([b1,b2]) for (b1,b2) in zip(back1_out,back2_out)]

        fwd = fwd1_out
        back = back1_out[::-1]
        n0 = len(unigrams) - 2
        self.prep_states(n0)
        base_spans = [(i + 1, i + 1) for i in range(n0)]
        for span in base_spans:
            left, right = span
            fwd_vec = fwd[right] - fwd[left - 1]
            back_vec = back[left] - back[right + 1]
            init_embedding = (fwd_vec + back_vec) / 2
            self.lstm_states[span] = self.lstm_states[span].add_input(input_vec=init_embedding)
            self.encoding[span] = self.lstm_states[span].output()
    def update_encoding(self, left_span, right_span):
        left_span_embedding = self.encoding[left_span]
        self.encoding.pop(left_span)
        right_span_embedding = self.encoding[right_span]
        self.encoding.pop(right_span)
        # for k in self.encoding.keys():
        #     embedding = self.encoding[k]
        #     self.lstm_states[k] = self.lstm_states[k].add_inp
        # ut((dynet.concatenate([embedding,embedding])))
        #     self.encoding[k] = self.lstm_states[k].output()

        # embedding = self.encoding[k]
        #new_embedding = self.upward_W * dynet.concatenate([embedding, embedding]) + self.upward_b

        # self.upwd = self.upwd.add_input(dynet.concatenate([left_span_embedding,right_span_embedding]))
        ## create new node and compute its state
        new_span = (left_span[0], right_span[1])
        new_state = self.upward_lstm.initial_state()
        left_node = self.lstm_states[left_span]
        right_node = self.lstm_states[right_span]
        new_state = new_state.add_input(left_node=left_node, right_node=right_node)

        ## upgrade encoding dict and state dict
        self.lstm_states[new_span] = new_state
        self.encoding[new_span] = self.lstm_states[new_span].output()

        ## clean up old state
        self.lstm_states.pop(left_span)
        self.lstm_states.pop(right_span)

        # self.encoding[new_span] = self.upward_W * dynet.concatenate(
        #     [left_span_embedding, right_span_embedding]) + self.upward_b

    def evaluate_labels(self, left_span, right_span, test=False):

        hidden_input = dynet.concatenate([self.encoding[left_span], self.encoding[right_span]])

        if self.droprate > 0 and not test:
            hidden_input = dynet.dropout(hidden_input,self.droprate)

        hidden_output = self.activation(self.hidden_W * hidden_input + self.hidden_b)
        scores = (self.output_W * hidden_output + self.output_b)

        return scores

    # def evaluate_labels(self,fwd_out,back_out,lefts,rights,test=False):
    #     #fwd_out_value = [f.npvalue() for f in fwd_out]
    #     #back_out_value=[b.npvalue() for b in back_out]
    #     fwd_span_out = []
    #     for left_index,right_index in zip(lefts,rights):
    #         fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index-1])
    #     fwd_span_vec = dynet.concatenate(fwd_span_out)
    #     #fwd_span_vec_value = [f.npvalue() for f in fwd_span_vec]
    #
    #     back_span_out = []
    #     for left_index,right_index in zip(lefts,rights):
    #         back_span_out.append(back_out[left_index] - back_out[right_index+1])
    #     back_span_vec = dynet.concatenate(back_span_out)
    #     #back_span_vec_value = [b.npvalue() for f in back_span_vec]
    #
    #     hidden_input = dynet.concatenate([fwd_span_vec,back_span_vec])
    #     #hidden_input_value = hidden_input.npvalue()
    #
    #     if self.droprate >0 and not test:
    #         hidden_input = dynet.dropout(hidden_input,self.droprate)
    #
    #     hidden_output = self.activation(self.hidden_W * hidden_input + self.hidden_b)
    #     #hidden_output_value = hidden_output.npvalue()
    #
    #     #W_value = self.output_W.npvalue()
    #     #b_value = self.output_b.npvalue()
    #
    #     scores = (self.output_W * hidden_output + self.output_b)
    #
    #     #scores_value = scores.npvalue()
    #     return scores
            
        
    def save(self,filename):

        self.model.save(filename)

        with open(filename + '.parameter', 'w') as f:
            f.write('\n')
            f.write('bigrams_size = {}\n'.format(self.bigrams_size))
            f.write('unigrams_size = {}\n'.format(self.unigrams_size))
            f.write('bigrams_dims = {}\n'.format(self.bigrams_dims))
            f.write('unigrams_dims = {}\n'.format(self.unigrams_dims))
            f.write('lstm_units = {}\n'.format(self.lstm_units))
            f.write('hidden_units = {}\n'.format(self.hidden_units))
            f.write('lable_size = {}\n'.format(self.label_size))
            f.write('span_nums = {}\n'.format(self.span_nums))



    @staticmethod
    def load(filename):
        """
        Loads file created by save() method
        """

        with open(filename + '.parameter') as f:
            f.readline()
            bigrams_size = int(f.readline().split()[-1])
            unigrams_size = int(f.readline().split()[-1])
            bigrams_dims = int(f.readline().split()[-1])
            unigrams_dims = int(f.readline().split()[-1])
            lstm_units = int(f.readline().split()[-1])
            hidden_units = int(f.readline().split()[-1])
            label_size = int(f.readline().split()[-1])
            span_nums = int(f.readline().split()[-1])

        network = Network(
            bigrams_size=bigrams_size,
            unigrams_size=unigrams_size,
            bigrams_dims=bigrams_dims,
            unigrams_dims=unigrams_dims,
            lstm_units=lstm_units,
            hidden_units=hidden_units,
            label_size=label_size,
            span_nums=span_nums
        )

        network.model.populate(filename)

        return network

    @staticmethod
    def train(
        corpus,
        bigrams_dims,
        unigrams_dims,
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
        beta
    ):

        start_time = time.time()

        fm = corpus
        bigrams_size = corpus.total_bigrams()
        unigrams_size = corpus.total_unigrams()

        network = Network(
            bigrams_size=bigrams_size,
            unigrams_size=unigrams_size,
            bigrams_dims=bigrams_dims,
            unigrams_dims=unigrams_dims,
            lstm_units = lstm_units,
            hidden_units = hidden_units,
            label_size = fm.total_labels(),
            span_nums = fm.total_span_nums(),
            droprate=droprate,
        )

        network.init_params()


        print('Hidden units : {} ,per LSTM units : {}'.format(
            hidden_units,
            lstm_units,
        ))

        print('Embeddings: bigrams = {}, unigrams = {}'.format(
            (bigrams_size,bigrams_dims),
            (unigrams_size,unigrams_dims)
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

        dev_sentences = SegSentence.load_sentence_file(dev_data_file)
        print('Loaded {} validation sentences!'.format(len(dev_sentences)))


        best_acc =FScore()
        for epoch in xrange(1,epochs + 1):
            print ('............ epoch {} ............'.format(epoch))

            total_cost = 0.0
            total_states = 0
            training_acc = FScore()

            #np.random.shuffle(training_data)

            for b in xrange(num_batched):
                batch = training_data[(b * batch_size) : (b + 1) * batch_size]

                dynet.renew_cg(immediate_compute=True, check_validity=True)
                error = []
                for example in batch:
                    loss,acc = Segmenter.exploration(example,fm,network,droprate,unk_params)
                    #loss = result['loss']
                    #total_states += result['state_cnt']
                    training_acc += acc
                    error.extend(loss)

                    if len(loss) == 0:
                        continue
                if len(error) == 0:
                    continue
                    # else:
                    # print (len(error))
                batch_error = dynet.esum(error)
                # error_value = batch_error.npvalue()
                # print(batch_error.value())
                batch_error.backward()
                network.trainer.update()

                #value = network.output_W.npvalue()
                #print (network.output_b.npvalue())
                #mean_cost = total_cost/total_states

                print(
                    '\rBatch {}  Mean Cost {:.4f}  [Train: {}]'.format(
                        b,
                        0,
                        training_acc,
                    ),
                    end='',
                )
                sys.stdout.flush()

                if ((b+1) % parse_every) == 0 or b == (num_batched - 1):
                    dev_acc = Segmenter.evaluate_corpus(
                        dev_sentences,
                        fm,
                        network,
                    )
                    print (' [Val: {}]'.format(dev_acc))

                    if dev_acc.fscore() >best_acc.fscore():
                        best_acc = dev_acc
                        network.save(model_save_file)
                        print('    [saved model : {}]'.format(model_save_file))

            current_time = time.time()
            runmins = (current_time - start_time)/60
            print(' Elapsed time: {:.2f}m'.format(runmins))

        return network
