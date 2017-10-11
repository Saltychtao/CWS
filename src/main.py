

import sys
import argparse

from seg_sentence import Sentence

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Chinese Word Segmenter')
    parser.add_argument(
        '--dynet-mem',
        dest='dynet_mem',
        help='Memory allocation for Dynet. (DEFAULT=2000)',
        default=4000,
    )
    parser.add_argument(
        '--dynet-l2',
        dest='dynet_l2',
        help='L2 regularization parameter. (DEFAULT=0)',
        default=0,
    )
    parser.add_argument(
        '--dynet-seed',
        dest='dynet_seed',
        help='Seed for PNG.(DEFAULT=0 : generate)',
        default=0,
    )
    parser.add_argument(
        '--model',
        dest='model',
        help='File to save or load model',
    )
    parser.add_argument(
        '--train',
        dest='train',
        help='Training sentences.'
    )
    parser.add_argument(
        '--test',
        dest='test',
        help=(
            'Evaluation sentences.'
            'Omit for training.'
        )
    )
    parser.add_argument(
        '--dev',
        dest='dev',
        help=(
            'Validation trees.'
            'Required for training'
        )
    )
    parser.add_argument(
        '--vocab',
        dest='vocab',
        help='JSON file from which to load vocabulary',
    )
    parser.add_argument(
        '--write-vocab',
        dest='vocab_output',
        help='Destination to save vocabulary from training datat'
    )
    parser.add_argument(
        '--bigram-dims',
        dest='bigrams_dims',
        type = int,
        default = 50,
        help = 'Embedding dimesions for word forms. (DEFAULT=50)',
    )
    parser.add_argument(
        '--unigram-dims',
        dest = 'unigrams_dims',
        type=int,
        default=50
    )
    parser.add_argument(
        '--word-dims',
        dest='word_dims',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--lstm-units',
        dest='lstm_units',
        type = int,
        default=100,
        help='Number of LSTM units in each layer/direction. (DEFAULT=200)'
    )
    parser.add_argument(
        '--hidden-units',
        dest='hidden_units',
        type = int,
        default = 100,
        help='Number of hidden units. (DEFAULT=50)'
    )
    parser.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=1,
        help='Number of training epochs. (DEFAULT=10)'
    )
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        type = int ,
        default = 10,
        help='Number of sentences per training updatexte. (DEFAULT=10'
    )
    parser.add_argument(
        '--droprate',
        dest='droprate',
        type = float,
        default=0.5,
        help='Dropout probability. (DEFAULT=0.5)',
    )
    parser.add_argument(
        '--unk-param',
        dest='unk_param',
        type = float,
        default = 0.875
    )
    parser.add_argument(
        '--np-seed',
        type=int,
        dest='np_seed'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        dest='alpha',
        default=0.0
    )
    parser.add_argument(
        '--beta',
        type=float,
        dest='beta',
        default=0
    )

    parser.add_argument(
        '--mode',
        type = str,
        dest = 'mode',
        help= 'Use one of \'cws\' , \'pos-taggin\' , \'joint\''
    )
    args = parser.parse_args()

    # Overriding DyNet defaults
    sys.argv.insert(1,str(args.dynet_mem))
    sys.argv.insert(1,'--dynet-mem')
    sys.argv.insert(1,str(args.dynet_l2))
    sys.argv.insert(1,'--dynet-l2')
    sys.argv.insert(1,str(args.dynet_seed))
    sys.argv.insert(1,'--dynet-seed')

    if (args.mode is None ) :
        print('Must specify --mode parameter.')
        print('    (Use -h or --help flag for full option list.')

    if not (args.mode  in ['cws','POS-tagging','joint']):
        print('--mode parameter must be either ')

    if args.vocab is not None:
        if args.mode == 'CWS':
            from Corpus import Corpus
            fm = Corpus.load_json(args.vocab)
        elif args.mode == 'POS-tagging':
            from Corpus import PosCorpus
            fm = PosCorpus.load_json(args.vocab)

    elif args.train is not None:
        fm = None
        if args.mode == 'CWS':
            from  Corpus import Corpus
            fm = Corpus(args.train)
        elif args.mode == 'POS-tagging':
            from Corpus import PosCorpus
            fm = PosCorpus(args.train)

            if args.vocab_output is not None:
                fm.save_json(args.vocab_output)
                print('Wrote vocabulary file {}'.format(args.vocab_output))
                sys.exit()
    else:
        print('Must specify either --vocab_file or --traing-data.')
        print('    (Use -h or --help flag for full option list.)')
        sys.exit()

    if args.model is None:
        print('Must specify --model or (or --write-vocab) parameter.')
        print('    (Use -h or --help flag for full option list.)')
        sys.exit()



    if args.test is not None:
        test_sentence = Sentence.load_sentence_file(args.test,args.mode)
        print('Loaded test trees from {}'.format(args.test))
        if args.mode == 'cws':
            from network import Network
            from Segmenter import Segmenter
            network = Network.load(args.model)
            print('Loaded CWS model from: {}'.format(args.model))
            fscore = Segmenter.evaluate_corpus(test_sentence,fm,network)
            #Segmenter.write_predicted('data/predicted',test_sentence,fm,network)
            print('F-Score : {}'.format(fscore))

        elif args.mode == 'POS-tagging':
            from tagger_network import Network
            from Tagger import Tagger
            network = Network.load(args.model)
            print('Loaded POS-tagging model from: {}'.format(args.model))
            accuracy = Tagger.evaluate_corpus(test_sentence,fm,network)
            print('Accuracy : {}'.format(accuracy))

    elif args.train is not None:


        if args.np_seed is not None:
            import numpy as np
            np.random.seed(args.np_seed)
        if args.mode == 'CWS':
            from network import Network
            network = Network.train(
                corpus = fm,
                bigrams_dims = args.bigrams_dims,
                unigrams_dims = args.unigrams_dims,
                lstm_units = args.lstm_units,
                hidden_units = args.hidden_units,
                epochs = args.epochs,
                batch_size = args.batch_size,
                train_data_file = args.train,
                dev_data_file = args.dev,
                model_save_file = args.model,
                droprate=args.droprate,
                unk_params=args.unk_param,
                alpha=args.alpha,
                beta=args.beta
            )
        elif args.mode == 'POS-tagging':
            from tagger_network import Network
            Network.train(
                corpus=fm,
                words_dims=args.word_dims,
                lstm_units=args.lstm_units,
                hidden_units=args.hidden_units,
                epochs=args.epochs,
                batch_size=args.batch_size,
                train_data_file=args.train,
                dev_data_file=args.dev,
                model_save_file=args.model,
                droprate=args.droprate,
                unk_params=args.unk_param,
                alpha=args.alpha
            )
