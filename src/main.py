
import sys
import argparse

from seg_sentence import SegSentence
from Segmenter import Segmenter

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Chinese Segmenter')
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
        '--lstm-units',
        dest='lstm_units',
        type = int,
        default=200,
        help='Number of LSTM units in each layer/direction. (DEFAULT=200)'
    )
    parser.add_argument(
        '--hidden-units',
        dest='hidden_units',
        type = int,
        default = 50,
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
        help='Number of sentences per training update. (DEFAULT=10'
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
        default=1.0
    )
    parser.add_argument(
        '--beta',
        type=float,
        dest='beta',
        default=0
    )

    args = parser.parse_args()

    # Overriding DyNet defaults
    sys.argv.insert(1,str(args.dynet_mem))
    sys.argv.insert(1,'--dynet-mem')
    sys.argv.insert(1,str(args.dynet_l2))
    sys.argv.insert(1,'--dynet-l2')
    sys.argv.insert(1,str(args.dynet_seed))
    sys.argv.insert(1,'--dynet-seed')

    if args.vocab is not None:
        from Corpus import Corpus
        fm = Corpus.load_json(args.vocab)
    elif args.train is not None:
        from  Corpus import Corpus
        fm = Corpus(args.train)
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
        from seg_sentence import SegSentence
        from network import Network
        from Segmenter import Segmenter

        test_sentence = SegSentence.load_sentence_file(args.test)
        print('Loaded test trees from {}'.format(args.test))
        network = Network.load(args.model)
        print('Loaded model from: {}'.format(args.model))
        accuracy = Segmenter.evaluate_corpus(test_sentence,fm,network)
        #Segmenter.write_predicted('data/predicted',test_sentence,fm,network)
        print('Accuracy: {}'.format(accuracy))

    elif args.train is not None:
        from network import Network
        if args.np_seed is not None:
            import numpy as np
            np.random.seed(args.np_seed)


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
