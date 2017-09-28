# Chinese Segmenter 

### Required dependency

    * Python 2.7
    * NumPy
    * [DyNet]

### Vocabulary files

Vocabulary may be loaded every time from a training sentence file, or it may be loaded from a JSON file, which is much faster. To learning the vocabulary from a training sentence file, try the command as following:
```
    python src/main.py --train data/ctb/ctb.train.seg.append --write-vocab data/vocab.json
```  

### Training

Trainging requires a file containing training sentences (`--train`) and a file containing validation sentence (`--dev`), which are parsed four times per training epoch to determine which model to keep. A file name must also be provided to store the saved model (`--model`). The following is an example of a command to train a model with all of the default settings:
```
    python src/main.py --train data/ctb/ctb.train.seg.append --dynet-mem 6000 --dev data/ctb/ctb.dev.seg.append --vocab data/vocab.json --model data/my_model --epoch 10
```

The following table provides an overview of additional training options:

Argument | Description | Default
--- | --- | ---
--dynet-mem | Memory (MB) to allocate for DyNet | 2000
--dynet-l2  | L2 regularization factor | 0
--dynet-seed | Seed for random parameter initialization | random
--bigrams-dims | Word embedding dimensions | 50
--unigrams-dims  | POS embedding dimensions  | 20
--lstm-units | LSTM units (per direction, for each of 2 layers) | 200
--hidden-units | Units for ReLU FC layer (each of 2 action types) | 200
--epochs | Number of training epochs | 10
--batch-size | Number of sentences per training update | 10
--droprate | Dropout probability | 0.5
--unk-param | Parameter z for random UNKing | 0.8375
--np-seed | Seed for shuffling and softmax sampling | random


#### Test Evaluation

There is also a facility to directly evaluate a model agaist a reference corpus, by supplying the `--test` argument:

```
python src/main.py --dynet-mem 4000 --test data/ctb/ctb.test.seg.append --vocab data/vocab.json --model data/my_model
```