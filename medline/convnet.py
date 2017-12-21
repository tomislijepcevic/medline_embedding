import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from keras.models import Model, load_model
from keras.layers import (
    Input,
    Activation,
    Dropout,
    Flatten,
    Dense,
    BatchNormalization
)
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras import constraints
from keras.optimizers import Adam

from medline.common import train_model as common_train
from medline.common import test_model as common_test

from medline.data import (
    read_train_texts,
    read_train_labels,
    read_val_texts,
    read_val_labels,
    read_test_texts,
    read_test_labels,
    read_words,
)


MAXLEN = 500
NB_WORD_FEATURES = 300


def train(name=None, wordvecs=None, **model_params):
    X_train = read_train_texts()
    Y_train = read_train_labels()
    X_val = read_val_texts()
    Y_val = read_val_labels()
    words = read_words()

    word_index = get_word_index(words)
    nb_words = len(word_index) + 1
    word_matrix = None

    if wordvecs is not None:
        word_matrix = get_word_matrix(words, NB_WORD_FEATURES,
                                      word_index, wordvecs)

    X_train = vectorize_texts(X_train, word_index, MAXLEN)
    X_val = vectorize_texts(X_val, word_index, MAXLEN)
    nb_classes = Y_train.shape[1]

    model = get_model(nb_classes, MAXLEN, nb_words, NB_WORD_FEATURES,
                      wv=word_matrix, **model_params)

    if name is not None:
        model.name = name + '_' + model.name

    model_dir = model.name
    train_gen = (X_train, Y_train)
    val_gen = (X_val, Y_val)
    common_train(model_dir, model, train_gen, val_gen)

    return model, model_dir


def get_word_index(words):
    return {word: index + 1 for index, word in enumerate(words)}


def get_word_matrix(words, nb_features, word_index, word_vecs):
    word_mat = np.zeros((len(words) + 1, nb_features))

    for word in words:
        if word in word_vecs:
            word_mat[word_index[word]] = word_vecs[word]

    return word_mat


def vectorize_texts(texts, word_index, maxlen):
    for text in texts:
        words = text.split()
        sequence = [0] * maxlen

        l = 0
        for word in words:
            index = word_index.get(word, None)

            if index is None:
                continue

            sequence[l] = index
            l += 1

            if l == maxlen:
                break

        yield sequence


def get_model(nb_classes, maxlen, nb_words, nb_word_features,
              wv=None, wv_static=False, wv_dropout_p=0,
              nb_filters=100, filter_lengths=(3, 4, 5),
              conv_bnorm=False, pool_size=1, kmax_pool=None,
              activation='relu', pool_dropout_p=.5,
              pool_bnorm=False, maxnorm=None):
    inputs = Input(shape=(maxlen,))
    x = inputs

    if wv is not None:
        x = Embedding(nb_words, nb_word_features,
                      input_length=maxlen, weights=[wv],
                      trainable=not wv_static)(x)
    else:
        x = Embedding(nb_words, nb_word_features,
                      input_length=maxlen)(x)

    if wv_dropout_p > 0:
        x = Dropout(wv_dropout_p)(x)

    conv_list = []
    layer = x

    for filter_length in filter_lengths:
        x = Conv1D(nb_filters, filter_length)(layer)

        if conv_bnorm:
            x = BatchNormalization()(x)

        if activation is not None:
            x = Activation(activation)(x)

        # if kmax_pool is not None and kmax_pool > 1:
        #     x = KMaxPooling(kmax_pool)(x)
        # elif pool_size == 1:
        if pool_size == 1:
            x = GlobalMaxPooling1D()(x)
        else:
            x = MaxPooling1D(pool_size=pool_size)(x)
            x = Flatten()(x)

        conv_list.append(x)

    if len(conv_list) > 1:
        x = concatenate(conv_list)
    else:
        x = conv_list[0]

    if pool_dropout_p > 0:
        x = Dropout(pool_dropout_p)(x)

    if pool_bnorm:
        x = BatchNormalization()(x)

    kernel_constraint = constraints.maxnorm(maxnorm) if maxnorm > 0 else None
    x = Dense(nb_classes, kernel_constraint=kernel_constraint)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam())

    model.name = '_'.join([
        'fl=%d' % nb_filters,
        'lens=%s' % '-'.join(map(str, filter_lengths)),
        'act=%s' % activation,
        'pool_s=%d' % pool_size,
        'kmax=%s' % str(kmax_pool),
        'wv_d=%.1f' % wv_dropout_p,
        'pool_d=%.1f' % pool_dropout_p,
        'conv_bn=%d' % conv_bnorm,
        'pool_bn=%d' % pool_bnorm,
        'maxnorm=%d' % maxnorm,
    ])

    return model


def test(model, scores_filepath):
    X_test = read_test_texts()
    Y_test = read_test_labels()

    words = read_words()
    word_index = get_word_index(words)
    maxlen = model.input_shape[1]

    X_test = vectorize_texts(X_test, word_index, maxlen)

    common_test(model, X_test, Y_test, scores_filepath)


def main():
    import argparse
    from os import path

    parser = argparse.ArgumentParser()
    parser.add_argument('--len', dest='filter_lengths',
                        metavar='N', type=int, nargs='+')
    parser.add_argument('--fl', dest='nb_filters', metavar='N', type=int)
    parser.add_argument('--act', dest='activation', metavar='N')
    parser.add_argument('--kmax', dest='kmax_pool', metavar='N', type=int)
    parser.add_argument('--pool_d', dest='pool_dropout_p',
                        metavar='N', type=float)
    parser.add_argument('--wv', dest='wordvec', metavar='N')
    parser.add_argument('--wv_static', action='store_true')
    parser.add_argument('--conv_bnorm', metavar='N', type=bool, default=False)
    parser.add_argument('--pool_bnorm', metavar='N', type=bool, default=False)
    parser.add_argument('--maxnorm', metavar='N', type=int, default=3)
    parser.add_argument('--model', metavar='N')
    args = parser.parse_args()

    if args.wordvec is not None:
        args.word_vecs = KeyedVectors.load_word2vec_format(args.wordvec)
        args.name = path.basename(args.wordvec)

        if args.wv_static:
            args.name = 'static_' + args.name

    args = vars(args)

    model_filepath = args.pop('model')

    if not path.isfile(model_filepath):
        model = load_model(model_filepath)
        model_dir = path.dirname(model_filepath)
    else:
        model, model_dir = train(**args)

    scores_filepath = path.join(model_dir, 'scores.tsv')
    test(model, scores_filepath)


if __name__ == '__main__':
    main()
