from gensim.models.doc2vec import Doc2Vec

from keras.models import Model, load_model
from keras.layers import (
    Input,
    Activation,
    Dense
)
from keras.optimizers import Adam
from keras.constraints import maxnorm

from medline.data import (
    read_train_labels,
    read_val_texts,
    read_val_labels,
    read_test_texts,
    read_test_labels,
)
from medline.common import train_model as common_train
from medline.common import test_model as common_test


def train(doc2vec, dest):
    X_train = iter(doc2vec.docvecs)
    Y_train = read_train_labels()
    X_val = read_val_texts()
    Y_val = read_val_labels()

    X_val = infer_texts(doc2vec, X_val)
    nb_features = doc2vec.vector_size
    nb_classes = Y_train.shape[1]

    model = get_model(nb_classes, nb_features)
    train_gen = (X_train, Y_train)
    val_gen = (X_val, Y_val)
    common_train(dest, model, train_gen, val_gen)

    return model


def infer_texts(docvec, texts):
    for text in texts:
        yield docvec.infer_vector(text.split())


def get_model(nb_classes, nb_features):
    inputs = Input(shape=(nb_features,))
    x = inputs

    x = Dense(nb_classes, kernel_constraint=maxnorm(3))(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam())

    return model


def test(model, docvec, scores_filepath):
    X_test = read_test_texts()
    Y_test = read_test_labels()

    X_test = infer_texts(docvec, X_test)

    common_test(model, X_test, Y_test, scores_filepath)


def main():
    import argparse
    from os import path

    parser = argparse.ArgumentParser()
    parser.add_argument('docvec')
    parser.add_argument('--dest', default='./')
    args = parser.parse_args()

    doc2vec = Doc2Vec.load(args.docvec)

    if path.isfile(args.dest):
        model = load_model(args.dest)
        model_dir = path.dirname(args.model)
    else:
        model = train(doc2vec, args.dest)
        model_dir = args.dest

    scores_filepath = path.join(model_dir, 'scores.tsv')

    doc2vec.delete_temporary_training_data(keep_doctags_vectors=False,
                                           keep_inference=True)
    test(model, doc2vec, scores_filepath)


if __name__ == '__main__':
    main()
