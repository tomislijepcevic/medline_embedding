def get_embedder(model):
    from keras import backend as K
    from ..data import read_words
    from ..convnet import get_word_index, vectorize_texts
    from ..common import to_batches

    maxlen = model.input_shape[1]
    words = read_words()
    word_index = get_word_index(words)

    get_penultimate_layer_output = K.function(
        [model.input],
        [model.layers[-3].output])

    def embedder(texts):
        X = vectorize_texts(texts, word_index, maxlen)

        for batch in to_batches(X, 50):
            yield from get_penultimate_layer_output([batch])[0]

    return embedder


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('data')
    parser.add_argument('dest')
    args = parser.parse_args()

    import glob
    from os import path
    from keras.models import load_model
    import pandas as pd

    model = load_model(args.model)
    embedder = get_embedder(model)

    for group_filepath in glob.glob(path.join(args.data, '*.csv')):
        filepath = path.join(args.dest, path.basename(group_filepath))

        if path.isfile(filepath):
            continue

        df = pd.read_csv(group_filepath)
        texts, terms = df['text'], df['term']

        X = list(embedder(texts))
        df = pd.DataFrame(X)
        df['y'] = terms

        df.to_csv(filepath, index=False)


if __name__ == '__main__':
    main()
