def get_embedder(model):
    from ..nnet_doc2vec import infer_texts

    def embedder(texts):
        return infer_texts(model, texts)

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
    from gensim.models.doc2vec import Doc2Vec
    import pandas as pd

    model = Doc2Vec.load(args.model)
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
