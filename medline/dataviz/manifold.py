def transform_with_tsne(X):
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=0, init='pca')

    X_tsne = tsne.fit_transform(X)

    return X_tsne


def transform_with_mds(X):
    from sklearn.manifold import MDS

    mds = MDS(n_components=2, random_state=0)

    X_mds = mds.fit_transform(X)

    return X_mds


def transform(X, algorithm='tsne'):

    if algorithm == 'tsne':
        X_manifold = transform_with_tsne(X)
    elif algorithm == 'mds':
        X_manifold = transform_with_mds(X)

    return X_manifold


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings')
    parser.add_argument('algorithm')
    parser.add_argument('dest')
    args = parser.parse_args()

    import glob
    from os import path
    import pandas as pd

    for group_filepath in glob.glob(path.join(args.embeddings, '*.csv')):
        filepath = path.join(args.dest, path.basename(group_filepath))

        if path.isfile(filepath):
            continue

        df = pd.read_csv(group_filepath)
        y = df.pop('y')

        X = df.values
        X = transform(X, args.algorithm)

        df = pd.DataFrame(X)
        df['y'] = y

        df.to_csv(filepath, index=False)


if __name__ == '__main__':
    main()
