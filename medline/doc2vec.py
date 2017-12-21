import multiprocessing as mp
import logging

from gensim.models.doc2vec import Doc2Vec, TaggedLineDocument
from medline.data import TRAIN_TEXTS_FILE

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    models = ['dbow', 'dm_mean', 'dm_sum', 'dm_concat']
    parser.add_argument('model', choices=models)
    parser.add_argument('dest')
    parser.add_argument('--size', default=300, type=int)
    args = parser.parse_args()

    common_params = {
        'min_count': 14,
        'iter': 10,
        'workers': mp.cpu_count(),
        'size': args.size
    }

    if args.model == 'dbow':
        model = Doc2Vec(dm=0, **common_params)
    elif args.model == 'dm_concat':
        model = Doc2Vec(dm=1, dm_concat=1, **common_params)
    elif args.model == 'dm_mean':
        model = Doc2Vec(dm=1, dm_mean=1, **common_params)
    elif args.model == 'dm_sum':
        model = Doc2Vec(dm=1, dm_mean=0, **common_params)

    texts = TaggedLineDocument(TRAIN_TEXTS_FILE)

    model.build_vocab(texts)
    model.train(texts, epochs=model.iter,
                total_examples=model.corpus_count)

    model.save(args.dest)


if __name__ == '__main__':
    main()
