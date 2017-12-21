import multiprocessing as mp
import logging

from gensim.models.word2vec import Word2Vec, LineSentence
from medline.data import TRAIN_TEXTS_FILE

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['skipgram', 'cbow'])
    parser.add_argument('dest')
    parser.add_argument('--size', default=300, type=int)
    args = parser.parse_args()

    common_params = {
        'size': 300,
        'min_count': 14,
        'workers': mp.cpu_count(),
        'size': args.size
    }

    if args.model == 'skipgram':
        model = Word2Vec(sg=1, **common_params)
    elif args.model == 'cbow':
        model = Word2Vec(sg=0, **common_params)

    texts = LineSentence(TRAIN_TEXTS_FILE)

    model.build_vocab(texts)
    model.train(texts, epochs=model.iter,
                total_examples=model.corpus_count)

    model.wv.save_word2vec_format(args.dest)


if __name__ == '__main__':
    main()
