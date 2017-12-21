import gzip
from os import listdir
from os.path import dirname, join, isfile, splitext
import multiprocessing as mp
import re
import logging

from Bio import Entrez
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec, LineSentence
from scipy.sparse import csr_matrix

DATA_DIR = join(dirname(dirname(__file__)), 'data')

TEXTS_FILE = join(DATA_DIR, 'X.txt')
LABELS_FILE = join(DATA_DIR, 'Y.txt')
LABELS_MAT_FILE = join(DATA_DIR, 'Y.npz')
CLASSES_FILE = join(DATA_DIR, 'classes.txt')
TRAIN_TEXTS_FILE = join(DATA_DIR, 'X_train.txt')
TEST_TEXTS_FILE = join(DATA_DIR, 'X_test.txt')
VAL_TEXTS_FILE = join(DATA_DIR, 'X_val.txt')
TRAIN_LABELS_MAT_FILE = join(DATA_DIR, 'Y_train.npz')
TEST_LABELS_MAT_FILE = join(DATA_DIR, 'Y_test.npz')
VAL_LABELS_MAT_FILE = join(DATA_DIR, 'Y_val.npz')
WORDS_FILE = join(DATA_DIR, 'words.txt')


def parse_medline_xmls(xmls_dir):
    xmls = [join(xmls_dir, f) for f in listdir(xmls_dir)
            if f.startswith('medline') and f.endswith('.xml.gz')]

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(mp.cpu_count())

    watcher = pool.apply_async(parse_medline_citation, (queue,))

    jobs = []
    for xml in xmls:
        job = pool.apply_async(parse_medline_xml, (xml, queue,))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put(None)
    watcher.get()
    pool.close()


def parse_medline_xml(file_path, queue):
    file_handle = gzip.open(file_path)
    results = parse_medline_handle(file_handle)

    for result in results:
        queue.put(result)

    print(file_path)


def parse_medline_handle(handle):
    content = Entrez.read(handle)
    pubmed_articles = content['PubmedArticle']

    for pubmed_article in pubmed_articles:
        try:
            citation = pubmed_article['MedlineCitation']

            headings = citation['MeshHeadingList']
            article = citation['Article']

            title = article['ArticleTitle']
            abstract_parts = article['Abstract']['AbstractText']

            abstract = ''
            for abstract_part in abstract_parts:
                abstract += abstract_part.title()

            text = '%s. %s' % (title.rstrip('.'), abstract)
            text = text.replace('\n', ' ').replace('\r', '')

            terms = []
            for heading in headings:
                term = heading['DescriptorName'].title()
                terms.append(term)

            yield (text, terms)

        except KeyError:
            continue


def parse_medline_citation(queue):
    with open(TEXTS_FILE, 'w') as texts_f, \
            open(LABELS_FILE, 'w') as terms_f:

        while True:
            result = queue.get()

            if result is None:
                break

            (text, terms) = result
            text = clean_text(text)
            terms = ';'.join(terms)

            texts_f.write(text + '\n')
            terms_f.write(terms + '\n')


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text


def construct_labels_mat():
    Y = [l.rstrip('\n').split(';') for l in open(LABELS_FILE, 'r')]

    mlb = MultiLabelBinarizer(sparse_output=True)
    Y = mlb.fit_transform(Y)
    classes = mlb.classes_

    Y, classes = filter_labels(Y, classes, min_support=10000)

    save_sparse_csr(LABELS_MAT_FILE, Y)

    with open(CLASSES_FILE, 'w') as f:
        for c in classes:
            f.write(c + '\n')


def filter_labels(Y, classes, min_support=10000):
    supports = Y.sum(axis=0)
    supports = np.asarray(supports)[0]

    indices = np.where(supports >= min_support)[0]
    classes = np.array(classes)
    classes = classes[indices]
    Y = Y[:, indices]

    return Y, classes


def save_sparse_csr(file_path, array):
    # .npz extension is added automatically
    file_path = splitext(file_path)[0]

    np.savez(file_path, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def read_sparse_csr(file_path):
    loader = np.load(file_path)

    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'], dtype=np.int8)


def split_dataset():
    X = open(TEXTS_FILE, 'rb').readlines()
    Y = read_sparse_csr(LABELS_MAT_FILE)

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=0.2, random_state=42)

    save_sparse_csr(TRAIN_LABELS_MAT_FILE, Y_train)
    save_sparse_csr(TEST_LABELS_MAT_FILE, Y_test)
    save_sparse_csr(VAL_LABELS_MAT_FILE, Y_val)

    with open(TRAIN_TEXTS_FILE, 'wb') as f:
        f.writelines(X_train)

    with open(TEST_TEXTS_FILE, 'wb') as f:
        f.writelines(X_test)

    with open(VAL_TEXTS_FILE, 'wb') as f:
        f.writelines(X_val)


def create_words_whitelist():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    model = Word2Vec(min_count=14)
    docs = LineSentence(TRAIN_TEXTS_FILE)
    model.build_vocab(docs)

    with open(WORDS_FILE, 'w') as f:
        for w in model.wv.vocab.keys():
            f.write(w + '\n')


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('xmls_dir')
    args = parser.parse_args()

    if not isfile(TEXTS_FILE) or not isfile(LABELS_FILE):
        print('Parsing MEDLINE xmls')
        parse_medline_xmls(args.xmls_dir)

    print('present', TEXTS_FILE)
    print('present', LABELS_FILE)

    if not isfile(LABELS_MAT_FILE):
        print('Constructing labels matrix')
        construct_labels_mat()

    print('present', LABELS_MAT_FILE)

    data_files = [TRAIN_TEXTS_FILE,
                  TEST_TEXTS_FILE,
                  VAL_TEXTS_FILE,
                  TRAIN_LABELS_MAT_FILE,
                  TEST_LABELS_MAT_FILE,
                  VAL_LABELS_MAT_FILE]

    if any([not isfile(f) for f in data_files]):
        print('Splitting dataset')
        split_dataset()

    for f in data_files:
        print('present', f)

    if not isfile(WORDS_FILE):
        print('Creating words whitelist')
        create_words_whitelist()

    print('present', WORDS_FILE)


if __name__ == '__main__':
    main()
