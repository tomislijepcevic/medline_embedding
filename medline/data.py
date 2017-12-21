from .dataprep import (
    TEXTS_FILE,
    LABELS_MAT_FILE,
    CLASSES_FILE,
    TRAIN_TEXTS_FILE,
    TEST_TEXTS_FILE,
    VAL_TEXTS_FILE,
    TRAIN_LABELS_MAT_FILE,
    TEST_LABELS_MAT_FILE,
    VAL_LABELS_MAT_FILE,
    WORDS_FILE,
    read_sparse_csr,
)


def read_lines(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line.rstrip('\n')


def read_classes():
    return list(read_lines(CLASSES_FILE))


def read_words():
    return list(read_lines(WORDS_FILE))


def read_texts():
    return read_lines(TEXTS_FILE)


def read_train_texts():
    return read_lines(TRAIN_TEXTS_FILE)


def read_val_texts():
    return read_lines(VAL_TEXTS_FILE)


def read_test_texts():
    return read_lines(TEST_TEXTS_FILE)


def read_labels():
    return read_sparse_csr(LABELS_MAT_FILE)


def read_train_labels():
    return read_sparse_csr(TRAIN_LABELS_MAT_FILE)


def read_val_labels():
    return read_sparse_csr(VAL_LABELS_MAT_FILE)


def read_test_labels():
    return read_sparse_csr(TEST_LABELS_MAT_FILE)
