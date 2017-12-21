from os import path
from os import makedirs
from itertools import islice, count
from contextlib import contextmanager
import numpy as np

from keras.callbacks import (
    TensorBoard,
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau
)
from sklearn.metrics import average_precision_score

NB_TRAIN_SAMPLES = 9250000
NB_VAL_SAMPLES = 2300000
NB_TEST_SAMPLES = 1000000
BATCH_SIZE = 50
EPOCHS = 20


def train_model(model_dir, model, train_data, val_data,
                batch_size=BATCH_SIZE, epochs=EPOCHS,
                nb_train_samples=NB_TRAIN_SAMPLES,
                nb_val_samples=NB_VAL_SAMPLES):

    makedirs(model_dir, exist_ok=True)
    save_summary(model, model_dir)

    train_gen = create_data_generator(*train_data, batch_size)
    val_gen = create_data_generator(*val_data, batch_size)
    steps_per_epoch = nb_train_samples // batch_size // epochs
    validation_steps = nb_val_samples // batch_size // epochs
    callbacks = get_callbacks(model_dir)

    learning_path = path.join(model_dir, 'learning.txt')

    with open(learning_path, 'w') as f, redirect_stdout(f):
        history = model.fit_generator(train_gen, epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      validation_data=val_gen,
                                      validation_steps=validation_steps,
                                      callbacks=callbacks, verbose=2,
                                      workers=10)

        return history


def create_data_generator(X, Y, batch_size):
    X = (np.array(X_batch) for X_batch in to_batches(X, batch_size))
    Y = sparse_matrix_to_batches(Y, batch_size)

    data_gen = zip(X, Y)
    data_gen = threadsafe_iter(data_gen)

    return data_gen


def to_batches(iterable, n):
    it = iter(iterable)

    while True:
        batch = list(islice(it, n))

        if not batch:
            return

        yield batch


def sparse_matrix_to_batches(mat, batch_size):
    c1 = count(start=0, step=batch_size)
    c2 = count(start=batch_size, step=batch_size)
    mat_batches = (mat[i:j].toarray() for i, j in zip(c1, c2))

    return mat_batches


class threadsafe_iter:
    def __init__(self, it):
        import threading

        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def get_callbacks(model_dir):
    logger_path = path.join(model_dir, 'training.csv')
    model_path = path.join(model_dir, 'model.hdf5')

    csv_logger = CSVLogger(logger_path)
    checkpointer = ModelCheckpoint(
        model_path, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0,
                               patience=3, mode='auto', verbose=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.001, verbose=1)
    tensor_board = TensorBoard(log_dir=model_dir, histogram_freq=1,
                               write_graph=True, write_images=False)

    callbacks = [csv_logger, checkpointer, early_stop, reduce_lr, tensor_board]

    return callbacks


@contextmanager
def redirect_stdout(new_target):
    import sys

    old_target, sys.stdout = sys.stdout, new_target  # replace sys.stdout
    try:
        yield new_target  # run some code with the replaced stdout
    finally:
        sys.stdout = old_target  # restore to the previous value


def save_summary(model, model_dir):
    summary_path = path.join(model_dir, 'summary.txt')

    with open(summary_path, 'w') as f, redirect_stdout(f):
        print(model.summary())


def test_model(model, X_test, Y_test, scores_filepath,
               nb_test_samples=NB_TEST_SAMPLES):
    Y_prob = (y_prob for X_batch in to_batches(X_test, BATCH_SIZE)
              for y_prob in model.predict(np.array(X_batch)))

    Y_test = Y_test[:nb_test_samples].toarray()
    Y_prob = np.array(list(islice(Y_prob, nb_test_samples)))

    ap_scores = average_precision_score(Y_test, Y_prob, average=None)

    with open(scores_filepath, 'w') as f:
        for ap_score in ap_scores:
            f.write(str(ap_score) + '\n')
