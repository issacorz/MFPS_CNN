"""
" Dataset module
" Handling protein sequence data.
"   Batch control, encoding, etc.
"""
import pandas as pd
import sys
import os

import tensorflow as tf
import numpy as np


CHARLEN = 20


def encoding_label_np(l, arr):
    arr[int(l)] = 1


#  DATASET Class


# works for large dataset
class DataSetForPSSM(object):
    def __init__(self, fpath, seqlen, n_classes, num_feature, is_raw, need_shuffle=True, set_scaling=True):
        self.SEQLEN = seqlen
        self.NCLASSES = n_classes
        # self.charset = CHARSET
        self.charset_size = CHARLEN

        # read raw file
        self.is_raw = is_raw
        if self.is_raw:
            print("Train raw sequence")
        else:
            self.charset_size = num_feature
            print("Train other features")
        self._data, self._label = self.read_raw(fpath)
        self._num_data = len(self._label)
        self._epochs_completed = 0
        self._index_in_epoch = 0

        self._perm = np.arange(self._num_data)
        if need_shuffle:
            # shuffle data
            print("Needs shuffle")
            np.random.shuffle(self._perm)
        print("Reading data done")

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_data:
            # print("%d epoch finish!" % self._epochs_completed)
            # finished epoch
            self._epochs_completed += 1
            # shuffle the data
            np.random.shuffle(self._perm)

            # start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_data

        end = self._index_in_epoch
        idxs = self._perm[start:end]
        return self.parse_data(idxs)

    def iter_batch(self, batch_size, max_iter):
        while True:
            batch = self.next_batch(batch_size)

            if self._epochs_completed >= max_iter:
                break
            elif len(batch) == 0:
                continue
            else:
                yield batch

    def iter_once(self, batch_size, with_raw=False):
        while True:
            start = self._index_in_epoch
            self._index_in_epoch += batch_size

            if self._index_in_epoch > self._num_data:
                end = self._num_data
                idxs = self._perm[start:end]
                if len(idxs) > 0:
                    yield self.parse_data(idxs, with_raw)
                break

            end = self._index_in_epoch
            idxs = self._perm[start:end]
            yield self.parse_data(idxs, with_raw)

    def full_batch(self):
        return self.parse_data(self._perm)



    def read_raw(self, fpath):
        print("Read %s start" % fpath)
        sequence = np.load(fpath)
        data = sequence['Data']
        label = sequence['label']

        return data, label

    def parse_data(self, idxs, with_raw=False):
        isize = len(idxs)

        labels_encode = np.zeros((isize, self.NCLASSES), dtype=np.uint8)
        raw = []

        label = self._label[idxs]
        data = self._data[idxs]
        x_train = np.reshape(data, (idxs.shape[0], self.charset_size * self.SEQLEN))

        for i, idx in enumerate(idxs):
            seq = label[i]
            encoding_label_np(seq, labels_encode[i])

        if with_raw:
            return x_train, labels_encode, raw
        else:
            return x_train, labels_encode
