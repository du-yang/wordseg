import json
import collections
import numpy as np


class DataSet(object):
    def __init__(self,x_data,y_data,):

        self._x_data = np.array(x_data)
        self._y_data = np.array(y_data)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(x_data)

    @property
    def x_data(self):
        return self._x_data

    @property
    def y_data(self):
        return self._y_data

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._x_data = self.x_data[perm0]
            self._y_data = self.y_data[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start

            x_rest_part = self._x_data[start:self._num_examples]
            y_rest_part = self._y_data[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._x_data = self._x_data[perm]
                self._y_data = self._y_data[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            x_new_part = self._x_data[start:end]
            y_new_part = self._y_data[start:end]
            return np.concatenate((x_rest_part, x_new_part), axis=0), np.concatenate(
                (y_rest_part, y_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._x_data[start:end], self._y_data[start:end]

def word_to_id(dict_data):
    counter = collections.Counter(''.join(dict_data.keys()))
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_id = dict(zip(words, range(3, len(words) + 3)))
    word_id['B'] = 1
    word_id['I'] = 2
    return word_id

def datas(dict_data,word_id,num_step):
    x_data = []
    y_data = []
    # word_id = word_to_id(dict_data)
    for line in dict_data:
        x_list = [word_id[word] for word in list(line)][:num_step]
        y_list = [word_id[word] for word in dict_data[line]][:num_step]
        x_len = len(x_list)
        y_len = len(y_list)
        assert x_len == y_len
        if x_len<num_step:
            x_list.extend([0]*(num_step-x_len))
            y_list.extend([0]*(num_step-y_len))
        x_data.append(x_list)
        y_data.append(y_list)
    return x_data,y_data

def read_data_sets(dataName,dictName,num_step):

    with open(dataName) as f:
        train_data = json.load(f)

    with open(dictName) as f:
        w2id_data = json.load(f)

    x_data, y_data = datas(train_data,w2id_data, num_step)

    return DataSet(x_data, y_data)

if __name__ == '__main__':
    # with open('data/jdData.json') as f:
    #     dict_data = json.load(f)
    # w2id = word_to_id(dict_data)

    # with open('data/w2id.json') as f:
    #     w2id = json.load(f)
    #
    # print(w2id['两'])
    dataset = read_data_sets('data/jdData.json','data/w2id.json',2)
    print(dataset.next_batch(2))

