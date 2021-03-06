import os
import itertools
import numpy as np

import mnist
import kmnist
import cifar


class Dataset(object):
    '''initialize dataset class:
            datasoure: 'mnist', 'kmnist', 'cifar-10'
            path_data: the data path
            is_sample: Ture / False
            sample_class: the class sample belongs to'''
    def __init__(self, datasource, path_data, is_sample, sample_class, **kwargs):
        self.datasource = datasource
        self.path_data = path_data
        self.rand = np.random.RandomState(9)
        if self.datasource == 'mnist':
            self.num_classes = 10
            self.dataset = mnist.read_data(os.path.join(self.path_data, 'MNIST'), False, is_sample, sample_class)
        elif self.datasource == 'kmnist':
            self.num_classes = 10
            self.dataset = kmnist.read_data(os.path.join(self.path_data, 'KMNIST'), False, is_sample, sample_class)
        elif self.datasource == 'cifar-10':
            self.num_classes = 10
            self.dataset = cifar.read_data(os.path.join(self.path_data, 'cifar-10-batches-py'), is_sample, sample_class)
        else:
            raise NotImplementedError
        self.split_dataset('train', 'val', int(self.dataset['train']['input'].shape[0] * 0.1),
            self.rand)
        self.num_example = {k: self.dataset[k]['input'].shape[0] for k in self.dataset.keys()}
        self.example_generator = {
            'train': self.iterate_example('train'),
            'val': self.iterate_example('val'),
            'test': self.iterate_example('test', shuffle=False),
        }

    def iterate_example(self, mode, shuffle=True):
        '''iterate example:
            mode: 'train', 'val', 'test'
            shuffle: True / False, defualt as True '''
        epochs = itertools.count()
        for i in epochs:
            example_ids = list(range(self.num_example[mode]))
            if shuffle:
                self.rand.shuffle(example_ids)
            for example_id in example_ids:
                yield {
                    'input': self.dataset[mode]['input'][example_id],
                    'label': self.dataset[mode]['label'][example_id],
                    'id': example_id,
                }

    def get_next_batch(self, mode, batch_size):
        '''get next batch:
            mode: 'train', 'test', 'val'
            batch_size: the size of batch'''
        inputs, labels, ids = [], [], []
        for i in range(batch_size):
            example = next(self.example_generator[mode])
            inputs.append(example['input'])
            labels.append(example['label'])
            ids.append(example['id'])
        return {
            'input': np.asarray(inputs),
            'label': np.asarray(labels),
            'id': np.asarray(ids),
        }

    def generate_example_epoch(self, mode):
        '''generate example epoch:
            mode: 'train', 'val', 'test' '''
        example_ids = range(self.num_example[mode])
        for example_id in example_ids:
            yield {
                'input': self.dataset[mode]['input'][example_id],
                'label': self.dataset[mode]['label'][example_id],
                'id': example_id,
            }

    def split_dataset(self, source, target, number, rand):
        '''split dataset into target and source
            number: the number of data in target dataset
            rand: to shuffle'''
        keys = ['input', 'label']
        indices = list(range(self.dataset[source]['input'].shape[0]))
        rand.shuffle(indices)
        ind_target = indices[:number]
        ind_remain = indices[number:]
        self.dataset[target] = {k: self.dataset[source][k][ind_target] for k in keys}
        self.dataset[source] = {k: self.dataset[source][k][ind_remain] for k in keys}
