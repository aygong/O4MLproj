import os
import pickle
import numpy as np

def read_data(path_dataset, is_sample=False, sample_class=0):
    '''read data and store into train and test datasets'''
    # train batch
    train_batch = {}
    for i in range(5):
        filename = os.path.join(path_dataset, 'data_batch_{}'.format(i+1))
        with open(filename, 'rb') as f:
            try:
                batch = pickle.load(f, encoding='bytes')
            except TypeError:
                batch = pickle.load(f) # for python 2
            for key in batch.keys():
                train_batch.setdefault(key, []).extend(batch[key])
    train_batch = {k: np.stack(v, 0) for k, v in train_batch.items()} # stack into one batch

    # test batch
    filename = os.path.join(path_dataset, 'test_batch')
    with open(filename, 'rb') as f:
        try:
            test_batch = pickle.load(f, encoding='bytes')
        except TypeError:
            test_batch = pickle.load(f)

    # Reshape images: (n, 3072) -> (n, 32, 32, 3)
    label_key = 'labels'.encode('utf-8')
    train_images = np.transpose(
        np.reshape(train_batch['data'.encode('utf-8')], [-1, 3, 32, 32]), [0,2,3,1])
    train_labels = np.asarray(train_batch[label_key])
    test_images = np.transpose(
        np.reshape(test_batch['data'.encode('utf-8')], [-1, 3, 32, 32]), [0,2,3,1])
    test_labels = np.asarray(test_batch[label_key])

    # Pre-processing (normalize)
    train_images = np.divide(train_images, 255, dtype=np.float32)
    test_images = np.divide(test_images, 255, dtype=np.float32)
    channel_mean = np.mean(train_images, axis=(0,1,2), dtype=np.float32, keepdims=True)
    channel_std = np.std(train_images, axis=(0,1,2), dtype=np.float32, keepdims=True)
    train_images = (train_images - channel_mean) / channel_std
    test_images = (test_images - channel_mean) / channel_std

    if is_sample:
        sample_index = np.where(train_labels == sample_class)
        sample_train_images = train_images[sample_index]
        sample_train_labels = train_labels[sample_index]
        sample_index = np.where(test_labels == sample_class)
        sample_test_images = test_images[sample_index]
        sample_test_labels = test_labels[sample_index]
        dataset = {
            'train': {'input': sample_train_images, 'label': sample_train_labels},
            'test': {'input': sample_test_images, 'label': sample_test_labels},
        }
    else:
        dataset = {
            'train': {'input': train_images, 'label': train_labels},
            'test': {'input': test_images, 'label': test_labels},
        }

    return dataset
