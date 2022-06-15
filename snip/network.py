import tensorflow as tf

from functools import reduce
from helpers import static_size


def load_network(
        datasource, arch, num_classes,
        initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
        ):
    networks = {
        'lenet300': lambda: LeNet300(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            ),
        'lenet5': lambda: LeNet5(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap),
        'alexnet-v1': lambda: AlexNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, k=1),
        'alexnet-v2': lambda: AlexNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, k=2),
        'vgg-c': lambda: VGG(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version='C'),
        'vgg-d': lambda: VGG(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version='D'),
        'vgg-like': lambda: VGG(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version='like'),
        'resnet-18': lambda: ResNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version=18),
        'resnet-34': lambda: ResNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version=34),
        'squeezenet-vanilla': lambda: SqueezeNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version='vanilla'),
        'squeezenet-bypass': lambda: SqueezeNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version='bypass'),
        'googlenet': lambda: GoogLeNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes),
        'densenet-121': lambda: DenseNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version=121),
        'densenet-169': lambda: DenseNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version=169),
        'densenet-201': lambda: DenseNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version=201),
        'densenet-264': lambda: DenseNet(
            initializer_w_bp, initializer_b_bp, initializer_w_ap, initializer_b_ap,
            datasource, num_classes, version=264),
    }
    return networks[arch]()


def get_initializer(initializer, dtype):
    if initializer == 'zeros':
        return tf.zeros_initializer(dtype=dtype)
    elif initializer == 'vs':
        return tf.variance_scaling_initializer(dtype=dtype)
    else:
        raise NotImplementedError


class LeNet300(object):
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 ):
        self.name = 'lenet300'
        self.input_dims = [28, 28, 1] # height, width, channel
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            weights['w1'] = tf.get_variable('w1', [784, 300], **w_params)
            weights['w2'] = tf.get_variable('w2', [300, 100], **w_params)
            weights['w3'] = tf.get_variable('w3', [100, 10], **w_params)
            weights['b1'] = tf.get_variable('b1', [300], **b_params)
            weights['b2'] = tf.get_variable('b2', [100], **b_params)
            weights['b3'] = tf.get_variable('b3', [10], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        inputs_flat = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        fc1 = tf.matmul(inputs_flat, weights['w1']) + weights['b1']
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.matmul(fc1, weights['w2']) + weights['b2']
        fc2 = tf.nn.relu(fc2)
        fc3 = tf.matmul(fc2, weights['w3']) + weights['b3']
        return fc3


class LeNet5(object):
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 ):
        self.name = 'lenet5'
        self.input_dims = [28, 28, 1] # height, width, channel
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            weights['w1'] = tf.get_variable('w1', [5, 5, 1, 20], **w_params)
            weights['w2'] = tf.get_variable('w2', [5, 5, 20, 50], **w_params)
            weights['w3'] = tf.get_variable('w3', [800, 500], **w_params)
            weights['w4'] = tf.get_variable('w4', [500, 10], **w_params)
            weights['b1'] = tf.get_variable('b1', [20], **b_params)
            weights['b2'] = tf.get_variable('b2', [50], **b_params)
            weights['b3'] = tf.get_variable('b3', [500], **b_params)
            weights['b4'] = tf.get_variable('b4', [10], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        conv1 = tf.nn.conv2d(inputs, weights['w1'], [1, 1, 1, 1], 'VALID') + weights['b1']
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        conv2 = tf.nn.conv2d(pool1, weights['w2'], [1, 1, 1, 1], 'VALID') + weights['b2']
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
        flatten = tf.reshape(pool2, [-1, reduce(lambda x, y: x*y, pool2.shape.as_list()[1:])])
        fc1 = tf.matmul(flatten, weights['w3']) + weights['b3']
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.matmul(fc1, weights['w4']) + weights['b4'] # logits
        return fc2


class AlexNet(object):
    ''' Similar to Alexnet in terms of the total number of conv and fc layers.

    Conv layers:
        The size of kernels and the number of conv filters are the same as the original.
        Due to the smaller input size (CIFAR rather than IMAGENET) we use different strides.
    FC layers:
        The size of fc layers are controlled by k (multiplied by 1024).
        In the original Alexnet, k=4 making the size of largest fc layers to be 4096.
    '''
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 datasource,
                 num_classes,
                 k,
                 ):
        self.datasource = datasource
        self.num_classes = num_classes
        self.k = k
        self.name = 'alexnet'
        self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3] # h,w,c
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        k = self.k
        weights = {}
        with tf.variable_scope(scope):
            weights['w1'] = tf.get_variable('w1', [11, 11, 3, 96], **w_params)
            weights['w2'] = tf.get_variable('w2', [5, 5, 96, 256], **w_params)
            weights['w3'] = tf.get_variable('w3', [3, 3, 256, 384], **w_params)
            weights['w4'] = tf.get_variable('w4', [3, 3, 384, 384], **w_params)
            weights['w5'] = tf.get_variable('w5', [3, 3, 384, 256], **w_params)
            weights['w6'] = tf.get_variable('w6', [256, 1024*k], **w_params)
            weights['w7'] = tf.get_variable('w7', [1024*k, 1024*k], **w_params)
            weights['w8'] = tf.get_variable('w8', [1024*k, self.num_classes], **w_params)
            weights['b1'] = tf.get_variable('b1', [96], **b_params)
            weights['b2'] = tf.get_variable('b2', [256], **b_params)
            weights['b3'] = tf.get_variable('b3', [384], **b_params)
            weights['b4'] = tf.get_variable('b4', [384], **b_params)
            weights['b5'] = tf.get_variable('b5', [256], **b_params)
            weights['b6'] = tf.get_variable('b6', [1024*k], **b_params)
            weights['b7'] = tf.get_variable('b7', [1024*k], **b_params)
            weights['b8'] = tf.get_variable('b8', [self.num_classes], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        bn_params = {
            'training': is_train,
            'trainable': trainable,
        }
        init_st = 4 if self.datasource == 'tiny-imagenet' else 2
        inputs = tf.nn.conv2d(inputs, weights['w1'], [1,init_st,init_st,1], 'SAME') + weights['b1']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w2'], [1, 2, 2, 1], 'SAME') + weights['b2']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w3'], [1, 2, 2, 1], 'SAME') + weights['b3']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w4'], [1, 2, 2, 1], 'SAME') + weights['b4']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w5'], [1, 2, 2, 1], 'SAME') + weights['b5']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        inputs = tf.matmul(inputs, weights['w6']) + weights['b6']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.matmul(inputs, weights['w7']) + weights['b7']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.matmul(inputs, weights['w8']) + weights['b8'] # logits
        return inputs


class VGG(object):
    '''
    Similar to the original VGG.
    Available models:
        - VGG-C
        - VGG-D
        - VGG-like

    Differences:
        The number of parameters in conv layers are the same as the original.
        The number of parameters in fc layers are reduced to 512 (4096 -> 512).
        The number of total parameters are different, not just because of the size of fc layers,
        but also due to the fact that the first fc layer receives 1x1 image rather than 7x7 image
        because the input is CIFAR not IMAGENET.
        No dropout is used. Instead, batch norm is used.

    Other refereneces.
        (1) The original paper:
        - paper: https://arxiv.org/pdf/1409.1556.pdf
        - code: http://www.robots.ox.ac.uk/~vgg/research/very_deep/
        * Dropout between fc layers.
        * There is no BatchNorm.
        (2) VGG-like by Zagoruyko, adapted for CIFAR-10.
        - project and code: http://torch.ch/blog/2015/07/30/cifar.html
        * Differences to the original VGG-16 (1):
            - # of fc layers 3 -> 2, so there are 15 (learnable) layers in total.
            - size of fc layers 4096 -> 512.
            - use BatchNorm and add more Dropout.
    '''
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 datasource,
                 num_classes,
                 version,
                 ):
        self.datasource = datasource
        self.num_classes = num_classes
        self.version = version
        self.name = 'VGG-{}'.format(version)
        self.input_dims = [64, 64, 3] if self.datasource == 'tiny-imagenet' else [32, 32, 3] # h,w,c
        self.inputs = self.construct_inputs()
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        dtype = tf.float32
        w_params = {
            'initializer': get_initializer(initializer_w, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, dtype),
            'dtype': dtype,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            weights['w1'] = tf.get_variable('w1', [3, 3, 3, 64], **w_params)
            weights['w2'] = tf.get_variable('w2', [3, 3, 64, 64], **w_params)
            weights['w3'] = tf.get_variable('w3', [3, 3, 64, 128], **w_params)
            weights['w4'] = tf.get_variable('w4', [3, 3, 128, 128], **w_params)
            weights['b1'] = tf.get_variable('b1', [64], **b_params)
            weights['b2'] = tf.get_variable('b2', [64], **b_params)
            weights['b3'] = tf.get_variable('b3', [128], **b_params)
            weights['b4'] = tf.get_variable('b4', [128], **b_params)
            if self.version == 'C':
                weights['w5'] = tf.get_variable('w5', [3, 3, 128, 256], **w_params)
                weights['w6'] = tf.get_variable('w6', [3, 3, 256, 256], **w_params)
                weights['w7'] = tf.get_variable('w7', [1, 1, 256, 256], **w_params)
                weights['w8'] = tf.get_variable('w8', [3, 3, 256, 512], **w_params)
                weights['w9'] = tf.get_variable('w9', [3, 3, 512, 512], **w_params)
                weights['w10'] = tf.get_variable('w10', [1, 1, 512, 512], **w_params)
                weights['w11'] = tf.get_variable('w11', [3, 3, 512, 512], **w_params)
                weights['w12'] = tf.get_variable('w12', [3, 3, 512, 512], **w_params)
                weights['w13'] = tf.get_variable('w13', [1, 1, 512, 512], **w_params)
                weights['b5'] = tf.get_variable('b5', [256], **b_params)
                weights['b6'] = tf.get_variable('b6', [256], **b_params)
                weights['b7'] = tf.get_variable('b7', [256], **b_params)
                weights['b8'] = tf.get_variable('b8', [512], **b_params)
                weights['b9'] = tf.get_variable('b9', [512], **b_params)
                weights['b10'] = tf.get_variable('b10', [512], **b_params)
                weights['b11'] = tf.get_variable('b11', [512], **b_params)
                weights['b12'] = tf.get_variable('b12', [512], **b_params)
                weights['b13'] = tf.get_variable('b13', [512], **b_params)
            elif self.version == 'D' or self.version == 'like':
                weights['w5'] = tf.get_variable('w5', [3, 3, 128, 256], **w_params)
                weights['w6'] = tf.get_variable('w6', [3, 3, 256, 256], **w_params)
                weights['w7'] = tf.get_variable('w7', [3, 3, 256, 256], **w_params)
                weights['w8'] = tf.get_variable('w8', [3, 3, 256, 512], **w_params)
                weights['w9'] = tf.get_variable('w9', [3, 3, 512, 512], **w_params)
                weights['w10'] = tf.get_variable('w10', [3, 3, 512, 512], **w_params)
                weights['w11'] = tf.get_variable('w11', [3, 3, 512, 512], **w_params)
                weights['w12'] = tf.get_variable('w12', [3, 3, 512, 512], **w_params)
                weights['w13'] = tf.get_variable('w13', [3, 3, 512, 512], **w_params)
                weights['b5'] = tf.get_variable('b5', [256], **b_params)
                weights['b6'] = tf.get_variable('b6', [256], **b_params)
                weights['b7'] = tf.get_variable('b7', [256], **b_params)
                weights['b8'] = tf.get_variable('b8', [512], **b_params)
                weights['b9'] = tf.get_variable('b9', [512], **b_params)
                weights['b10'] = tf.get_variable('b10', [512], **b_params)
                weights['b11'] = tf.get_variable('b11', [512], **b_params)
                weights['b12'] = tf.get_variable('b12', [512], **b_params)
                weights['b13'] = tf.get_variable('b13', [512], **b_params)
            weights['w14'] = tf.get_variable('w14', [512, 512], **w_params)
            weights['b14'] = tf.get_variable('b14', [512], **b_params)
            if not self.version == 'like':
                weights['w15'] = tf.get_variable('w15', [512, 512], **w_params)
                weights['w16'] = tf.get_variable('w16', [512, self.num_classes], **w_params)
                weights['b15'] = tf.get_variable('b15', [512], **b_params)
                weights['b16'] = tf.get_variable('b16', [self.num_classes], **b_params)
            else:
                weights['w15'] = tf.get_variable('w15', [512, self.num_classes], **w_params)
                weights['b15'] = tf.get_variable('b15', [self.num_classes], **b_params)
        return weights

    def forward_pass(self, weights, inputs, is_train, trainable=True):
        def _conv_block(inputs, bn_params, filt, st=1):
            inputs = tf.nn.conv2d(inputs, filt['w'], [1, st, st, 1], 'SAME') + filt['b']
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            return inputs

        bn_params = {
            'training': is_train,
            'trainable': trainable,
        }
        init_st = 2 if self.datasource == 'tiny-imagenet' else 1

        inputs = _conv_block(inputs, bn_params, {'w': weights['w1'], 'b': weights['b1']}, init_st)
        inputs = _conv_block(inputs, bn_params, {'w': weights['w2'], 'b': weights['b2']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w3'], 'b': weights['b3']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w4'], 'b': weights['b4']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w5'], 'b': weights['b5']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w6'], 'b': weights['b6']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w7'], 'b': weights['b7']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w8'], 'b': weights['b8']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w9'], 'b': weights['b9']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w10'], 'b': weights['b10']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        inputs = _conv_block(inputs, bn_params, {'w': weights['w11'], 'b': weights['b11']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w12'], 'b': weights['b12']})
        inputs = _conv_block(inputs, bn_params, {'w': weights['w13'], 'b': weights['b13']})
        inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        assert reduce(lambda x, y: x*y, inputs.shape.as_list()[1:3]) == 1

        inputs = tf.reshape(inputs, [-1, reduce(lambda x, y: x*y, inputs.shape.as_list()[1:])])
        inputs = tf.matmul(inputs, weights['w14']) + weights['b14']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        if not self.version == 'like':
            inputs = tf.matmul(inputs, weights['w15']) + weights['b15']
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            inputs = tf.matmul(inputs, weights['w16']) + weights['b16']
        else:
            inputs = tf.matmul(inputs, weights['w15']) + weights['b15']

        return inputs


class ResNet(object):
    """
    Define the neural network for ResNet.
    Reference:
        - K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition,"
          in Proc. IEEE CVPR, 2016, pp. 770-778.
        - https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    """
    def __init__(self,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 datasource,
                 num_classes,
                 version,
                 ):
        self.datasource = datasource
        if version == 18:
            self.name = 'resnet-18'
            self.conv1_num_Cout = 64
            self.stacked_blocks = [2, 2, 2, 2]
        elif version == 34:
            self.name = 'resnet-34'
            self.conv1_num_Cout = 64
            self.stacked_blocks = [3, 4, 6, 3]
        # Height × width × channels
        if self.datasource == 'tiny-imagenet':
            self.input_dims = [64, 64, 3]
        elif self.datasource == 'cifar-10':
            self.input_dims = [32, 32, 3]
        elif 'mnist' in self.datasource:
            self.input_dims = [28, 28, 1]
        else:
            raise Exception(f"{self.datasource} is not supported.")
        # The number of classes
        self.output_dims = num_classes
        # Construct inputs
        self.inputs = self.construct_inputs()
        # Construct weights
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        # Compute the number of parameters
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])
    
    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        w_params = {
            'initializer': get_initializer(initializer_w, tf.float32),
            'dtype': tf.float32,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, tf.float32),
            'dtype': tf.float32,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            # Construct the weights of the input layer
            num_layers = 1
            weights['w1'] = tf.get_variable('w1', [3, 3, self.input_dims[2], self.conv1_num_Cout], **w_params)
            weights['b1'] = tf.get_variable('b1', [self.conv1_num_Cout], **b_params)

            # Construct the weights of the stacked blocks
            num_Cin = self.conv1_num_Cout
            for index_outer in range(len(self.stacked_blocks)):
                num_Cout = self.conv1_num_Cout * (2 ** index_outer)
                for index_inner in range(self.stacked_blocks[index_outer]):
                    weights[f'w{num_layers+1}'] = tf.get_variable(
                        f'w{num_layers+1}', [3, 3, num_Cin if index_inner == 0 else num_Cout, num_Cout], **w_params)
                    weights[f'b{num_layers+1}'] = tf.get_variable(f'b{num_layers+1}', [num_Cout], **b_params)
                    weights[f'w{num_layers+2}'] = tf.get_variable(f'w{num_layers+2}', [3, 3, num_Cout, num_Cout], **w_params)
                    weights[f'b{num_layers+2}'] = tf.get_variable(f'b{num_layers+2}', [num_Cout], **b_params)
                    num_layers += 2
                num_Cin = num_Cout
            
            # Construct the weights of the output layer
            num_layers += 1
            weights[f'w{num_layers}'] = tf.get_variable(f'w{num_layers}', [num_Cout, self.output_dims], **w_params)
            weights[f'b{num_layers}'] = tf.get_variable(f'b{num_layers}', [self.output_dims], **b_params)
        
        return weights
    
    def forward_pass(self, weights, inputs, is_train, trainable=True):
        def building_block(inputs, bn_params, filt1, filt2, zeropad=False, strides=1):
            # Shortcut connection
            if zeropad:
                shortcut = tf.nn.avg_pool(inputs, [1, strides, strides, 1], [1, strides, strides, 1], 'SAME')
                shortcut = tf.concat([shortcut] + [tf.zeros_like(shortcut)]*int(zeropad), -1)
            else:
                shortcut = inputs
            # The plain network
            inputs = tf.nn.conv2d(inputs, filt1['w'], [1, strides, strides, 1], 'SAME') + filt1['b']
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            inputs = tf.nn.conv2d(inputs, filt2['w'], [1, 1, 1, 1], 'SAME') + filt2['b']
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = inputs + shortcut
            inputs = tf.nn.relu(inputs)
            return inputs
        
        bn_params = {
                'training': is_train,
                'trainable': trainable,
            }

        # The input layer
        num_layers = 1
        inputs = tf.nn.conv2d(inputs, weights['w1'], [1, 1, 1, 1], 'SAME') + weights['b1']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)

        # The stacked blocks (conv2_x to conv5_x)
        for nl in range(len(self.stacked_blocks)):
            # Downsampling is performed by conv3_x to conv5_x
            # The first sub-block
            inputs = building_block(inputs,
                bn_params,
                {'w': weights[f'w{num_layers+1}'], 'b': weights[f'b{num_layers+1}']},
                {'w': weights[f'w{num_layers+2}'], 'b': weights[f'b{num_layers+2}']},
                zeropad=False if nl == 0 else True,
                strides=1 if nl == 0 else 2,
            )
            num_layers += 2
            # The following sub-blocks
            for _ in range(1, self.stacked_blocks[nl]):
                inputs = building_block(inputs,
                    bn_params,
                    {'w': weights[f'w{num_layers+1}'], 'b': weights[f'b{num_layers+1}']},
                    {'w': weights[f'w{num_layers+2}'], 'b': weights[f'b{num_layers+2}']},
                )
                num_layers += 2
        
        # The output layer
        num_layers += 1
        # The average pooling
        inputs = tf.nn.avg_pool(inputs, [1, inputs.shape[1], inputs.shape[2], 1], [1, 1, 1, 1], 'VALID')
        inputs = tf.squeeze(inputs, [1, 2])
        # The fully connected layer
        inputs = tf.matmul(inputs, weights[f'w{num_layers}']) + weights[f'b{num_layers}']

        return inputs


class SqueezeNet(object):
    """
    Define the neural network for SqueezeNet.
    Reference:
        - F. N. Iandola, S. Han, M. W. Moskewicz, K. Ashraf, W. J. Dally, and K. Keutzer,
          "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and < 0.5 MB model size,"
          arXiv preprint arXiv:1602.07360, 2016.
        - https://arxiv.org/pdf/1602.07360.pdf?ref=https://githubhelp.com
    """
    def __init__(self,
                initializer_w_bp,
                initializer_b_bp,
                initializer_w_ap,
                initializer_b_ap,
                datasource,
                num_classes,
                version,
                ):
        self.datasource = datasource
        if version == 'vanilla':
            self.name = 'squeezenet-vanilla'
            self.bypass = False
        elif version == 'bypass':
            self.name = 'squeezenet-bypass'
            self.bypass = True
        # Height × width × channels
        if self.datasource == 'tiny-imagenet':
            self.input_dims = [64, 64, 3]
        elif self.datasource == 'cifar-10':
            self.input_dims = [32, 32, 3]
        else:
            raise Exception(f"{self.datasource} is not supported.")
        # The number of filters in fire modules
        self.module_filters = [[16, 64], [16, 64], [32, 128], [32, 128], 
                            [48, 192], [48, 192], [64, 256], [64, 256]]
        self.num_modules = len(self.module_filters)
        # The number of classes
        self.output_dims = num_classes
        # Construct inputs
        self.inputs = self.construct_inputs()
        # Construct weights
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        # Compute the number of parameters
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        w_params = {
            'initializer': get_initializer(initializer_w, tf.float32),
            'dtype': tf.float32,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, tf.float32),
            'dtype': tf.float32,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            # Construct the weights of the input layer
            num_layers = 1
            weights['w1'] = tf.get_variable('w1', [7, 7, self.input_dims[2], 96], **w_params)
            weights['b1'] = tf.get_variable('b1', [96], **b_params)

            # Construct the weights of the fire modules
            num_Cin = 96
            for index in range(self.num_modules):
                num_Cout = self.module_filters[index]
                weights[f'w{num_layers+1}'] = tf.get_variable(f'w{num_layers+1}', [1, 1, num_Cin, num_Cout[0]], **w_params)
                weights[f'b{num_layers+1}'] = tf.get_variable(f'b{num_layers+1}', [num_Cout[0]], **b_params)
                weights[f'w{num_layers+2}'] = tf.get_variable(f'w{num_layers+2}', [1, 1, num_Cout[0], num_Cout[1]], **w_params)
                weights[f'b{num_layers+2}'] = tf.get_variable(f'b{num_layers+2}', [num_Cout[1]], **b_params)
                weights[f'w{num_layers+3}'] = tf.get_variable(f'w{num_layers+3}', [3, 3, num_Cout[0], num_Cout[1]], **w_params)
                weights[f'b{num_layers+3}'] = tf.get_variable(f'b{num_layers+3}', [num_Cout[1]], **b_params)
                num_layers += 3
                num_Cin = num_Cout[1] * 2
            
            # Construct the weights of the output layer
            num_layers += 1
            weights[f'w{num_layers}'] = tf.get_variable(f'w{num_layers}', [1, 1, num_Cin, self.output_dims], **w_params)
            weights[f'b{num_layers}'] = tf.get_variable(f'b{num_layers}', [self.output_dims], **b_params)

        return weights
    
    def forward_pass(self, weights, inputs, is_train, trainable=True):
        def fire_module(inputs, bn_params, filt1, filt2, filt3):
            s1x = tf.nn.conv2d(inputs, filt1['w'], [1, 1, 1, 1], 'SAME') + filt1['b']
            s1x = tf.layers.batch_normalization(s1x, **bn_params)
            s1x = tf.nn.relu(s1x)
            e1x = tf.nn.conv2d(s1x, filt2['w'], [1, 1, 1, 1], 'SAME') + filt2['b']
            e1x = tf.layers.batch_normalization(e1x, **bn_params)
            e3x = tf.nn.conv2d(s1x, filt3['w'], [1, 1, 1, 1], 'SAME') + filt3['b']
            e3x = tf.layers.batch_normalization(e3x, **bn_params)
            inputs = tf.concat([e1x] + [e3x], -1)
            inputs = tf.nn.relu(inputs)
            return inputs
        
        bn_params = {
                'training': is_train,
                'trainable': trainable,
            }
        
        # The input layer
        num_layers = 1
        inputs = tf.nn.conv2d(inputs, weights['w1'], [1, 2, 2, 1], 'SAME') + weights['b1']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')

        # The fire modules
        # The fire modules 2 to 4
        for index in range(2, 5):
            inputs = fire_module(inputs,
                        bn_params,
                        {'w': weights[f'w{num_layers+1}'], 'b': weights[f'b{num_layers+1}']},
                        {'w': weights[f'w{num_layers+2}'], 'b': weights[f'b{num_layers+2}']},
                        {'w': weights[f'w{num_layers+3}'], 'b': weights[f'b{num_layers+3}']},
            )
            num_layers += 3
            if index == 2 and self.bypass:
                inputs_bypass = inputs
            elif index == 3 and self.bypass:
                inputs = inputs + inputs_bypass
        inputs = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')

        # The fire modules 5 to 8
        inputs_bypass = inputs
        for index in range(5, 9):
            inputs = fire_module(inputs,
                        bn_params,
                        {'w': weights[f'w{num_layers+1}'], 'b': weights[f'b{num_layers+1}']},
                        {'w': weights[f'w{num_layers+2}'], 'b': weights[f'b{num_layers+2}']},
                        {'w': weights[f'w{num_layers+3}'], 'b': weights[f'b{num_layers+3}']},
            )
            num_layers += 3
            if index == 6 and self.bypass:
                inputs_bypass = inputs
            elif index % 2 == 1 and self.bypass:
                inputs = inputs + inputs_bypass
        inputs = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
        
        # The fire module 9
        inputs_bypass = inputs
        inputs = fire_module(inputs,
                    bn_params,
                    {'w': weights[f'w{num_layers+1}'], 'b': weights[f'b{num_layers+1}']},
                    {'w': weights[f'w{num_layers+2}'], 'b': weights[f'b{num_layers+2}']},
                    {'w': weights[f'w{num_layers+3}'], 'b': weights[f'b{num_layers+3}']},
        )
        num_layers += 3
        if self.bypass:
            inputs = inputs + inputs_bypass

        # The output layer
        num_layers += 1
        inputs = tf.nn.conv2d(inputs, weights[f'w{num_layers}'], [1, 1, 1, 1], 'SAME') + weights[f'b{num_layers}']
        inputs = tf.nn.avg_pool(inputs, [1, inputs.shape[1], inputs.shape[2], 1], [1, 1, 1, 1], 'VALID')
        inputs = tf.squeeze(inputs, [1, 2])

        return inputs


class GoogLeNet(object):
    """
    Define the neural network for GoogLeNet.
    Reference:
        - C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich,
          "Going deeper with convolutions," in Proc. IEEE CVPR, 2015, pp. 1-9.
        - https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf
    """
    def __init__(self,
                initializer_w_bp,
                initializer_b_bp,
                initializer_w_ap,
                initializer_b_ap,
                datasource,
                num_classes,
                ):
        self.datasource = datasource
        self.name = 'googlenet'
        # Height × width × channels
        if self.datasource == 'tiny-imagenet':
            self.input_dims = [64, 64, 3]
        elif self.datasource == 'cifar-10':
            self.input_dims = [32, 32, 3]
        else:
            raise Exception(f"{self.datasource} is not supported.")
        # The number of filters in inception modules
        self.module_filters = [[64, 96, 128, 16, 32, 32],
                                [128, 128, 192, 32, 96, 64],
                                [192, 96, 208, 16, 48, 64],
                                [160, 112, 224, 24, 64, 64],
                                [128, 128, 256, 24, 64, 64],
                                [112, 144, 288, 32, 64, 64],
                                [256, 160, 320, 32, 128, 128],
                                [256, 160, 320, 32, 128, 128],
                                [384, 192, 384, 48, 128, 128]]
        self.num_modules = len(self.module_filters)
        # The number of classes
        self.output_dims = num_classes
        # Construct inputs
        self.inputs = self.construct_inputs()
        # Construct weights
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        # Compute the number of parameters
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        w_params = {
            'initializer': get_initializer(initializer_w, tf.float32),
            'dtype': tf.float32,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, tf.float32),
            'dtype': tf.float32,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            # Construct the weights of the input layers
            num_layers = 3
            weights['w1'] = tf.get_variable('w1', [7, 7, self.input_dims[2], 64], **w_params)
            weights['b1'] = tf.get_variable('b1', [64], **b_params)
            weights['w2'] = tf.get_variable('w2', [1, 1, 64, 64], **w_params)
            weights['b2'] = tf.get_variable('b2', [64], **b_params)
            weights['w3'] = tf.get_variable('w3', [3, 3, 64, 192], **w_params)
            weights['b3'] = tf.get_variable('b3', [192], **b_params)

            # Construct the weights of the inception modules
            num_Cin = 192
            for index in range(self.num_modules):
                filters = self.module_filters[index]
                # #1 × 1
                weights[f'w{num_layers+1}'] = tf.get_variable(f'w{num_layers+1}', [1, 1, num_Cin, filters[0]], **w_params)
                weights[f'b{num_layers+1}'] = tf.get_variable(f'b{num_layers+1}', [filters[0]], **b_params)
                # #3 × 3 reduce and #3 × 3
                weights[f'w{num_layers+2}'] = tf.get_variable(f'w{num_layers+2}', [1, 1, num_Cin, filters[1]], **w_params)
                weights[f'b{num_layers+2}'] = tf.get_variable(f'b{num_layers+2}', [filters[1]], **b_params)
                weights[f'w{num_layers+3}'] = tf.get_variable(f'w{num_layers+3}', [1, 1, filters[1], filters[2]], **w_params)
                weights[f'b{num_layers+3}'] = tf.get_variable(f'b{num_layers+3}', [filters[2]], **b_params)
                # #5 × 5 reduce and #5 × 5
                weights[f'w{num_layers+4}'] = tf.get_variable(f'w{num_layers+4}', [1, 1, num_Cin, filters[3]], **w_params)
                weights[f'b{num_layers+4}'] = tf.get_variable(f'b{num_layers+4}', [filters[3]], **b_params)
                weights[f'w{num_layers+5}'] = tf.get_variable(f'w{num_layers+5}', [1, 1, filters[3], filters[4]], **w_params)
                weights[f'b{num_layers+5}'] = tf.get_variable(f'b{num_layers+5}', [filters[4]], **b_params)
                # pool proj
                weights[f'w{num_layers+6}'] = tf.get_variable(f'w{num_layers+6}', [1, 1, num_Cin, filters[5]], **w_params)
                weights[f'b{num_layers+6}'] = tf.get_variable(f'b{num_layers+6}', [filters[5]], **b_params)
                # Update the number of channels
                num_Cin = filters[0] + filters[2] + filters[4] + filters[5]
                num_layers += 6
            
            # Construct the weights of the output layer
            num_layers += 1
            weights[f'w{num_layers}'] = tf.get_variable(f'w{num_layers}', [num_Cin, self.output_dims], **w_params)
            weights[f'b{num_layers}'] = tf.get_variable(f'b{num_layers}', [self.output_dims], **b_params)

        return weights
    
    def forward_pass(self, weights, inputs, is_train, trainable=True):
        def fire_module(inputs, bn_params, filt1, filt2, filt3, filt4, filt5, filt6):
            # #1 × 1
            path1 = tf.nn.conv2d(inputs, filt1['w'], [1, 1, 1, 1], 'SAME') + filt1['b']
            path1 = tf.layers.batch_normalization(path1, **bn_params)
            path1 = tf.nn.relu(path1)
            # #3 × 3 reduce and #3 × 3
            path2 = tf.nn.conv2d(inputs, filt2['w'], [1, 1, 1, 1], 'SAME') + filt2['b']
            path2 = tf.layers.batch_normalization(path2, **bn_params)
            path2 = tf.nn.relu(path2)
            path2 = tf.nn.conv2d(path2, filt3['w'], [1, 1, 1, 1], 'SAME') + filt3['b']
            path2 = tf.layers.batch_normalization(path2, **bn_params)
            path2 = tf.nn.relu(path2)
            # #5 × 5 reduce and #5 × 5
            path3 = tf.nn.conv2d(inputs, filt4['w'], [1, 1, 1, 1], 'SAME') + filt4['b']
            path3 = tf.layers.batch_normalization(path3, **bn_params)
            path3 = tf.nn.relu(path3)
            path3 = tf.nn.conv2d(path3, filt5['w'], [1, 1, 1, 1], 'SAME') + filt5['b']
            path3 = tf.layers.batch_normalization(path3, **bn_params)
            path3 = tf.nn.relu(path3)
            # #3 × 3 reduce and #3 × 3
            path4 = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 1, 1, 1], 'SAME')
            path4 = tf.nn.conv2d(path4, filt6['w'], [1, 1, 1, 1], 'SAME') + filt6['b']
            path4 = tf.layers.batch_normalization(path4, **bn_params)
            path4 = tf.nn.relu(path4)
            # Concatenate tensors from four paths
            inputs = tf.concat([path1] + [path2] + [path3] + [path4], -1)
            return inputs
        
        bn_params = {
                'training': is_train,
                'trainable': trainable,
            }
        
        # The input layers
        num_layers = 3
        inputs = tf.nn.conv2d(inputs, weights['w1'], [1, 2, 2, 1], 'SAME') + weights['b1']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
        inputs = tf.nn.conv2d(inputs, weights['w2'], [1, 1, 1, 1], 'SAME') + weights['b2']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w3'], [1, 1, 1, 1], 'SAME') + weights['b3']
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # The inception modules
        for _ in range(1, 3):
            inputs = fire_module(inputs,
                        bn_params,
                        {'w': weights[f'w{num_layers+1}'], 'b': weights[f'b{num_layers+1}']},
                        {'w': weights[f'w{num_layers+2}'], 'b': weights[f'b{num_layers+2}']},
                        {'w': weights[f'w{num_layers+3}'], 'b': weights[f'b{num_layers+3}']},
                        {'w': weights[f'w{num_layers+4}'], 'b': weights[f'b{num_layers+4}']},
                        {'w': weights[f'w{num_layers+5}'], 'b': weights[f'b{num_layers+5}']},
                        {'w': weights[f'w{num_layers+6}'], 'b': weights[f'b{num_layers+6}']},
            )
            num_layers += 6
        inputs = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
        for _ in range(3, 8):
            inputs = fire_module(inputs,
                        bn_params,
                        {'w': weights[f'w{num_layers+1}'], 'b': weights[f'b{num_layers+1}']},
                        {'w': weights[f'w{num_layers+2}'], 'b': weights[f'b{num_layers+2}']},
                        {'w': weights[f'w{num_layers+3}'], 'b': weights[f'b{num_layers+3}']},
                        {'w': weights[f'w{num_layers+4}'], 'b': weights[f'b{num_layers+4}']},
                        {'w': weights[f'w{num_layers+5}'], 'b': weights[f'b{num_layers+5}']},
                        {'w': weights[f'w{num_layers+6}'], 'b': weights[f'b{num_layers+6}']},
            )
            num_layers += 6
        inputs = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
        for _ in range(8, 10):
            inputs = fire_module(inputs,
                        bn_params,
                        {'w': weights[f'w{num_layers+1}'], 'b': weights[f'b{num_layers+1}']},
                        {'w': weights[f'w{num_layers+2}'], 'b': weights[f'b{num_layers+2}']},
                        {'w': weights[f'w{num_layers+3}'], 'b': weights[f'b{num_layers+3}']},
                        {'w': weights[f'w{num_layers+4}'], 'b': weights[f'b{num_layers+4}']},
                        {'w': weights[f'w{num_layers+5}'], 'b': weights[f'b{num_layers+5}']},
                        {'w': weights[f'w{num_layers+6}'], 'b': weights[f'b{num_layers+6}']},
            )
            num_layers += 6
        inputs = tf.nn.dropout(inputs, 0.6)

        # The output layer
        num_layers += 1
        # The average pooling
        inputs = tf.nn.avg_pool(inputs, [1, inputs.shape[1], inputs.shape[2], 1], [1, 1, 1, 1], 'VALID')
        inputs = tf.squeeze(inputs, [1, 2])
        # The fully connected layer
        inputs = tf.matmul(inputs, weights[f'w{num_layers}']) + weights[f'b{num_layers}']

        return inputs


class DenseNet(object):
    """
    Define the neural network for DenseNet.
    Reference:
        - G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger,
          "Densely connected convolutional networks," in Proc. IEEE CVPR, 2017, pp. 4700-4708.
        - https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
    """
    def __init__(self,
                initializer_w_bp,
                initializer_b_bp,
                initializer_w_ap,
                initializer_b_ap,
                datasource,
                num_classes,
                version,
                ):
        self.datasource = datasource
        # Height × width × channels
        if self.datasource == 'tiny-imagenet':
            self.input_dims = [64, 64, 3]
        elif self.datasource == 'cifar-10':
            self.input_dims = [32, 32, 3]
        else:
            raise Exception(f"{self.datasource} is not supported.")
        # The number of filters in dense blocks
        if version == 121:
            self.name = 'densenet-121'
            self.block_layers = [6, 12, 24, 16]
        elif version == 169:
            self.name = 'densenet-169'
            self.block_layers = [6, 12, 32, 32]
        elif version == 201:
            self.name = 'densenet-201'
            self.block_layers = [6, 12, 48, 32]
        elif version == 264:
            self.name = 'densenet-264'
            self.block_layers = [6, 12, 64, 48]
        self.num_blocks = len(self.block_layers)
        self.growth_rate = 32
        # The number of classes
        self.output_dims = num_classes
        # Construct inputs
        self.inputs = self.construct_inputs()
        # Construct weights
        self.weights_bp = self.construct_weights(initializer_w_bp, initializer_b_bp, False, 'bp')
        self.weights_ap = self.construct_weights(initializer_w_ap, initializer_b_ap, True, 'ap')
        # Compute the number of parameters
        self.num_params = sum([static_size(v) for v in self.weights_ap.values()])

    def construct_inputs(self):
        return {
            'input': tf.placeholder(tf.float32, [None] + self.input_dims),
            'label': tf.placeholder(tf.int32, [None]),
        }

    def construct_weights(self, initializer_w, initializer_b, trainable, scope):
        w_params = {
            'initializer': get_initializer(initializer_w, tf.float32),
            'dtype': tf.float32,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        b_params = {
            'initializer': get_initializer(initializer_b, tf.float32),
            'dtype': tf.float32,
            'trainable': trainable,
            'collections': [self.name, tf.GraphKeys.GLOBAL_VARIABLES],
        }
        weights = {}
        with tf.variable_scope(scope):
            # Construct the weights of the input layer
            num_layers = 1
            weights['w1'] = tf.get_variable('w1', [7, 7, self.input_dims[2], self.growth_rate], **w_params)
            weights['b1'] = tf.get_variable('b1', [self.growth_rate], **b_params)

            for index_outer in range(self.num_blocks):
                num_Cin = self.growth_rate
                # Construct the weights of the dense blocks
                for index_inner in range(self.block_layers[index_outer]):
                    weights[f'w{num_layers+1}'] = tf.get_variable(f'w{num_layers+1}', [1, 1, num_Cin, 4 * self.growth_rate], **w_params)
                    weights[f'b{num_layers+1}'] = tf.get_variable(f'b{num_layers+1}', [4 * self.growth_rate], **b_params)
                    weights[f'w{num_layers+2}'] = tf.get_variable(f'w{num_layers+2}', [3, 3, 4 * self.growth_rate, self.growth_rate], **w_params)
                    weights[f'b{num_layers+2}'] = tf.get_variable(f'b{num_layers+2}', [self.growth_rate], **b_params)
                    num_layers += 2
                    num_Cin += self.growth_rate
                
                # Construct the weights of the transition layers
                if index_outer < self.num_blocks - 1:
                    weights[f'w{num_layers+1}'] = tf.get_variable(f'w{num_layers+1}', [1, 1, num_Cin, self.growth_rate], **w_params)
                    weights[f'b{num_layers+1}'] = tf.get_variable(f'b{num_layers+1}', [self.growth_rate], **b_params)
                    num_layers += 1
        
            # Construct the weights of the ouput layer
            num_layers += 1
            weights[f'w{num_layers}'] = tf.get_variable(f'w{num_layers}', [num_Cin, self.output_dims], **w_params)
            weights[f'b{num_layers}'] = tf.get_variable(f'b{num_layers}', [self.output_dims], **b_params)

            return weights
    
    def forward_pass(self, weights, inputs, is_train, trainable=True):
        def dense_block(inputs, bn_params, num_layers, num_growth):
            # Define the dense blocks
            growth_inputs = inputs
            for _ in range(num_growth):
                inputs = tf.layers.batch_normalization(growth_inputs, **bn_params)
                inputs = tf.nn.relu(inputs)
                inputs = tf.nn.conv2d(inputs, weights[f'w{num_layers+1}'], [1, 1, 1, 1], 'SAME') + weights[f'b{num_layers+1}']
                inputs = tf.layers.batch_normalization(inputs, **bn_params)
                inputs = tf.nn.relu(inputs)
                inputs = tf.nn.conv2d(inputs, weights[f'w{num_layers+2}'], [1, 1, 1, 1], 'SAME') + weights[f'b{num_layers+2}']
                num_layers += 2
                growth_inputs = tf.concat([growth_inputs] + [inputs], -1)
            return growth_inputs, num_layers

        def transition_layer(inputs, bn_params, num_layers):
            # Define the transition layers
            inputs = tf.layers.batch_normalization(inputs, **bn_params)
            inputs = tf.nn.relu(inputs)
            inputs = tf.nn.conv2d(inputs, weights[f'w{num_layers+1}'], [1, 1, 1, 1], 'SAME') + weights[f'b{num_layers+1}']
            num_layers += 1
            return inputs, num_layers
                
        bn_params = {
                'training': is_train,
                'trainable': trainable,
            }
        
        # The input layer
        num_layers = 1
        inputs = tf.layers.batch_normalization(inputs, **bn_params)
        inputs = tf.nn.relu(inputs)
        inputs = tf.nn.conv2d(inputs, weights['w1'], [1, 2, 2, 1], 'SAME') + weights['b1']
        inputs = tf.nn.max_pool(inputs, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        for index in range(self.num_blocks):
            # The dense blocks
            inputs, num_layers = dense_block(inputs, bn_params, num_layers, self.block_layers[index])
            # The transition layers
            if index < self.num_blocks - 1:
                inputs, num_layers = transition_layer(inputs, bn_params, num_layers)
                inputs = tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        
        # The output layer
        num_layers += 1
        # The average pooling
        inputs = tf.nn.avg_pool(inputs, [1, inputs.shape[1], inputs.shape[2], 1], [1, 1, 1, 1], 'VALID')
        inputs = tf.squeeze(inputs, [1, 2])
        # The fully connected layer
        inputs = tf.matmul(inputs, weights[f'w{num_layers}']) + weights[f'b{num_layers}']

        return inputs