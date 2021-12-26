import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.utils import tf_utils

from typing import Tuple


# Ported from DeeplabV3+ in https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py
def sep_conv_bn(x,
                filters,
                prefix,
                stride=1,
                kernel_size=3,
                rate=1,
                depth_activation=False,
                epsilon=1e-3,
                activation='relu'):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
            activation:
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = layers.Activation(tf.nn.relu)(x)
    x = layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                               padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation(activation=activation)(x)
    x = layers.Conv2D(filters, (1, 1), padding='same',
                      use_bias=False, name=prefix + '_pointwise')(x)
    x = layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = layers.Activation(activation=activation)(x)

    return x


class WFF(layers.SeparableConv2D):
    """
    Weighted Feature Fusion
    """

    def __init__(self, filters, kernel_size, epsilon=tf.keras.backend.epsilon(), *args, **kwargs):
        self.epsilon = epsilon
        super(WFF, self).__init__(filters, kernel_size, *args, **kwargs)
        self.input_spec = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if not isinstance(input_shape[0], tuple):
            raise ValueError('A WFF layer should be called on a list of inputs')

        batch_sizes = {s[0] for s in input_shape if s} - {None}
        if len(batch_sizes) > 1:
            raise ValueError(
                'Can not merge tensors with different '
                'batch sizes. Got tensors with shapes : ' + str(input_shape))

        for i, dim in enumerate(zip(*input_shape)):
            if i == 0:
                continue
            if dim.count(dim[0]) != len(dim):
                raise ValueError(f'Tensor shapes should be the same, given {input_shape}.')

        num_input = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_input, 1, 1, 1, 1),
                                 initializer=tf.keras.initializers.constant(1 / num_input),
                                 regularizer='l2',
                                 trainable=True,
                                 dtype=tf.float32)

        super(WFF, self).build(input_shape[0])
        self.input_spec = [self.input_spec] * num_input

    def compute_output_shape(self, input_shape):
        return super(WFF, self).compute_output_shape(input_shape[0])

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum(tf.multiply(inputs, w), axis=0)
        x = x / (tf.reduce_sum(x) + self.epsilon)
        x = super(WFF, self).call(x)
        return x

    def get_config(self):
        config = super(WFF, self).get_config()
        return {**config, 'epsilon': self.epsilon}


class ClipBbox(layers.Layer):
    def __init__(self, image_shape, *args, **kwargs):
        assert isinstance(image_shape, (tuple, list)) and len(image_shape) == 3
        self.height, self.width, _ = image_shape
        self.image_shape = image_shape
        super(ClipBbox, self).__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        """
        input a denormalized bboxes
        :param inputs:
        :param args:
        :param kwargs:
        :return:
        """
        x1 = tf.clip_by_value(inputs[..., 0], 0, self.width - 1)
        y1 = tf.clip_by_value(inputs[..., 1], 0, self.height - 1)
        x2 = tf.clip_by_value(inputs[..., 2], 0, self.width - 1)
        y2 = tf.clip_by_value(inputs[..., 3], 0, self.height - 1)
        return tf.stack([x1, y1, x2, y2], axis=-1)

    def get_config(self):
        config = super(ClipBbox, self).get_config()
        return {
            'image_shape': (self.height, self.width, 3),
            **config
        }


def _generate_anchors(size: int, ratios: Tuple[float], scales: Tuple[float]):
    """
    from xuannianz's implementation of generate_anchors.
    :param size:
    :param ratios:
    :param scales:
    :return:
    """
    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4), dtype=np.float32)
    anchors[:, 2:] = size * np.tile(np.repeat(scales, len(ratios))[None], (2, 1)).T

    area = anchors[:, 2] * anchors[:, 3]
    anchors[:, 2] = np.sqrt(area / np.tile(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.tile(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def _shift_anchors(anchors, stride, feature_map_shape):
    """
    produce shifted anchors based on feature_map shape and stride
    :param anchors:
    :param stride:
    :param feature_map_shape:
    :return:
    """
    shift_x = (np.arange(0, feature_map_shape[2]) + 0.5) * stride
    shift_y = (np.arange(0, feature_map_shape[1]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()
    )).T

    A = anchors.shape[0]
    K = shifts.shape[0]
    final_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    final_anchors = final_anchors.reshape((K * A, 4))
    return final_anchors


class DenormalizeBbox(layers.Layer):
    def __init__(self,
                 feature_maps_shape,
                 bbox_points=4,
                 anchor_sizes=(32, 64, 128, 256, 512),
                 anchor_strides=(8, 16, 32, 64, 128),
                 anchor_ratios=(1, 0.5, 2),
                 anchor_scales=(2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)),
                 *args, **kwargs):
        self.bbox_points = bbox_points

        self.anchors = np.zeros((0, 4), dtype=np.float32)
        for anchor_size, anchor_stride, feature_map_shape in zip(anchor_sizes, anchor_strides, feature_maps_shape):
            _anchors = _generate_anchors(anchor_size, anchor_ratios, anchor_scales)
            _anchors = _shift_anchors(_anchors, anchor_stride, feature_map_shape)
            self.anchors = np.append(self.anchors, _anchors, axis=0)

        self.anchors = np.expand_dims(self.anchors, axis=0)
        super(DenormalizeBbox, self).__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        bbox_deltas = inputs[..., :self.bbox_points]

        cxa = (self.anchors[..., 0] + self.anchors[..., 2]) / 2
        cya = (self.anchors[..., 1] + self.anchors[..., 3]) / 2
        wa = self.anchors[..., 2] - self.anchors[..., 0]
        ha = self.anchors[..., 3] - self.anchors[..., 1]
        ty, tx, th, tw = bbox_deltas[..., 0], bbox_deltas[..., 1], bbox_deltas[..., 2], bbox_deltas[..., 3]

        w = tf.exp(tw) * wa
        h = tf.exp(th) * ha
        cy = ty * ha + cya
        cx = tx * wa + cxa
        ymin = cy - h / 2.
        xmin = cx - w / 2.
        ymax = cy + h / 2.
        xmax = cx + w / 2.
        return tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    def get_config(self):
        config = super(DenormalizeBbox, self).get_config()
        return {'feature_maps_shape': self.feature_maps_shape,
                'bbox_points': self.bbox_points,
                'anchor_ratios': self.anchor_ratios,
                'anchor_scales': self.anchor_scales,
                **config}

    def compute_output_shape(self, input_shape):
        return input_shape[-1]
