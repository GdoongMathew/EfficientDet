import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.utils import tf_utils


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
