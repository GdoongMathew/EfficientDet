import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.python.keras.utils import tf_utils
from collections import namedtuple

_efficientdet_config = namedtuple('EfficienDet_Config', ('Backbone', 'WiFPN_W', 'WiFPN_D'))


class WFF(SeparableConv2D):
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
            if i == 0: continue
            if dim.count(dim[0]) != len(dim):
                raise ValueError(f'Tensor shapes should be the same, given {input_shape}.')

        num_input = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_input, ),
                                 initializer=tf.keras.initializers.constant(1 / num_input),
                                 trainable=True,
                                 dtype=tf.float32)

        super(WFF, self).build(input_shape[0])
        self.input_spec = [self.input_spec] * num_input

    def compute_output_shape(self, input_shape):
        return super(WFF, self).compute_output_shape(input_shape[0])

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(x) + self.epsilon)
        x = super(WFF, self).call(x)
        return x

    def get_config(self):
        config = super(WFF, self).get_config()
        return {**config, 'epsilon': self.epsilon}




