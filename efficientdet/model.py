import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Model
from tensorflow.python.keras.utils import tf_utils
from collections import namedtuple

config = namedtuple('Config', ('Backbone', 'BiFPN_W', 'BiFPN_D'))

_efficientdet_config = {
    'EfficientDetD0': config('EfficientNetB0', 64, 3),
    'EfficientDetD1': config('EfficientNetB1', 88, 4),
    'EfficientDetD2': config('EfficientNetB2', 112, 5),
    'EfficientDetD3': config('EfficientNetB3', 160, 6),
    'EfficientDetD4': config('EfficientNetB4', 224, 7),
    'EfficientDetD5': config('EfficientNetB5', 288, 7),
    'EfficientDetD6': config('EfficientNetB6', 384, 8),
    'EfficientDetD7': config('EfficientNetB7', 384, 8),
}


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


def bifpn_network(features, num_channels, activation='swish'):
    features = sorted(features, key=lambda x: x.shape[2])
    num_feature = len(features)
    prev_feature = None
    output_layers_input = []

    for i, feature in enumerate(features):
        if i == 0:
            prev_feature = UpSampling2D(2, interpolation='bilinear')(feature)
            output_layers_input.append([feature])
            continue
        td_layer = WFF(num_channels, 3, strides=1, padding='same')([prev_feature, feature])
        td_layer = BatchNormalization()(td_layer)
        td_layer = Activation(activation=activation)(td_layer)

        if i < num_feature - 1:
            output_layers_input.append([td_layer, feature])
            prev_feature = UpSampling2D(2, interpolation='bilinear')(td_layer)

    outputs = [td_layer]
    output = MaxPooling2D(3, strides=2, padding='same')(td_layer)
    for i, output_in in enumerate(output_layers_input[::-1]):
        output = WFF(num_channels, 3, strides=1, padding='same')([*output_in, output])
        output = BatchNormalization()(output)
        output = Activation(activation=activation)(output)
        outputs.insert(0, output)
        if i != len(output_layers_input) - 1:
            output = MaxPooling2D(3, strides=2, padding='same')(output)
    return outputs


def segmentation_head(features, num_filters, num_class, activation='swish'):
    x = features[0]
    for feature in features[1:]:
        x = Conv2DTranspose(num_filters, 3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)
        x = Concatenate()([x, feature])

    x = Conv2DTranspose(num_class, 3, strides=2, padding='same')(x)
    return x


def EfficientDet(model_name,
                 input_shape=(1024, 1024, 3),
                 classes=1000,
                 weights=None,
                 activation='swish'):
    _imagenet_weight = weights if weights == 'imagenet' else None
    _config = _efficientdet_config[model_name]

    input_x = Input(shape=input_shape)
    backbone_net = efn.__getattribute__(_config.Backbone)(input_tensor=input_x,
                                                          input_shape=input_shape,
                                                          include_top=False,
                                                          weights=_imagenet_weight)

    # reset channels
    p3 = backbone_net.get_layer('block3d_add').output
    p3 = Conv2D(_config.BiFPN_W, 1, padding='same')(p3)
    p3 = BatchNormalization()(p3)

    p4 = backbone_net.get_layer('block5f_add').output
    p4 = Conv2D(_config.BiFPN_W, 1, padding='same')(p4)
    p4 = BatchNormalization()(p4)

    p5 = backbone_net.get_layer('block7b_add').output
    p5 = Conv2D(_config.BiFPN_W, 1, padding='same')(p5)
    p5 = BatchNormalization()(p5)

    p6 = MaxPooling2D(3, strides=2, padding='same')(p5)
    p7 = MaxPooling2D(3, strides=2, padding='same')(p6)

    p_layers = [p3, p4, p5, p6, p7]
    for _ in range(_config.BiFPN_D):
        p_layers = bifpn_network(p_layers, _config.BiFPN_W)

    x = segmentation_head(p_layers, _config.BiFPN_W, classes)
    x = UpSampling2D(4, interpolation='bilinear')(x)
    x = Activation('softmax')(x)

    model = Model(inputs=input_x, outputs=x)
    if weights and weights != 'imagenet':
        model.load_weights(weights)

    return model

if __name__ == '__main__':
    EfficientDet('EfficientDetD4').summary()