import tensorflow as tf
import efficientnet.tfkeras as efn
import efficientnetv2 as efnv2
from tensorflow.keras import layers
from tensorflow.keras import Model
from .custom_layers import WFF
from .head import segmentation_head, box_head, class_head

from collections import namedtuple

StructConfig = namedtuple('Config', ('Backbone', 'BiFPN_W', 'BiFPN_D', 'Box_Repeat', 'Anchor_Scale', 'Branch'))

_efficientdet_config = {
    'EfficientDetD0': StructConfig('EfficientNetB0', 64, 3, 3, 4.,
                                   ('block3b_add', 'block5c_add', 'block7a_project_bn')),
    'EfficientDetD1': StructConfig('EfficientNetB1', 88, 4, 3, 4.,
                                   ('block3c_add', 'block5d_add', 'block7b_add')),
    'EfficientDetD2': StructConfig('EfficientNetB2', 112, 5, 3, 4.,
                                   ('block3c_add', 'block5d_add', 'block7b_add')),
    'EfficientDetD3': StructConfig('EfficientNetB3', 160, 6, 4, 4.,
                                   ('block3c_add', 'block5e_add', 'block7b_add')),
    'EfficientDetD4': StructConfig('EfficientNetB4', 224, 7, 4, 4.,
                                   ('block3d_add', 'block5f_add', 'block7b_add')),
    'EfficientDetD5': StructConfig('EfficientNetB5', 288, 7, 4, 4.,
                                   ('block3e_add', 'block5g_add', 'block7c_add')),
    'EfficientDetD6': StructConfig('EfficientNetB6', 384, 8, 5, 4.,
                                   ('block3f_add', 'block5h_add', 'block7c_add')),
    'EfficientDetD7': StructConfig('EfficientNetB6', 384, 8, 5, 5.,
                                   ('block3f_add', 'block5h_add', 'block7c_add')),
    'EfficientDetD7x': StructConfig('EfficientNetB7', 384, 8, 5, 4.,
                                    ('block3g_add', 'block5j_add', 'block7d_add')),

    'EfficientNetV2DS': StructConfig('EfficientNetV2_S', 224, 7, 4, 4.,
                                     ('fused_block3d_add', 'normal_block5i_add', 'normal_block6o_add')),
    'EfficientNetV2DM': StructConfig('EfficientNetV2_M', 288, 7, 4, 4.,
                                     ('fused_block3e_add', 'normal_block5n_add', 'normal_block7e_add')),
    'EfficientNetV2DL': StructConfig('EfficientNetV2_L', 384, 8, 5, 4.,
                                     ('fused_block3g_add', 'normal_block5s_add', 'normal_block7g_add')),
    'EfficientNetV2DXL': StructConfig('EfficientNetV2_XL', 384, 8, 5, 4.,
                                      ('fused_block3h_add', 'normal_block5x_add', 'normal_block7h_add')),
}


def bifpn_network(features, num_channels, activation='swish'):
    features = sorted(features, key=lambda x: x.shape[2])
    num_feature = len(features)
    prev_feature = None
    output_layers_input = []

    for i, feature in enumerate(features):
        if i == 0:
            prev_feature = layers.UpSampling2D(2, interpolation='bilinear')(feature)
            output_layers_input.append([feature])
            continue
        td_layer = WFF(num_channels, 3, strides=1, padding='same')([prev_feature, feature])
        td_layer = layers.BatchNormalization()(td_layer)
        td_layer = layers.Activation(activation=activation)(td_layer)

        if i < num_feature - 1:
            output_layers_input.append([td_layer, feature])
            prev_feature = layers.UpSampling2D(2, interpolation='bilinear')(td_layer)

    outputs = [td_layer]
    output = layers.MaxPooling2D(3, strides=2, padding='same')(td_layer)
    for i, output_in in enumerate(output_layers_input[::-1]):
        output = WFF(num_channels, 3, strides=1, padding='same')([*output_in, output])
        output = layers.BatchNormalization()(output)
        output = layers.Activation(activation=activation)(output)
        outputs.insert(0, output)
        if i != len(output_layers_input) - 1:
            output = layers.MaxPooling2D(3, strides=2, padding='same')(output)
    return outputs


def EfficientDet(model_name,
                 input_shape=(1024, 1024, 3),
                 classes=1000,
                 weights=None,
                 activation='swish',
                 use_p8=False,
                 aspect_ratios=(1., 2., 0.5),
                 num_scales=3
                 ):
    _imagenet_weight = weights if weights == 'imagenet' else None
    _config = _efficientdet_config[model_name]

    input_x = layers.Input(shape=input_shape)

    lib = efnv2 if 'V2' in _config.Backbone else efn
    backbone_net = lib.__getattribute__(_config.Backbone)(input_tensor=input_x,
                                                          input_shape=input_shape,
                                                          include_top=False,
                                                          weights=_imagenet_weight)

    # reset channels
    p3 = backbone_net.get_layer(_config.Branch[0]).output
    p3 = layers.Conv2D(_config.BiFPN_W, 1, padding='same')(p3)
    p3 = layers.BatchNormalization()(p3)

    p4 = backbone_net.get_layer(_config.Branch[1]).output
    p4 = layers.Conv2D(_config.BiFPN_W, 1, padding='same')(p4)
    p4 = layers.BatchNormalization()(p4)

    p5 = backbone_net.get_layer(_config.Branch[2]).output
    p5 = layers.Conv2D(_config.BiFPN_W, 1, padding='same')(p5)
    p5 = layers.BatchNormalization()(p5)

    p6 = layers.MaxPooling2D(3, strides=2, padding='same')(p5)
    p7 = layers.MaxPooling2D(3, strides=2, padding='same')(p6)

    p_layers = [p3, p4, p5, p6, p7]
    if use_p8:
        p_layers.append(layers.MaxPooling2D(3, strides=2, padding='same')(p7))

    for _ in range(_config.BiFPN_D):
        p_layers = bifpn_network(p_layers, _config.BiFPN_W, activation=activation)

    seg = segmentation_head(p_layers, _config.BiFPN_W, classes, activation=activation, use_conv=True)

    num_anchors = len(aspect_ratios) * num_scales

    box = box_head(p_layers,
                   num_anchors=num_anchors,
                   num_filters=_config.BiFPN_W,
                   repeats=_config.Box_Repeat,
                   activation=activation,
                   separable_conv=True)
    cls = class_head(p_layers,
                     classes,
                     num_anchors=num_anchors,
                     num_filters=_config.BiFPN_W,
                     activation=activation,
                     repeats=_config.Box_Repeat,
                     separable_conv=True,
                     dropout=0.5)
    # box_net = BoxNet(_config.BiFPN_W, 4, num_anchors=9, separable_conv=True)
    # class_net = ClassNet(_config.BiFPN_W, 4, num_classes=classes, num_anchors=9, separable_conv=True)
    #
    # cls = [class_net([feature, i]) for i, feature in enumerate(p_layers)]
    # cls = layers.Concatenate(axis=1, name='classification')(cls)
    # box = [box_net([feature, i]) for i, feature in enumerate(p_layers)]
    # box = layers.Concatenate(axis=1, name='regression')(box)

    model = Model(inputs=input_x, outputs=[seg, *box, *cls])
    if weights and weights != 'imagenet':
        model.load_weights(weights)

    return model


def EfficientDetD0(input_shape=(512, 512, 3),
                   classes=1000,
                   weights=None,
                   **kwargs):
    return EfficientDet('EfficientDetD0',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientDetD1(input_shape=(512, 512, 3),
                   classes=1000,
                   weights=None,
                   **kwargs):
    return EfficientDet('EfficientDetD1',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientDetD2(input_shape=(512, 512, 3),
                   classes=1000,
                   weights=None,
                   **kwargs):
    return EfficientDet('EfficientDetD2',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientDetD3(input_shape=(512, 512, 3),
                   classes=1000,
                   weights=None,
                   **kwargs):
    return EfficientDet('EfficientDetD3',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientDetD4(input_shape=(512, 512, 3),
                   classes=1000,
                   weights=None,
                   **kwargs):
    return EfficientDet('EfficientDetD4',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientDetD5(input_shape=(512, 512, 3),
                   classes=1000,
                   weights=None,
                   **kwargs):
    return EfficientDet('EfficientDetD5',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientDetD6(input_shape=(512, 512, 3),
                   classes=1000,
                   weights=None,
                   **kwargs):
    return EfficientDet('EfficientDetD6',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientDetD7(input_shape=(512, 512, 3),
                   classes=1000,
                   weights=None,
                   **kwargs):
    return EfficientDet('EfficientDetD7',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientDetD7x(input_shape=(512, 512, 3),
                    classes=1000,
                    weights=None,
                    **kwargs):
    return EfficientDet('EfficientDetD7x',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientNetV2DS(input_shape=(512, 512, 3),
                     classes=1000,
                     weights=None,
                     **kwargs):
    return EfficientDet('EfficientNetV2DS',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientNetV2DM(input_shape=(512, 512, 3),
                     classes=1000,
                     weights=None,
                     **kwargs):
    return EfficientDet('EfficientNetV2DM',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientNetV2DL(input_shape=(512, 512, 3),
                     classes=1000,
                     weights=None,
                     **kwargs):
    return EfficientDet('EfficientNetV2DL',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)


def EfficientNetV2DXL(input_shape=(512, 512, 3),
                      classes=1000,
                      weights=None,
                      **kwargs):
    return EfficientDet('EfficientNetV2DXL',
                        input_shape=input_shape,
                        classes=classes,
                        weights=weights,
                        **kwargs)
