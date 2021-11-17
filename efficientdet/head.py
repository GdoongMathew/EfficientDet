import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from efficientnetv2.utils import CONV_KERNEL_INITIALIZER
from .custom_layers import sep_conv_bn
from functools import partial


def segmentation_head(features,
                      num_filters,
                      classes,
                      activation='swish',
                      use_conv=False,
                      name='segmentation_head'):

    inputs = [layers.Input(shape=feature.shape[1:]) for feature in features]
    x = inputs[0]

    for in_x in inputs[1:]:

        upsample_layer = layers.Conv2DTranspose(num_filters, 3,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=CONV_KERNEL_INITIALIZER) \
            if use_conv else layers.UpSampling2D(2, interpolation='bilinear')
        x = upsample_layer(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation=activation)(x)
        x = layers.Concatenate()([x, in_x])

    upsample_layer = layers.Conv2DTranspose(num_filters, 3,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=CONV_KERNEL_INITIALIZER) \
        if use_conv else layers.UpSampling2D(2, interpolation='bilinear')

    x = upsample_layer(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation=activation)(x)

    # from deeplab v3+
    x = sep_conv_bn(x, 256, 'sep_conv0', depth_activation=True, activation=activation)
    x = sep_conv_bn(x, 256, 'sep_conv1', depth_activation=True, activation=activation)

    x = layers.UpSampling2D(4, interpolation='bilinear')(x)
    x = layers.Conv2D(classes, 1, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.Activation('softmax')(x)

    seg_model = Model(inputs=inputs, outputs=x, name=name)

    return seg_model(features)


def box_head(features,
             num_anchors=9,
             num_filters=32,
             repeats=4,
             separable_conv=True,
             activation='swish',
             name='box_head',
             dropout=None
             ):
    inputs = [layers.Input(shape=feature.shape[1:]) for feature in features]
    conv2d = partial(layers.SeparableConv2D,
                     kernel_size=3,
                     filters=num_filters,
                     depth_multiplier=1,
                     activation=None,
                     padding='same',
                     bias_initializer=tf.zeros_initializer(),
                     depthwise_initializer=CONV_KERNEL_INITIALIZER,
                     pointwise_initializer=CONV_KERNEL_INITIALIZER
                     ) if separable_conv else \
        partial(layers.Conv2D,
                kernel_size=3,
                filters=num_filters,
                activation=None,
                bias_initializer=tf.zeros_initializer(),
                padding='same')

    conv2ds = [conv2d() for _ in range(repeats)]
    box_out = layers.SeparableConv2D(
        4 * num_anchors,
        3,
        depth_multiplier=1,
        pointwise_initializer=CONV_KERNEL_INITIALIZER,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        activation=None,
        bias_initializer=tf.zeros_initializer(),
        padding='same',
        name='box_out_sepconv2d'
    ) if separable_conv else layers.Conv2D(
        4 * num_anchors,
        3,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        bias_initializer=tf.zeros_initializer(),
        activation=None,
        padding='same',
        name='box_out_conv2d'
    )

    outputs = []
    for in_x in inputs:
        for i in range(repeats):
            in_x = conv2ds[i](in_x)
            in_x = layers.BatchNormalization()(in_x)
            if activation:
                in_x = layers.Activation(activation=activation)(in_x)
            if dropout:
                in_x = layers.Dropout(dropout)(in_x)

        in_x = box_out(in_x)
        outputs.append(in_x)

    box_model = Model(inputs=inputs, outputs=outputs, name=name)
    return box_model(features)


def class_head(features,
               classes,
               num_anchors=9,
               num_filters=32,
               activation='swish',
               repeats=4,
               separable_conv=True,
               dropout=None,
               name='class_head'):

    inputs = [layers.Input(shape=feature.shape[1:]) for feature in features]
    conv2d = partial(layers.SeparableConv2D,
                     kernel_size=3,
                     filters=num_filters,
                     depth_multiplier=1,
                     activation=None,
                     padding='same',
                     depthwise_initializer=CONV_KERNEL_INITIALIZER,
                     pointwise_initializer=CONV_KERNEL_INITIALIZER
                     ) if separable_conv else \
        partial(layers.Conv2D,
                kernel_size=3,
                filters=num_filters,
                activation=None,
                padding='same')

    conv2ds = [conv2d(bias_initializer=tf.zeros_initializer()) for _ in range(repeats)]
    class_out = conv2d(
        classes * num_anchors,
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        name='class_out_conv2d'
    )

    outputs = []
    for in_x in inputs:
        for i in range(repeats):
            in_x = conv2ds[i](in_x)
            in_x = layers.BatchNormalization()(in_x)
            if activation:
                in_x = layers.Activation(activation=activation)(in_x)
            if dropout:
                in_x = layers.Dropout(dropout)(in_x)

        in_x = class_out(in_x)
        outputs.append(in_x)

    class_model = Model(inputs=inputs, outputs=outputs, name=name)
    return class_model(features)
