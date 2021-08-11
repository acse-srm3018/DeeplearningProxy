"""
Module for building network for Attention Recurrent residul U-net.

These model include the sequence of convolutional, residual, convLSTM
attention and deconvolution blocks.
"""
from blocks import *
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.layers import concatenate, TimeDistributed, RepeatVector, ConvLSTM2D
from keras.models import Model
import numpy as np


def create_vae(input_shape, depth):
    """
    Create VAE to create something new.

    Parameters:
    -------
    input_shape : Tuple
    depth : int
    Returns:
    -------
    encoder: encoder
    model : recurrnet R-UNET model

    """
    # Encoder
    input = Input(shape=input_shape, name='image')

    enc1 = conv_bn_relu(16, 3, 3, stride=(2, 2))(input)
    time_enc1 = RepeatConv(depth)(enc1)
    enc2 = conv_bn_relu(32, 3, 3, stride=(1, 1))(enc1)
    time_enc2 = RepeatConv(depth)(enc2)
    enc3 = conv_bn_relu(64, 3, 3, stride=(2, 2))(enc2)
    time_enc3 = RepeatConv(depth)(enc3)
    enc4 = conv_bn_relu(128, 3, 3, stride=(1, 1))(enc3)
    time_enc4 = RepeatConv(depth)(enc4)

    x = res_conv(128, 3, 3)(enc4)
    x = res_conv(128, 3, 3)(x)
    x = res_conv(128, 3, 3)(x)

    encoder = Model(input, x, name='encoder')

    x = RepeatConv(depth)(enc4)
    x = ConvLSTM2D(128, (3, 3), strides=(1, 1),
                   padding='same', activation='relu',
                   return_sequences=True)(x)

    x = time_res_conv(128, 3, 3)(x)
    x = time_res_conv(128, 3, 3)(x)
    dec4 = time_res_conv(128, 3, 3)(x)

    gating_4 = GatingSignal(dec4, 128)
    att4 = AttnGatingBlock(dec4, gating_4, 128)
    merge4 = concatenate([time_enc4, att4], axis=4)
    dec3 = time_dconv_bn_nolinear(128, 3, 3, stride=(1, 1))(merge4)
    gating_3 = GatingSignal(dec3, 128)
    att3 = AttnGatingBlock(dec3, gating_3, 128)
    merge3 = concatenate([time_enc3, att3], axis=4)
    dec2 = time_dconv_bn_nolinear(64, 3, 3, stride=(2, 2))(merge3)
    gating_2 = GatingSignal(dec2, 64)
    att2 = AttnGatingBlock(dec2, gating_2, 64)
    merge2 = concatenate([time_enc2, att2], axis=4)
    dec1 = time_dconv_bn_nolinear(32, 3, 3, stride=(1, 1))(merge2)
    gating_1 = GatingSignal(dec1, 32)
    att1 = AttnGatingBlock(dec1, gating_1, 32)
    merge1 = concatenate([time_enc1, att1], axis=4)
    dec0 = time_dconv_bn_nolinear(16, 3, 3, stride=(2, 2))(merge1)

    output = TimeDistributed(Conv2D(1, (3, 3), padding='same',
                                    activation=None))(dec0)
    print('output shape is ', K.int_shape(output))
    # Full net
    full_model = Model(input, output)

    return full_model, encoder
