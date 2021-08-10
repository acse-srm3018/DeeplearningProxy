"""import required libraries and modules."""

import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.merge import add
# from keras.engine import InputSpec
from keras.layers import InputSpec
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers import BatchNormalization, ConvLSTM2D
from keras.layers import TimeDistributed, Reshape, RepeatVector
from keras import regularizers


reg_weights = 0.00001


def conv_bn_relu(filter_num, row_num, col_num, stride):
    """
    Create Convolutional Batch Norm layer.

    Parameters:
    ---------
    filter_num : int
    number of filters to use in convolution layer.
    row_num : int
    number of row
    col_num : int
    number of column
    stride : int
    size of stride
    Returns:
    ---------
    conv_func
    """
    def conv_func(x):
        x = Conv2D(filter_num, (row_num, col_num),
                   strides=stride,
                   padding='same',
                   kernel_regularizer=regularizers.l2(reg_weights))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    return conv_func


def time_conv_bn_relu(filter_num, row_num, col_num, stride):
    """
    Create Convolutional Batch Norm layer.

    Parameters:
    ---------
    filter_num : int
    number of filters to use in convolution layer.
    row_num : int
    number of row
    col_num : int
    number of column
    stride : int
    size of stride
    Returns:
    ---------
    conv_func
    """
    def conv_func(x):
        x = TimeDistributed(Conv2D(filter_num, (row_num, col_num),
                            strides=stride,
                            padding='same',
                            kernel_regularizer=regularizers.
                            l2(reg_weights)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation("relu"))(x)
        return x

    return conv_func


def res_conv(filter_num, row_num, col_num, stride=(1, 1)):
    """
    Create Convolutional Batch Norm layer.

    Parameters:
    ---------
    filter_num : int
    number of filters to use in convolution layer.
    row_num : int
    number of row
    col_num : int
    number of column
    stride : int
    size of stride
    *default = (1,1)
    Returns:
    ---------
    conv_func
    """
    def _res_func(x):
        identity = x

        a = Conv2D(filter_num, (row_num, col_num),
                   strides=stride, padding='same',
                   kernel_regularizer=regularizers.l2(reg_weights))(x)
        a = BatchNormalization()(a)
        a = Activation("relu")(a)
        a = Conv2D(filter_num, (row_num, col_num),
                   strides=stride, padding='same',
                   kernel_regularizer=regularizers.l2(reg_weights))(a)
        y = BatchNormalization()(a)

        return add([identity, y])

    return _res_func


def time_res_conv(filter_num, row_num, col_num, stride=(1, 1)):
    """
    Create Convolutional Batch Norm layer.

    Parameters:
    ---------
    filter_num : int
    number of filters to use in convolution layer.
    row_num : int
    number of row
    col_num : int
    number of column
    stride : int
    size of stride
    Returns:
    ---------
    conv_func
    """
    def _res_func(x):
        identity = x

        a = TimeDistributed(Conv2D(filter_num, (row_num, col_num),
                                   strides=stride, padding='same',
                                   kernel_regularizer=regularizers.
                                   l2(reg_weights)))(x)
        a = TimeDistributed(BatchNormalization())(a)
        a = TimeDistributed(Activation("relu"))(a)
        a = TimeDistributed(Conv2D(filter_num, (row_num, col_num),
                                   strides=stride, padding='same',
                                   kernel_regularizer=regularizers.
                                   l2(reg_weights)))(x)
        y = TimeDistributed(BatchNormalization())(a)

        return add([identity, y])

    return _res_func

class AttentionBlock(nn.Layer):
    """Create Attention block class."""
    def __init__(self, F_g, F_l, F_out):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2D(F_g, F_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(F_out))

        self.W_x = nn.Sequential(
            nn.Conv2D(F_l, F_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(F_out))

        self.psi = nn.Sequential(
            nn.Conv2D(F_out, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(1), nn.Sigmoid())

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        res = x * psi
        return res
        
        
def dconv_bn_nolinear(nb_filter, nb_row, nb_col, stride=(2, 2),
                      activation="relu"):
    """
    Create convolutional Batch Norm layer in decoders.

    Parameters:
    ---------
    filter_num : int
    number of filters to use in convolution layer.
    row_num : int
    number of row
    col_num : int
    number of column
    stride : int
    size of stride
    Returns:
    ---------
    dconv_bn
    """
    def _dconv_bn(x):
        x = UnPooling2D(size=stride)(x)
        x = ReflectionPadding2D(padding=(int(nb_row/2), int(nb_col/2)))(x)
        x = Conv2D(nb_filter, (nb_row, nb_col), padding='valid',
                   kernel_regularizer=regularizers.l2(reg_weights))(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    return _dconv_bn


def time_dconv_bn_nolinear(nb_filter, nb_row, nb_col,
                           stride=(2, 2), activation="relu"):
    """
    Create time convolutional Batch Norm layer in decoders.

    Parameters:
    ---------
    filter_num : int
    number of filters to use in convolution layer.
    row_num : int
    number of row
    col_num : int
    number of column
    stride : int
    size of stride
    Returns:
    ---------
    dconv_bn
    """
    def _dconv_bn(x):
        x = TimeDistributed(UnPooling2D(size=stride))(x)
        x = TimeDistributed(ReflectionPadding2D(padding=(int(nb_row/2),
                            int(nb_col/2))))(x)
        x = TimeDistributed(Conv2D(nb_filter, (nb_row, nb_col),
                                   padding='valid',
                                   kernel_regularizer=regularizers.
                                   l2(reg_weights)))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation(activation))(x)
        return x

    return _dconv_bn


class ReflectionPadding2D(Layer):
    """class for reflectionPadding2D."""

    def __init__(self, padding=(1, 1), data_format="channels_last", **kwargs):
        """
        Construct class parameters.

        parameters:
        -------
        padding
        dim_ordering
        """
        super(ReflectionPadding2D, self).__init__(**kwargs)

        if data_format == 'channels_last':
            dim_ordering = K.image_data_format()

        self.padding = padding
        if isinstance(padding, dict):
            if set(padding.keys()) <= {'top_pad', 'bottom_pad',
                                       'left_pad', 'right_pad'}:
                self.top_pad = padding.get('top_pad', 0)
                self.bottom_pad = padding.get('bottom_pad', 0)
                self.left_pad = padding.get('left_pad', 0)
                self.right_pad = padding.get('right_pad', 0)
            else:
                raise ValueError('Unexpected key'
                                 'found in `padding` dictionary.'
                                 'Keys have to be in {"top_pad", "bottom_pad",'
                                 '"left_pad", "right_pad"}.'
                                 'Found: ' + str(padding.keys()))
        else:
            padding = tuple(padding)
            if len(padding) == 2:
                self.top_pad = padding[0]
                self.bottom_pad = padding[0]
                self.left_pad = padding[1]
                self.right_pad = padding[1]
            elif len(padding) == 4:
                self.top_pad = padding[0]
                self.bottom_pad = padding[1]
                self.left_pad = padding[2]
                self.right_pad = padding[3]
            else:
                raise TypeError('`padding` should be tuple of int '
                                'of length 2 or 4, or dict. '
                                'Found: ' + str(padding))

        # if data_format not in {'channels_last'}:
        #     raise ValueError('data_format must be in {"channels_last"}.')
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]

    def call(self, x, mask=None):
        """Call x to apply padding."""
        top_pad = self.top_pad
        bottom_pad = self.bottom_pad
        left_pad = self.left_pad
        right_pad = self.right_pad

        paddings = [[0, 0], [left_pad, right_pad],
                    [top_pad, bottom_pad], [0, 0]]

        return tf.pad(x, paddings, mode='REFLECT', name=None)

    def compute_output_shape(self, input_shape):
        """
        Compute the shape of output.

        Parameters:
        --------
        input_shape: Tuple
        shape of input
        """
        if self.data_format == 'channels_last':
            rows = input_shape[1] + self.top_pad + self.bottom_pad
            cols = input_shape[2] + self.left_pad + self.right_pad

            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])
        else:
            raise ValueError('Invalid data_format:', self.data_format)

    def get_config(self):
        """Get the Configure."""
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UnPooling2D(UpSampling2D):
    """Unpool 2D from 2D upsampling."""

    def __init__(self, size=(2, 2)):
        """Construct size."""
        super(UnPooling2D, self).__init__(size)

    def call(self, x, mask=None):
        """Call th x data."""
        shapes = x.get_shape().as_list()
        w = self.size[0] * shapes[1]
        h = self.size[1] * shapes[2]
        return tf.image.resize(x, (w, h))


class InstanceNormalize(Layer):
    """Normalization Instance of class."""

    def __init__(self, **kwargs):
        """Initialize the keyaarguments."""
        super(InstanceNormalize, self).__init__(**kwargs)
        self.epsilon = 1e-3

    def call(self, x, mask=None):
        """Call mean and variance for normalization."""
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, self.epsilon)))

    def compute_output_shape(self, input_shape):
        """Compute the shape of output."""
        return input_shape


class RepeatConv(Layer):
    """
    Repeats the input n times.

    Example:
    -------
        model = Sequential()
        model.add(Dense(32, input_dim=32))
        now: model.output_shape == (None, 32)
        note: `None` is the batch dimension
        model.add(RepeatVector(3))
        now: model.output_shape == (None, 3, 32)

    Arguments
    ---------
        n: integer, repetition factor.
    Input shape
    ----------
        4D tensor of shape `(num_samples, w, h, c)`.
    Output shape
    -----------
        5D tensor of shape `(num_samples, n, w, h, c)`.
    """

    def __init__(self, n, **kwargs):
        """Initialize the class parameters."""
        super(RepeatConv, self).__init__(**kwargs)
        self.n = n
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return (input_shape[0], self.n, input_shape[1],
                input_shape[2], input_shape[3])

    def call(self, inputs):
        """Call the inputs."""
        x = K.expand_dims(inputs, 1)
        pattern = tf.stack([1, self.n, 1, 1, 1])
        return K.tile(x, pattern)

    def get_config(self):
        """Get configure."""
        config = {'n': self.n}
        base_config = super(RepeatConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
