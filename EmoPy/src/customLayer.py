from keras.engine import Layer
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, ConvLSTM2D, Conv3D, MaxPooling2D, Dropout, \
    MaxPooling3D, K

class SliceLayer(Layer):
    def __init__(self, start=0, items=8, **kwargs):
        super(SliceLayer, self).__init__(**kwargs)
        self.start = start
        self.items = items
        self.trainable = False

    def build(self, input_shape):
        super(SliceLayer, self).build(input_shape)

    def call(self, x):
        return x[:, :, :, self.start: self.start + self.items]

    def compute_output_shape(self, input_shape):
        height, width, in_channels = input_shape[1:]
        return input_shape[0], height, width, self.items

    def get_config(self):
        config = {'start': self.start, 'items': self.items}
        base_config = super(SliceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ChannelShuffle(Layer):
    def __init__(self, groups=None, groups_factor=8, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        self.groups = groups
        self.groups_factor = groups_factor
        self.trainable = False

    def build(self, input_shape):
        super(ChannelShuffle, self).build(input_shape)

    def call(self, x):
        height, width, in_channels = x.shape.as_list()[1:]

        if self.groups is None:
            if in_channels % self.groups_factor:
                raise ValueError("%s %% %s" % (in_channels, self.groups_factor))

            self.groups = in_channels // self.groups_factor

        channels_per_group = in_channels // self.groups

        x = K.reshape(x, [-1, height, width, self.groups, channels_per_group])
        x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
        x = K.reshape(x, [-1, height, width, in_channels])

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'groups': self.groups, 'groups_factor': self.groups_factor}
        base_config = super(ChannelShuffle, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PadZeros(Layer):
    def __init__(self, diff, **kwargs):
        super(PadZeros, self).__init__(**kwargs)
        self.diff = diff
        self.trainable = False

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(PadZeros, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        import tensorflow as tf
        return tf.pad(x, ((0, 0), (0, 0), (0, 0), (0, self.diff)), mode='CONSTANT')

    def compute_output_shape(self, input_shape):
        batch, b_width, b_height, b_channels = input_shape
        return batch, b_width, b_height, b_channels + self.diff

    def get_config(self):
        config = {'diff': self.diff}
        base_config = super(PadZeros, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))