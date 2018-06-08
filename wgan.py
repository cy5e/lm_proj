import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, layer_norm
from tensorflow.python.ops.image_ops_impl import ResizeMethod
from tensorflow.python.ops.nn_ops import leaky_relu
from utils.network_summary import count_parameters



class Generator:
    def __init__(self, layer_sizes, layer_padding, batch_size, num_channels=1,
                 inner_layers=0, name="g"):
        """
        Initialize a generator.
        :param layer_sizes: A list with the filter sizes for each MultiLayer e.g. [64, 64, 128, 128]
        :param layer_padding: A list with the padding type for each layer e.g. ["SAME", "SAME", "SAME", "SAME"]
        :param batch_size: An integer indicating the batch size
        :param num_channels: An integer indicating the number of input channels
        :param inner_layers: An integer indicating the number of inner layers per MultiLayer
        """
        self.reuse = False
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.layer_sizes = layer_sizes
        self.layer_padding = layer_padding
        self.inner_layers = inner_layers
        self.conv_layer_num = 0
        self.build = True
        self.name = name

    def upscale(self, x, h_size, w_size):
        """
        Upscales an image using nearest neighbour
        :param x: Input image
        :param h_size: Image height size
        :param w_size: Image width size
        :return: Upscaled image
        """
        [b, h, w, c] = [int(dim) for dim in x.get_shape()]

        return tf.image.resize_nearest_neighbor(x, (h_size, w_size))

    def conv_layer(self, inputs, num_filters, filter_size, strides, activation=None,
                   transpose=False, w_size=None, h_size=None):
        """
        Add a convolutional layer to the network.
        :param inputs: Inputs to the conv layer.
        :param num_filters: Num of filters for conv layer.
        :param filter_size: Size of filter.
        :param strides: Stride size.
        :param activation: Conv layer activation.
        :param transpose: Whether to apply upscale before convolution.
        :param w_size: Used only for upscale, w_size to scale to.
        :param h_size: Used only for upscale, h_size to scale to.
        :return: Convolution features
        """
        self.conv_layer_num += 1
        if transpose:
            outputs = self.upscale(inputs, h_size=h_size, w_size=w_size)
            outputs = tf.layers.conv2d_transpose(outputs, num_filters, filter_size,
                                                 strides=strides,
                                       padding="SAME", activation=activation)
        elif not transpose:
            outputs = tf.layers.conv2d(inputs, num_filters, filter_size, strides=strides,
                                                 padding="SAME", activation=activation)
        return outputs

    def resize_batch(self, batch_images, size):

        """
        Resize image batch using nearest neighbour
        :param batch_images: Image batch
        :param size: Size to upscale to
        :return: Resized image batch.
        """
        images = tf.image.resize_images(batch_images, size=size, method=ResizeMethod.NEAREST_NEIGHBOR)

        return images

    def add_encoder_layer(self, input, name, training, dropout_rate, layer_to_skip_connect, local_inner_layers,
                          num_features, dim_reduce=False):

        """
        Adds a resnet encoder layer.
        :param input: The input to the encoder layer
        :param training: Flag for training or validation
        :param dropout_rate: A float or a placeholder for the dropout rate
        :param layer_to_skip_connect: Layer to skip-connect this layer to
        :param local_inner_layers: A list with the inner layers of the current Multi-Layer
        :param num_features: Number of feature maps for the convolutions
        :param dim_reduce: Boolean value indicating if this is a dimensionality reducing layer or not
        :return: The output of the encoder layer
        """
        [b1, h1, w1, d1] = input.get_shape().as_list()

        if len(layer_to_skip_connect) >= 2:
            layer_to_skip_connect = layer_to_skip_connect[-2]
        else:
            layer_to_skip_connect = None

        if layer_to_skip_connect is not None:
            [b0, h0, w0, d0] = layer_to_skip_connect.get_shape().as_list()
            if h0 > h1:
                skip_connect_layer = self.conv_layer(layer_to_skip_connect, int(layer_to_skip_connect.get_shape()[3]),
                                                     [3, 3], strides=(2, 2))
            else:
                skip_connect_layer = layer_to_skip_connect
            current_layers = [input, skip_connect_layer]
        else:
            current_layers = [input]

        current_layers.extend(local_inner_layers)
        current_layers = remove_duplicates(current_layers)
        outputs = tf.concat(current_layers, axis=3)

        if dim_reduce:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(2, 2))
            outputs = leaky_relu(outputs)
            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=True)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        else:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1))
            outputs = leaky_relu(features=outputs)
            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=True)

        return outputs

    def add_decoder_layer(self, input, name, training, dropout_rate, layer_to_skip_connect, local_inner_layers,
                          num_features, dim_upscale=False, h_size=None, w_size=None):

        """
        Adds a resnet decoder layer.
        :param input: Input features
        :param name: Layer Name
        :param training: Training placeholder or boolean flag
        :param dropout_rate: Float placeholder or float indicating the dropout rate
        :param layer_to_skip_connect: Layer to skip connect to.
        :param local_inner_layers: A list with the inner layers of the current MultiLayer
        :param num_features: Num feature maps for convolution
        :param dim_upscale: Dimensionality upscale
        :param h_size: Height to upscale to
        :param w_size: Width to upscale to
        :return: The output of the decoder layer
        """
        [b1, h1, w1, d1] = input.get_shape().as_list()
        if len(layer_to_skip_connect) >= 2:
            layer_to_skip_connect = layer_to_skip_connect[-2]
        else:
            layer_to_skip_connect = None

        if layer_to_skip_connect is not None:
            [b0, h0, w0, d0] = layer_to_skip_connect.get_shape().as_list()

            if h0 < h1:
                skip_connect_layer = self.conv_layer(layer_to_skip_connect,
                                                     int(layer_to_skip_connect.get_shape()[3]),
                                                     [3, 3], strides=(1, 1),
                                                     transpose=True,
                                                     h_size=h_size,
                                                     w_size=w_size)
            else:
                skip_connect_layer = layer_to_skip_connect
            current_layers = [input, skip_connect_layer]
        else:
            current_layers = [input]

        current_layers.extend(local_inner_layers)
        current_layers = remove_duplicates(current_layers)
        outputs = tf.concat(current_layers, axis=3)

        if dim_upscale:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1),
                                      transpose=True, w_size=w_size, h_size=h_size)
            outputs = leaky_relu(features=outputs)
            outputs = batch_norm(outputs,
                                 decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=True)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        else:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1),
                                       transpose=False)
            outputs = leaky_relu(features=outputs)
            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                 center=True, is_training=training,
                                 renorm=True)

        return outputs

    def __call__(self, z_inputs, conditional_input, training=False, dropout_rate=0.0):
        """
        Apply network on data.
        :param z_inputs: Random noise to inject [batch_size, z_dim]
        :param conditional_input: A batch of images to use as conditionals [batch_size, height, width, channels]
        :param training: Training placeholder or boolean
        :param dropout_rate: Dropout rate placeholder or float
        :return: Returns x_g (generated images), encoder_layers(encoder features), decoder_layers(decoder features)
        """
        conditional_input = tf.convert_to_tensor(conditional_input)
        with tf.variable_scope(self.name, reuse=self.reuse):
            # reshape from inputs
            outputs = conditional_input
            encoder_layers = []
            current_layers = [outputs]
            with tf.variable_scope('conv_layers'):

                for i, layer_size in enumerate(self.layer_sizes):
                    encoder_inner_layers = [outputs]
                    with tf.variable_scope('g_conv{}'.format(i)):
                        if i==0: #first layer is a single conv layer instead of MultiLayer for best results
                            outputs = self.conv_layer(outputs, num_filters=64,
                                                      filter_size=(3, 3), strides=(2, 2))
                            outputs = leaky_relu(features=outputs)
                            outputs = batch_norm(outputs, decay=0.99, scale=True,
                                                 center=True, is_training=training,
                                                 renorm=True)
                            current_layers.append(outputs)
                            encoder_inner_layers.append(outputs)
                        else:
                            for j in range(self.inner_layers[i]): #Build the inner Layers of the MultiLayer
                                outputs = self.add_encoder_layer(input=outputs,
                                                                 training=training,
                                                                 name="encoder_layer_{}_{}".format(i, j),
                                                                 layer_to_skip_connect=current_layers,
                                                                 num_features=self.layer_sizes[i],
                                                                 dim_reduce=False,
                                                                 local_inner_layers=encoder_inner_layers,
                                                                 dropout_rate=dropout_rate)
                                encoder_inner_layers.append(outputs)
                                current_layers.append(outputs)
                            #add final dim reducing conv layer for this MultiLayer
                            outputs = self.add_encoder_layer(input=outputs, name="encoder_layer_{}".format(i),
                                                             training=training, layer_to_skip_connect=current_layers,
                                                             local_inner_layers=encoder_inner_layers,
                                                             num_features=self.layer_sizes[i],
                                                             dim_reduce=True, dropout_rate=dropout_rate)
                            current_layers.append(outputs)
                        encoder_layers.append(outputs)

            g_conv_encoder = outputs

            with tf.variable_scope("vector_expansion"):  # Used for expanding the z injected noise to match the
                                                         # dimensionality of the various decoder MultiLayers, injecting
                                                         # noise into multiple decoder layers in a skip-connection way
                                                         # improves quality of results. We inject in the first 3 decode
                                                         # multi layers
                num_filters = 8
                z_layers = []
                concat_shape = [layer_shape.get_shape().as_list() for layer_shape in encoder_layers]

                for i in range(len(self.inner_layers)):
                    h = concat_shape[len(encoder_layers) - 1 - i][1]
                    w = concat_shape[len(encoder_layers) - 1 - i][1]
                    z_dense = tf.layers.dense(z_inputs, h * w * num_filters)
                    z_reshape_noise = tf.reshape(z_dense, [self.batch_size, h, w, num_filters])
                    num_filters /= 2
                    num_filters = int(num_filters)
                    print(z_reshape_noise)
                    z_layers.append(z_reshape_noise)

            outputs = g_conv_encoder
            decoder_layers = []
            current_layers = [outputs]
            with tf.variable_scope('g_deconv_layers'):
                for i in range(len(self.layer_sizes)+1):
                    if i<3: #Pass the injected noise to the first 3 decoder layers for sharper results
                        outputs = tf.concat([z_layers[i], outputs], axis=3)
                        current_layers[-1] = outputs
                    idx = len(self.layer_sizes) - 1 - i
                    num_features = self.layer_sizes[idx]
                    inner_layers = self.inner_layers[idx]
                    upscale_shape = encoder_layers[idx].get_shape().as_list()
                    if idx<0:
                        num_features = self.layer_sizes[0]
                        inner_layers = self.inner_layers[0]
                        outputs = tf.concat([outputs, conditional_input], axis=3)
                        upscale_shape = conditional_input.get_shape().as_list()

                    with tf.variable_scope('g_deconv{}'.format(i)):
                        decoder_inner_layers = [outputs]
                        for j in range(inner_layers):
                            if i==0 and j==0:
                                outputs = self.add_decoder_layer(input=outputs,
                                                                 name="decoder_inner_conv_{}_{}"
                                                                 .format(i, j),
                                                                 training=training,
                                                                 layer_to_skip_connect=current_layers,
                                                                 num_features=num_features,
                                                                 dim_upscale=False,
                                                                 local_inner_layers=decoder_inner_layers,
                                                                 dropout_rate=dropout_rate)
                                decoder_inner_layers.append(outputs)
                            else:
                                outputs = self.add_decoder_layer(input=outputs,
                                                                 name="decoder_inner_conv_{}_{}"
                                                                 .format(i, j), training=training,
                                                                 layer_to_skip_connect=current_layers,
                                                                 num_features=num_features,
                                                                 dim_upscale=False,
                                                                 local_inner_layers=decoder_inner_layers,
                                                                 w_size=upscale_shape[1],
                                                                 h_size=upscale_shape[2],
                                                                 dropout_rate=dropout_rate)
                                decoder_inner_layers.append(outputs)
                        current_layers.append(outputs)
                        decoder_layers.append(outputs)

                        if idx>=0:
                            upscale_shape = encoder_layers[idx - 1].get_shape().as_list()
                            if idx == 0:
                                upscale_shape = conditional_input.get_shape().as_list()
                            outputs = self.add_decoder_layer(
                                input=outputs,
                                name="decoder_outer_conv_{}".format(i),
                                training=training,
                                layer_to_skip_connect=current_layers,
                                num_features=num_features,
                                dim_upscale=True, local_inner_layers=decoder_inner_layers, w_size=upscale_shape[1],
                                h_size=upscale_shape[2], dropout_rate=dropout_rate)
                            current_layers.append(outputs)
                        if (idx-1)>=0:
                            outputs = tf.concat([outputs, encoder_layers[idx-1]], axis=3)
                            current_layers[-1] = outputs

                high_res_layers = []

                for p in range(2):
                    outputs = self.conv_layer(outputs, self.layer_sizes[0], [3, 3], strides=(1, 1),
                                                         transpose=False)
                    outputs = leaky_relu(features=outputs)

                    outputs = batch_norm(outputs,
                                         decay=0.99, scale=True,
                                         center=True, is_training=training,
                                         renorm=True)
                    high_res_layers.append(outputs)
                outputs = self.conv_layer(outputs, self.num_channels, [3, 3], strides=(1, 1),
                                                     transpose=False)
            # output images
            with tf.variable_scope('g_tanh'):
                gan_decoder = tf.tanh(outputs, name='outputs')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        if self.build:
            print("generator_total_layers", self.conv_layer_num)
            count_parameters(self.variables, name="generator_parameter_num")
        self.build = False
        return gan_decoder, encoder_layers, decoder_layers


class Discriminator:
    def __init__(self, batch_size, layer_sizes, inner_layers, use_wide_connections=False, name="d"):
        """
        Initialize a discriminator network.
        :param batch_size: Batch size for discriminator.
        :param layer_sizes: A list with the feature maps for each MultiLayer.
        :param inner_layers: An integer indicating the number of inner layers.
        """
        self.reuse = False
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.inner_layers = inner_layers
        self.conv_layer_num = 0
        self.use_wide_connections = use_wide_connections
        self.build = True
        self.name = name

    def upscale(self, x, scale):
        """
            Upscales an image using nearest neighbour
            :param x: Input image
            :param h_size: Image height size
            :param w_size: Image width size
            :return: Upscaled image
        """
        [b, h, w, c] = [int(dim) for dim in x.get_shape()]

        return tf.image.resize_nearest_neighbor(x, (h * scale, w * scale))

    def conv_layer(self, inputs, num_filters, filter_size, strides, activation=None, transpose=False):
        """
        Add a convolutional layer to the network.
        :param inputs: Inputs to the conv layer.
        :param num_filters: Num of filters for conv layer.
        :param filter_size: Size of filter.
        :param strides: Stride size.
        :param activation: Conv layer activation.
        :param transpose: Whether to apply upscale before convolution.
        :return: Convolution features
        """
        self.conv_layer_num += 1
        if transpose:
            outputs = tf.layers.conv2d_transpose(inputs, num_filters, filter_size, strides=strides,
                                       padding="SAME", activation=activation)
        elif not transpose:
            outputs = tf.layers.conv2d(inputs, num_filters, filter_size, strides=strides,
                                                 padding="SAME", activation=activation)
        return outputs

    def add_encoder_layer(self, input, name, training, layer_to_skip_connect, local_inner_layers, num_features,
                          dim_reduce=False, dropout_rate=0.0):

        """
        Adds a resnet encoder layer.
        :param input: The input to the encoder layer
        :param training: Flag for training or validation
        :param dropout_rate: A float or a placeholder for the dropout rate
        :param layer_to_skip_connect: Layer to skip-connect this layer to
        :param local_inner_layers: A list with the inner layers of the current Multi-Layer
        :param num_features: Number of feature maps for the convolutions
        :param dim_reduce: Boolean value indicating if this is a dimensionality reducing layer or not
        :return: The output of the encoder layer
        :return:
        """
        [b1, h1, w1, d1] = input.get_shape().as_list()
        if layer_to_skip_connect is not None:
            [b0, h0, w0, d0] = layer_to_skip_connect.get_shape().as_list()

            if h0 > h1:
                skip_connect_layer = self.conv_layer(layer_to_skip_connect, int(layer_to_skip_connect.get_shape()[3]),
                                                     [3, 3], strides=(2, 2))
            else:
                skip_connect_layer = layer_to_skip_connect
        else:
            skip_connect_layer = layer_to_skip_connect
        current_layers = [input, skip_connect_layer]
        current_layers.extend(local_inner_layers)
        current_layers = remove_duplicates(current_layers)
        outputs = tf.concat(current_layers, axis=3)
        if dim_reduce:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(2, 2))
            outputs = leaky_relu(features=outputs)
            outputs = layer_norm(inputs=outputs, center=True, scale=True)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
        else:
            outputs = self.conv_layer(outputs, num_features, [3, 3], strides=(1, 1))
            outputs = leaky_relu(features=outputs)
            outputs = layer_norm(inputs=outputs, center=True, scale=True)

        return outputs


    def __call__(self, conditional_input, generated_input, training=False, dropout_rate=0.0):
        """
        :param conditional_input: A batch of conditional inputs (x_i) of size [batch_size, height, width, channel]
        :param generated_input: A batch of generated inputs (x_g) of size [batch_size, height, width, channel]
        :param training: Placeholder for training or a boolean indicating training or validation
        :param dropout_rate: A float placeholder for dropout rate or a float indicating the dropout rate
        :param name: Network name
        :return:
        """
        conditional_input = tf.convert_to_tensor(conditional_input)
        generated_input = tf.convert_to_tensor(generated_input)
        with tf.variable_scope(self.name, reuse=self.reuse):
            concat_images = tf.concat([conditional_input, generated_input], axis=3)
            outputs = concat_images
            encoder_layers = []
            current_layers = [outputs]
            with tf.variable_scope('conv_layers'):
                for i, layer_size in enumerate(self.layer_sizes):
                    encoder_inner_layers = [outputs]
                    with tf.variable_scope('g_conv{}'.format(i)):
                        if i == 0:
                            outputs = self.conv_layer(outputs, num_filters=64,
                                                      filter_size=(3, 3), strides=(2, 2))
                            outputs = leaky_relu(features=outputs)
                            outputs = layer_norm(inputs=outputs, center=True, scale=True)
                            current_layers.append(outputs)
                        else:
                            for j in range(self.inner_layers[i]):
                                outputs = self.add_encoder_layer(input=outputs,
                                                                 name="encoder_inner_conv_{}_{}"
                                                                 .format(i, j), training=training,
                                                                 layer_to_skip_connect=current_layers[-2],
                                                                 num_features=self.layer_sizes[i],
                                                                 dropout_rate=dropout_rate,
                                                                 dim_reduce=False,
                                                                 local_inner_layers=encoder_inner_layers)
                                current_layers.append(outputs)
                                encoder_inner_layers.append(outputs)
                            outputs = self.add_encoder_layer(input=outputs,
                                                             name="encoder_outer_conv_{}"
                                                             .format(i),
                                                             training=training,
                                                             layer_to_skip_connect=
                                                                     current_layers[-2],
                                                             local_inner_layers=
                                                                     encoder_inner_layers,
                                                             num_features=self.layer_sizes[i],
                                                             dropout_rate=dropout_rate,
                                                             dim_reduce=True)
                            current_layers.append(outputs)
                        encoder_layers.append(outputs)


            with tf.variable_scope('discriminator_dense_block'):
                if self.use_wide_connections:
                    mean_encoder_layers = []
                    concat_encoder_layers = []
                    for layer in encoder_layers:
                        mean_encoder_layers.append(tf.reduce_mean(layer, axis=[1, 2]))
                        concat_encoder_layers.append(tf.layers.flatten(layer))
                    feature_level_flatten = tf.concat(mean_encoder_layers, axis=1)
                    location_level_flatten = tf.concat(concat_encoder_layers, axis=1)
                else:
                    feature_level_flatten = tf.reduce_mean(encoder_layers[-1], axis=[1, 2])
                    location_level_flatten = tf.layers.flatten(encoder_layers[-1])

                feature_level_dense = tf.layers.dense(feature_level_flatten, units=1024, activation=leaky_relu)
                combo_level_flatten = tf.concat([feature_level_dense, location_level_flatten], axis=1)
            with tf.variable_scope('discriminator_out_block'):
                outputs = tf.layers.dense(combo_level_flatten, 1, name='outputs')

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        #view_names_of_variables(self.variables)
        if self.build:
            print("discr layers", self.conv_layer_num)
            count_parameters(self.variables, name="discriminator_parameter_num")
        self.build = False
        return outputs, current_layers


class GAN:
    def __init__(self, input_x_i, input_x_j, dropout_rate, generator_layer_sizes,
                 discriminator_layer_sizes, generator_layer_padding, z_inputs, batch_size=100, z_dim=100,
                 num_channels=1, is_training=True, augment=True, discr_inner_conv=0, gen_inner_conv=0, num_gpus=1, 
                 use_wide_connections=False):

        """
        Initializes a GAN object.
        :param input_x_i: Input image x_i
        :param input_x_j: Input image x_j
        :param dropout_rate: A dropout rate placeholder or a scalar to use throughout the network
        :param generator_layer_sizes: A list with the number of feature maps per layer (generator) e.g. [64, 64, 64, 64]
        :param discriminator_layer_sizes: A list with the number of feature maps per layer (discriminator)
                                                                                                   e.g. [64, 64, 64, 64]
        :param generator_layer_padding: A list with the type of padding per layer (e.g. ["SAME", "SAME", "SAME","SAME"]
        :param z_inputs: A placeholder for the random noise injection vector z (usually gaussian or uniform distribut.)
        :param batch_size: An integer indicating the batch size for the experiment.
        :param z_dim: An integer indicating the dimensionality of the random noise vector (usually 100-dim).
        :param num_channels: Number of image channels
        :param is_training: A boolean placeholder for the training/not training flag
        :param augment: A boolean placeholder that determines whether to augment the data using rotations
        :param discr_inner_conv: Number of inner layers per multi layer in the discriminator
        :param gen_inner_conv: Number of inner layers per multi layer in the generator
        :param num_gpus: Number of GPUs to use for training
        """
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.z_inputs = z_inputs
        self.num_gpus = num_gpus

        self.g = Generator(batch_size=self.batch_size, layer_sizes=generator_layer_sizes,
                                  num_channels=num_channels, layer_padding=generator_layer_padding,
                                  inner_layers=gen_inner_conv, name="generator")

        self.d = Discriminator(batch_size=self.batch_size, layer_sizes=discriminator_layer_sizes,
                               inner_layers=discr_inner_conv, use_wide_connections=use_wide_connections, name="discriminator")

        self.input_x_i = input_x_i
        self.input_x_j = input_x_j
        self.dropout_rate = dropout_rate
        self.training_phase = is_training
        self.augment = augment

    def rotate_data(self, image_a, image_b):
        """
        Rotate 2 images by the same number of degrees
        :param image_a: An image a to rotate k degrees
        :param image_b: An image b to rotate k degrees
        :return: Two images rotated by the same amount of degrees
        """
        random_variable = tf.unstack(tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32, seed=None, name=None))
        image_a = tf.image.rot90(image_a, k=random_variable[0])
        image_b = tf.image.rot90(image_b, k=random_variable[0])
        return [image_a, image_b]

    def rotate_batch(self, batch_images_a, batch_images_b):
        """
        Rotate two batches such that every element from set a with the same index as an element from set b are rotated
        by an equal amount of degrees
        :param batch_images_a: A batch of images to be rotated
        :param batch_images_b: A batch of images to be rotated
        :return: A batch of images that are rotated by an element-wise equal amount of k degrees
        """
        shapes = map(int, list(batch_images_a.get_shape()))
        batch_size, x, y, c = shapes
        with tf.name_scope('augment'):
            batch_images_unpacked_a = tf.unstack(batch_images_a)
            batch_images_unpacked_b = tf.unstack(batch_images_b)
            new_images_a = []
            new_images_b = []
            for image_a, image_b in zip(batch_images_unpacked_a, batch_images_unpacked_b):
                rotate_a, rotate_b = self.augment_rotate(image_a, image_b)
                new_images_a.append(rotate_a)
                new_images_b.append(rotate_b)

            new_images_a = tf.stack(new_images_a)
            new_images_a = tf.reshape(new_images_a, (batch_size, x, y, c))
            new_images_b = tf.stack(new_images_b)
            new_images_b = tf.reshape(new_images_b, (batch_size, x, y, c))
            return [new_images_a, new_images_b]

    def generate(self, conditional_images, z_input=None):
        """
        Generate samples with the GAN
        :param conditional_images: Images to condition GAN on.
        :param z_input: Random noise to condition the GAN on. If none is used then the method will generate random
        noise with dimensionality [batch_size, z_dim]
        :return: A batch of generated images, one per conditional image
        """
        if z_input is None:
            z_input = tf.random_normal([self.batch_size, self.z_dim], mean=0, stddev=1)

        generated_samples, encoder_layers, decoder_layers = self.g(z_input,
                               conditional_images,
                               training=self.training_phase,
                               dropout_rate=self.dropout_rate)
        return generated_samples

    def augment_rotate(self, image_a, image_b):
        r = tf.unstack(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, seed=None, name=None))
        rotate_boolean = tf.equal(0, r, name="check-rotate-boolean")
        [image_a, image_b] = tf.cond(rotate_boolean[0], lambda: self.rotate_data(image_a, image_b),
                        lambda: [image_a, image_b])
        return image_a, image_b

    def data_augment_batch(self, batch_images_a, batch_images_b):
        """
        Apply data augmentation to a set of image batches if self.augment is set to true
        :param batch_images_a: A batch of images to augment
        :param batch_images_b: A batch of images to augment
        :return: A list of two augmented image batches
        """
        [images_a, images_b] = tf.cond(self.augment, lambda: self.rotate_batch(batch_images_a, batch_images_b),
                                       lambda: [batch_images_a, batch_images_b])
        return images_a, images_b

    def save_features(self, name, features):
        """
        Save feature activations from a network
        :param name: A name for the summary of the features
        :param features: The features to save
        """
        for i in range(len(features)):
            shape_in = features[i].get_shape().as_list()
            channels = shape_in[3]
            y_channels = 8
            x_channels = channels / y_channels

            activations_features = tf.reshape(features[i], shape=(shape_in[0], shape_in[1], shape_in[2],
                                                                        y_channels, x_channels))

            activations_features = tf.unstack(activations_features, axis=4)
            activations_features = tf.concat(activations_features, axis=2)
            activations_features = tf.unstack(activations_features, axis=3)
            activations_features = tf.concat(activations_features, axis=1)
            activations_features = tf.expand_dims(activations_features, axis=3)
            tf.summary.image('{}_{}'.format(name, i), activations_features)

    def loss(self, gpu_id):

        """
        Builds models, calculates losses, saves tensorboard information.
        :param gpu_id: The GPU ID to calculate losses for.
        :return: Returns the generator and discriminator losses.
        """
        with tf.name_scope("losses_{}".format(gpu_id)):

            input_a, input_b = self.data_augment_batch(self.input_x_i[gpu_id], self.input_x_j[gpu_id])
            x_g = self.generate(input_a)

            g_same_class_outputs, g_discr_features = self.d(x_g, input_a, training=self.training_phase,
                                          dropout_rate=self.dropout_rate)

            t_same_class_outputs, t_discr_features = self.d(input_b, input_a, training=self.training_phase,
                                          dropout_rate=self.dropout_rate)

            # Remove comments to save discriminator feature activations
            # self.save_features(name="generated_discr_layers", features=g_discr_features)
            # self.save_features(name="real_discr_layers", features=t_discr_features)

            d_real = t_same_class_outputs
            d_fake = g_same_class_outputs
            d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            g_loss = -tf.reduce_mean(d_fake)

            alpha = tf.random_uniform(
                shape=[self.batch_size, 1],
                minval=0.,
                maxval=1.
            )
            input_shape = input_a.get_shape()
            input_shape = [int(n) for n in input_shape]
            differences_g = x_g - input_b
            differences_g = tf.reshape(differences_g, (self.batch_size, input_shape[1]*input_shape[2]*input_shape[3]))
            interpolates_g = input_b + tf.reshape(alpha * differences_g, (self.batch_size, input_shape[1],
                                                                          input_shape[2], input_shape[3]))
            pre_grads, grad_features = self.d(interpolates_g, input_a, dropout_rate=self.dropout_rate,
                                     training=self.training_phase)
            gradients = tf.gradients(pre_grads, [interpolates_g, input_a])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            d_loss += 10 * gradient_penalty

            tf.add_to_collection('g_losses', g_loss)
            tf.add_to_collection('d_losses', d_loss)
            tf.summary.scalar('g_losses', g_loss)
            tf.summary.scalar('d_losses', d_loss)

            tf.summary.scalar('d_loss_real', tf.reduce_mean(d_real))
            tf.summary.scalar('d_loss_fake', tf.reduce_mean(d_fake))
            tf.summary.image('output_generated_images', [tf.concat(tf.unstack(x_g, axis=0), axis=0)])
            tf.summary.image('output_input_a', [tf.concat(tf.unstack(input_a, axis=0), axis=0)])
            tf.summary.image('output_input_b', [tf.concat(tf.unstack(input_b, axis=0), axis=0)])

        return {
            "g_losses": tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            "d_losses": tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')
        }

    def train(self, opts, losses):

        """
        Returns ops for training our GAN system.
        :param opts: A dict with optimizers.
        :param losses: A dict with losses.
        :return: A dict with training ops for the dicriminator and the generator.
        """
        opt_ops = dict()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt_ops["g_opt_op"] = opts["g_opt"].minimize(losses["g_losses"],
                                          var_list=self.g.variables,
                                          colocate_gradients_with_ops=True)
            opt_ops["d_opt_op"] = opts["d_opt"].minimize(losses["d_losses"],
                                                         var_list=self.d.variables,
                                                         colocate_gradients_with_ops=True)
        return opt_ops

    def init_train(self, learning_rate=1e-4, beta1=0.0, beta2=0.9):
        """
        Initialize training by constructing the summary, loss and ops
        :param learning_rate: The learning rate for the Adam optimizer
        :param beta1: Beta1 for the Adam optimizer
        :param beta2: Beta2 for the Adam optimizer
        :return: summary op, losses and training ops.
        """

        losses = dict()
        opts = dict()

        if self.num_gpus > 0:
            device_ids = ['/gpu:{}'.format(i) for i in range(self.num_gpus)]
        else:
            device_ids = ['/cpu:0']
        for gpu_id, device_id in enumerate(device_ids):
            with tf.device(device_id):
                total_losses = self.loss(gpu_id=gpu_id)
                for key, value in total_losses.items():
                    if key not in losses.keys():
                        losses[key] = [value]
                    else:
                        losses[key].append(value)

        for key in list(losses.keys()):
            losses[key] = tf.reduce_mean(losses[key], axis=0)
            opts[key.replace("losses", "opt")] = tf.train.AdamOptimizer(beta1=beta1, beta2=beta2,
                                                                            learning_rate=learning_rate)

        summary = tf.summary.merge_all()
        apply_grads_ops = self.train(opts=opts, losses=losses)

        return summary, losses, apply_grads_ops

    def sample_same_images(self):
        """
        Samples images from the GAN using input_x_i as image conditional input and z_inputs as the gaussian noise.
        :return: Inputs and generated images
        """
        conditional_inputs = self.input_x_i[0]
        generated = self.generate(conditional_inputs,
           z_input=self.z_inputs)

        return self.input_x_i[0], generated



def remove_duplicates(input_features):
    """
    Remove duplicate entries from layer list.
    :param input_features: A list of layers
    :return: Returns a list of unique feature tensors (i.e. no duplication).
    """
    feature_name_set = set()
    non_duplicate_feature_set = []
    for feature in input_features:
        if feature.name not in feature_name_set:
            non_duplicate_feature_set.append(feature)
        feature_name_set.add(feature.name)
    return non_duplicate_feature_set