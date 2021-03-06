import tensorflow as tf
import numpy as np
from config import cfg

class ConvModel(object):

    def __init__(self):
        filter_size1 = cfg.filter_size1
        num_filters1 = cfg.num_filters1
        filter_size2 = cfg.filter_size2
        num_filters2 = cfg.num_filters2
        filter_size3 = cfg.filter_size3
        num_filters3 = cfg.num_filters3
        fc_size = cfg.fc_size
        img_size = cfg.nwidth
        num_classes = cfg.nb_labels
        num_channels = 3 # rgb

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.tf_images = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='images')
            x_image = tf.reshape(self.tf_images, [-1, img_size, img_size, num_channels]) #-1 put everything as 1 array
            self.tf_labels= tf.placeholder(tf.int64, shape=[None], name='labels')
            one_hot_labels = tf.one_hot(self.tf_labels, depth=num_classes)

            y_true_cls = tf.argmax(one_hot_labels, axis=1)

            self.keep_prob_fc=tf.placeholder(tf.float32)
            self.keep_prob_conv=tf.placeholder(tf.float32)

            layer_conv1, weights_conv1 = self.new_conv_layer(input=x_image,
                                num_input_channels=num_channels,
                                filter_size=filter_size1,
                                num_filters=num_filters1,
                                use_pooling=True,
                                use_dropout=False)
                    
            layer_conv2, weights_conv2 = self.new_conv_layer(input=layer_conv1,
                            num_input_channels=num_filters1,
                            filter_size=filter_size2,
                            num_filters=num_filters2,
                            use_pooling=True,
                            use_dropout=False)
                
            layer_conv3, weights_conv3 = self.new_conv_layer(input=layer_conv2,
                            num_input_channels=num_filters2,
                            filter_size=filter_size3,
                            num_filters=num_filters3,
                            use_pooling=True,
                            use_dropout=True)
            layer_flat, num_features = self.flatten_layer(layer_conv3)
            layer_fc1 = self.new_fc_layer(input=layer_flat,
                            num_inputs=num_features,
                            num_outputs=fc_size,
                            use_relu=True,
                            use_dropout=True)
            layer_fc2 = self.new_fc_layer(input=layer_fc1,
                            num_inputs=fc_size,
                            num_outputs=num_classes,
                            use_relu=False,
                            use_dropout=False)

            y_pred = tf.nn.softmax(layer_fc2)
            y_pred_cls = tf.argmax(y_pred, axis=1)

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                                    labels=one_hot_labels)
            self.cost = tf.reduce_mean(cross_entropy)

            self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            self.summary = self.build_summary()

    def new_conv_layer(self, input,           
                    num_input_channels,
                    filter_size,       
                    num_filters,       
                    use_pooling=True,
                    use_dropout=True):  # Use 2x2 max-pooling.

        # Shape of the filter-weights for the convolution.
        # This format is determined by the TensorFlow API.
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        weights = self.new_weights(shape=shape)

        # Create new biases, one for each filter.
        biases = self.new_biases(length=num_filters)

        # Create the TensorFlow operation for convolution.
        # Note the strides are set to 1 in all dimensions.
        # The first and last stride must always be 1,
        # because the first is for the image-number and
        # the last is for the input-channel.
        # But e.g. strides=[1, 2, 2, 1] would mean that the filter
        # is moved 2 pixels across the x- and y-axis of the image.
        # The padding is set to 'SAME' which means the input image
        # is padded with zeroes so the size of the output is the same.
        layer = tf.nn.conv2d(input=input,
                            filter=weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        layer += biases

        # Use pooling to down-sample the image resolution?
        if use_pooling:
            # This is 2x2 max-pooling, which means that we
            # consider 2x2 windows and select the largest value
            # in each window. Then we move 2 pixels to the next window.
            layer = tf.nn.max_pool(value=layer,
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding='SAME')

        # Rectified Linear Unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This adds some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)
        
        if use_dropout:
            layer = tf.nn.dropout(layer,self.keep_prob_conv)

        # Note that ReLU is normally executed before the pooling,
        # but since relu(max_pool(x)) == max_pool(relu(x)) we can
        # save 75% of the relu-operations by max-pooling first.

        # We return both the resulting layer and the filter-weights
        # because we will plot the weights later.
        return layer, weights


    def flatten_layer(self, layer):
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]

        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()
        
        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])

        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]

        # Return both the flattened layer and the number of features.
        return layer_flat, num_features


    def new_fc_layer(self, input,          # The previous layer.
                    num_inputs,     # Num. inputs from prev. layer.
                    num_outputs,    # Num. outputs.
                    use_relu=True,
                    use_dropout=True): # Use Rectified Linear Unit (ReLU)?

        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)
        
        if use_dropout:
            layer = tf.nn.dropout(layer,self.keep_prob_fc)
            
        return layer


    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    #outputs random value from a truncated normal distribution

    def new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]))

    def build_summary(self):
        summary_list = []
        summary_list.append(tf.summary.scalar('train/accuracy', self.accuracy))
        summary_list.append(tf.summary.image('example_img', self.tf_images))

        return tf.summary.merge(summary_list)
        