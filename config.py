import tensorflow as tf 

flags = tf.app.flags    

flags.DEFINE_string("logdir", "tensorflow_log", "Log storage for TF")
    
flags.DEFINE_integer("conv_1_size", 9,'description')
flags.DEFINE_integer("conv_1_nb", 256,'description')
flags.DEFINE_integer("conv_2_size", 6,'description')
flags.DEFINE_integer("conv_2_nb", 64,'description')
# flags.DEFINE_integer("conv_2_dropout", 0.7,'description')
flags.DEFINE_integer("caps_1_vec_len", 16,'description')
flags.DEFINE_integer("caps_1_size", 5,'description')
flags.DEFINE_integer("caps_1_nb_filter", 16,'description')
flags.DEFINE_integer("caps_2_vec_len", 32,'description')
# flags.DEFINE_integer("caps_2_vec_len", 60,'description')
flags.DEFINE_float("learning_rate", 0.0001,'description')
flags.DEFINE_integer("routing_steps", 1, 'description')
flags.DEFINE_integer("nb_labels", 10, "Amount of labels to train")
flags.DEFINE_integer("nwidth", 32, "Transform image width")
flags.DEFINE_integer("nheight", 32, "Transform image height")
flags.DEFINE_integer("batch_size", 100, "Size of each batch")
flags.DEFINE_integer("epochs", 50, "Amount of iterations of training data")

# Convolutional model
flags.DEFINE_integer("filter_size1", 5, "Convolution filters are 5 x 5 pixels.")
flags.DEFINE_integer("num_filters1", 32, "There are 32 of these filters.")

# Convolutional Layer 2.
flags.DEFINE_integer("filter_size2", 4, "Convolution filters are 4 x 4 pixels.")
flags.DEFINE_integer("num_filters2", 64, "There are 64 of these filters.")

# Convolutional Layer 3.
flags.DEFINE_integer("filter_size3", 3, "Convolution filters are 3 x 3 pixels.")
flags.DEFINE_integer("num_filters3", 128, "There are 128 of these filters.")

# Fully-connected layer.
flags.DEFINE_integer("fc_size", 500, "Size of fully connected layer")
cfg = tf.app.flags.FLAGS