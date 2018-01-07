import tensorflow as tf 

flags = tf.app.flags    
    
flags.DEFINE_integer("conv_1_size", 9,'description')
flags.DEFINE_integer("conv_1_nb", 256,'description')
flags.DEFINE_integer("conv_2_size", 6,'description')
flags.DEFINE_integer("conv_2_nb", 64,'description')
# flags.DEFINE_integer("conv_2_dropout", 0.7,'description')
flags.DEFINE_integer("caps_1_vec_len", 16,'description')
flags.DEFINE_integer("caps_1_size", 5,'description')
flags.DEFINE_integer("caps_1_nb_filter", 16,'description')
# flags.DEFINE_integer("caps_2_vec_len", 32,'description')
flags.DEFINE_integer("caps_2_vec_len", 60,'description')
flags.DEFINE_float("learning_rate", 0.0001,'description')
flags.DEFINE_integer("routing_steps", 1, 'description')
flags.DEFINE_integer("nb_labels", 4, "Amount of labels to train")
flags.DEFINE_integer("nwidth", 60, "Transform image width")
flags.DEFINE_integer("nheight", 60, "Transform image height")
flags.DEFINE_integer("nbreeds", 5, "Amount of breeds to filter")
flags.DEFINE_integer("batch_size", 50, "Size of each batch")
flags.DEFINE_integer("epochs", 10, "Amount of iterations of training data")

cfg = tf.app.flags.FLAGS