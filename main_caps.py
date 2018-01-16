import tensorflow as tf
from config import cfg
from tqdm import tqdm
import time
from datetime import timedelta

from capsNet import CapsNet
from convModel import ConvModel
from utils import get_data, serialize_data, next_batch

from tensorflow.python import debug as tf_debug

tf.logging.set_verbosity("INFO")    

root_dir = "/home/anne/src/dog_identification/"  
train_zip = root_dir + "data/train.zip"
valid_zip = root_dir + "data/valid.zip"
# training_filename_p = root_dir + "data/train_conv.p"
training_filename_p = root_dir + "data/train.p"
validation_filename = root_dir + "data/valid.p"
labels_filename = root_dir + "data/labels.csv.zip"

nbreeds = cfg.nb_labels
batch_size = cfg.batch_size 


def train(model):
    x_train, y_train, breed_codes = get_data(training_filename_p, labels_filename, nbreeds)
    tf.logging.info("Training for %d breeds, using %d images" % (nbreeds, x_train.shape[0]))
    # x_valid, y_valid = get_data(validation_filename, labels_filename, nbreeds)

    n_batches = cfg.n_batches # Amount of batches per epoch

    config = tf.ConfigProto(
        # device_count = {'GPU': 0}
    )
    config.gpu_options.allow_growth = True

    with model.graph.as_default():

        tf.logging.info("Starting training")
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(cfg.logdir, sess.graph)

        # For debugging
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        for epoch in range(cfg.epochs):
            start_time = time.time()
            for step in tqdm(range(n_batches), total=n_batches, ncols=70, leave=False, unit='b'):

                x_batch, y_batch = next_batch(x_train, y_train, batch_size)

                # train the model
                feed_dict = {
                    model.tf_images: x_batch,
                    model.tf_labels: y_batch
                }
                # tensors = [model.train_op,
                #             model.tf_margin_loss,
                #             model.accuracy]
                # _, loss, acc = sess.run(tensors, feed_dict=feed_dict)

                feed_dict = {model.tf_images: x_batch,
                             model.tf_labels: y_batch,
                             model.keep_prob_conv: 0.3,
                             model.keep_prob_fc: 0.4}
                _, summary = sess.run([model.train_op, model.summary], feed_dict=feed_dict)

                summary = sess.run(model.summary, feed_dict=feed_dict)
                writer.add_summary(summary, step)

            print("Epoch %d took %s" %(epoch, str(timedelta(seconds=int(round(time.time()-start_time))))))
            # print "Loss: %.4f, Acc: %.4f" % (loss, acc)
            acc = sess.run(model.accuracy, feed_dict)
            print "Accuracy: %.4f" % acc

def main(_):
    serialize_data(train_zip, labels_filename, training_filename_p)
    # model = CapsNet()
    model = ConvModel()
    train(model)


if __name__ == "__main__":
    tf.app.run()
