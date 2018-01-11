import tensorflow as tf
from config import cfg
from tqdm import tqdm

from capsNet import CapsNet
from utils import get_data, serialize_data, next_batch

tf.logging.set_verbosity("INFO")    

root_dir = "/home/anne/src/dog_identification/"  
train_zip = root_dir + "data/train.zip"
valid_zip = root_dir + "data/valid.zip"
training_filename_p = root_dir + "data/train.p"
validation_filename = root_dir + "data/valid.p"
labels_filename = root_dir + "data/labels.csv.zip"

nbreeds = cfg.nb_labels
batch_size = cfg.batch_size 


def train(model):
    x_train, y_train, breed_codes = get_data(training_filename_p, labels_filename, nbreeds)
    tf.logging.info("Training for %d breeds, using %d images" % (nbreeds, x_train.shape[0]))
    # x_valid, y_valid = get_data(validation_filename, labels_filename, nbreeds)

    # n_batches = len(x_train)//batch_size # amount of batches in entire training set
    n_batches = 50 # Amount of batches per epoch

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with model.graph.as_default():

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch in range(cfg.epochs):

            start_index = 0
            for step in tqdm(range(n_batches), total=n_batches, ncols=70, leave=False, unit='b'):

                x_batch, y_batch = next_batch(x_train, y_train, batch_size, start_index)

                # train the model
                feed_dict = {
                    model.tf_images: x_batch,
                    model.tf_labels: y_batch
                }
                tensors = [model.train_op,
                            model.tf_margin_loss,
                            model.tf_accuracy]
                _, loss, acc = sess.run(tensors, feed_dict=feed_dict)

                # set new starting index for next batch
                start_index += batch_size
            print "Loss: %.4f, Acc: %.4f" % (loss, acc)

def main(_):
    serialize_data(train_zip, labels_filename, training_filename_p)
    tf.logging.info("Starting training")
    model = CapsNet()
    # model = ConvNet()
    train(model)


if __name__ == "__main__":
    tf.app.run()
