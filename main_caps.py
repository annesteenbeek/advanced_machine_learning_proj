import tensorflow as tf
from config import cfg
from tqdm import tqdm

from capsNet import CapsNet
from utils import get_data, serialize_data, next_batch

tf.logging.set_verbosity("INFO")    

root_dir = "/home/anne/src/dog_identification/"  
train_zip = root_dir + "data/train.zip"
valid_zip = root_dir + "data/valid.zip"
training_filename = root_dir + "data/train.p"
validation_filename = root_dir + "data/valid.p"
labels_filename = root_dir + "data/labels.csv.zip"

nbreeds = cfg.nb_labels
batch_size = cfg.batch_size 


def train(model, supervisor):
    x_train, y_train = get_data(training_filename, labels_filename, nbreeds)
    # x_valid, y_valid = get_data(validation_filename, labels_filename, nbreeds)

    n_batches = len(x_train)//batch_size # amount of batches in entire training set

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with supervisor.managed_session(config=config) as sess:
        for epoch in range(cfg.epochs):
            if supervisor.should_stop():
                print('supervisor stopped!')
                break
            
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
                # _, loss, acc = sess.run(tensors, feed_dict=feed_dict)

                # set new starting index for next batch
                start_index += batch_size

def main(_):
    serialize_data(train_zip, training_filename)
    tf.logging.info("Starting training")
    model = CapsNet()
    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)
    train(model, sv)


if __name__ == "__main__":
    tf.app.run()
