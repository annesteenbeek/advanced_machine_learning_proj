import numpy as np
from config import cfg
import os
import pickle
import PIL.Image
from zipfile import ZipFile
from io import BytesIO
from scipy.stats import itemfreq
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def serialize_data(trainfile, labelfile, outname):
    """
    Create a serialized file format of a pandas dataframe for easy data retrieval.
    To save space, image preprocessing is also done here.
    """

    # Make sure file has not already been serialezed
    tf.logging.info("Creating serialized file format for " + trainfile)
    if os.path.isfile(outname):
        tf.logging.info("pickle file already exists, continueing")
        return
    labels = pd.read_csv(labelfile,
                        compression='zip', 
                        header=0, 
                        sep=',', 
                        quotechar='"')
    nwidth = cfg.nwidth
    nheight = cfg.nheight
    archivezip = ZipFile(trainfile, 'r')

    #nwigth x nheight = number of features because images are nwigth x nheight pixels in RGB
    s = (len(archivezip.namelist()[:])-1, nwidth, nheight,3) 
    allImages = np.zeros(s)

    # Iterate over labels to make sure numpy array has same indexes as labels
    for index, row in tqdm(labels.iterrows(), total=len(labels.index)): 
        imagename = 'train/' + row['id'] + ".jpg"
        imagename = BytesIO(archivezip.read(imagename))
        image = PIL.Image.open(imagename)
        allImages[index] = preprocess_data(image,nwidth, nheight)
    pickle.dump(allImages, open(outname, "wb"))

def preprocess_data(image, nwidth, nheight):
    """ This function creates normalized, square images,
        this is so that all images are the same shape, and
        normalize so that pixel values will be in the range [0,1] """
    image = image.resize((nwidth, nheight))
    image = np.array(image)
    image = np.clip(image/255.0, 0.0, 1.0)

    return image

def get_data(image_file, label_file, top_breeds=None):
    """
    Loads the images and the corresponding labels.

    Inputs:
    image_file: The pickle file that contains all the preprocessed images"
    label_file: The csv file that contains all the labels corresponding to the images"
    top_breeds: Number of breeds to load, this makes it easy to reduce the dataset
    """

    labels_raw = pd.read_csv(label_file,
                            compression='zip', 
                            header=0, 
                            sep=',', 
                            quotechar='"')

    labels_freq_pd = itemfreq(labels_raw["breed"])
    labels_freq_pd = labels_freq_pd[labels_freq_pd[:, 1].argsort()[::-1]] #[::-1] ==> to sort in descending order 

    if top_breeds is None: # take all breeds
        main_labels = labels_freq_pd[:,0][:]
    else: # only take top frequent breeds
        main_labels = labels_freq_pd[:,0][0:top_breeds]

    # get dictionary of numeric value for each breed
    breed_codes = dict(enumerate(labels_raw['breed'].astype('category').cat.categories))

    # translate all breeds to their corresponding code
    labels_codes = labels_raw['breed'].astype('category').cat.codes.as_matrix()
    # labels_codes = labels_codes.reshape(labels_codes.shape[0], 1)

    labels_raw_np = labels_raw["breed"].as_matrix() #transform in numpy
    labels_raw_np = labels_raw_np.reshape(labels_raw_np.shape[0],1)

    # get indexes of main breeds
    main_indexes = labels_raw.index[labels_raw['breed'].isin(main_labels)].tolist()

    # Get the labels, and get the images
    label_codes_filtered = labels_codes[main_indexes]

    images = pickle.load(open(image_file, "rb")) # load the pickle file
    images_filtred = images[main_indexes,:,:,:]   

    return images_filtred, label_codes_filtered, breed_codes

def next_batch(all_images, all_labels, batch_size, start_index=None):
    """
    Returns a subset of data which can be randomized or not,
    If not randomized, give starting index
    """
    n_images = len(all_images)
    if start_index is None:
        idx = np.arange(0, n_images)
        np.random.shuffle(idx)
    else: 
        idx = np.arange(start_index, start_index+batch_size) % n_images
        
    x = [all_images[i] for i in idx]
    y = [all_labels[i] for i in idx]
    
    return np.asarray(x), np.asarray(y)


if __name__ == "__main__":
    root_dir = "/home/anne/src/dog_identification/"  
    train_zip = root_dir + "data/train.zip"
    valid_zip = root_dir + "data/valid.zip"
    training_filename_p = root_dir + "data/train.p"
    validation_filename = root_dir + "data/valid.p"
    labels_filename = root_dir + "data/labels.csv.zip"
    serialize_data(train_zip, labels_filename, training_filename_p)
    x_train, y_train, breed_codes = get_data(training_filename_p, labels_filename, 5)
    x_batch, y_batch = next_batch(x_train, y_train, 50, 0)

    print x_train.shape
    print y_train.shape
    print x_batch.shape
    print y_batch.shape
