import numpy as np
from config import cfg
import os
import pickle
import PIL.Image
from zipfile import ZipFile
from io import BytesIO
from scipy.stats import itemfreq
import pandas as pd


def serialize_data(archivename, outname):
    # Make sure file has not already been serialezed
    if os.path.isfile(outname):
        print "pickle file already exists, exiting"
        return
    nwidth = cfg.nwidth
    nheight = cfg.nheight
    archivezip = ZipFile(archivename, 'r')

    #nwigth x nheight = number of features because images are nwigth x nheight pixels in RGB
    s = (len(archivezip.namelist()[:])-1, nwidth, nheight,3) 
    allImages = np.zeros(s)
    for i in range(1,len(archivezip.namelist()[:])):
        imagename = BytesIO(archivezip.read(archivezip.namelist()[i]))
        image = PIL.Image.open(imagename)
        allImages[i-1] = preprocess_data(image, nwidth, nheight)
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

    labels_raw_np = labels_raw["breed"].as_matrix() #transform in numpy
    labels_raw_np = labels_raw_np.reshape(labels_raw_np.shape[0],1)

    # Get indexes of the images of the main breeds
    labels_filtered_index = np.where(labels_raw_np == main_labels)
    # Get the labels, and get the images
    labels_filtered = labels_raw.iloc[labels_filtered_index[0],:]
    
    images = pickle.load(open(image_file, "rb")) # load the pickle file
    images_filtred = images[labels_filtered_index[0],:,:,:]   

    return images_filtred, labels_filtered

def next_batch(images, labels, batch_size, start_index=None):
    """
    Returns a subset of data which can be randomized or not,
    If not randomized, give starting index
    """
    n_images = len(images)
    if start_index is None:
        indexes = np.arange(0, n_images)
    else: 
        indexes = np.arange(start_index, start_index+batch_size) % n_images

    return images[indexes, :, :, :], labels[indexes]


