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
import matplotlib.pyplot as plt
import seaborn as sns
import tables
import math
from mpl_toolkits.axes_grid1 import ImageGrid

from keras.applications import xception

def serialize_data(trainfile, labelfile, outname, img_size=None):
    """
    Create a serialized file format of a pandas dataframe for easy data retrieval.
    To save space, image preprocessing is also done here.
    """

    # Make sure file has not already been serialezed
    print("Creating serialized file format for " + trainfile)
    if os.path.isfile(outname):
        print("pickle file already exists, continueing")
        return
    labels = pd.read_csv(labelfile,
                        compression='zip', 
                        header=0, 
                        sep=',', 
                        quotechar='"')
    if img_size is None:
        nwidth = cfg.nwidth
        nheight = cfg.nheight
    else:
        nwidth = img_size
        nheight = img_size
    archivezip = ZipFile(trainfile, 'r')

    #nwigth x nheight = number of features because images are nwigth x nheight pixels in RGB
    s = (len(archivezip.namelist()[:])-1, nwidth, nheight,3) 
    allImages = np.zeros(s)

    # Iterate over labels to make sure numpy array has same indexes as labels
    for index, row in tqdm(labels.iterrows(), total=len(labels.index), ncols=70): 
        imagename = 'train/' + row['id'] + ".jpg"
        imagename = BytesIO(archivezip.read(imagename))
        image = PIL.Image.open(imagename)
        allImages[index] = preprocess_data(image,nwidth, nheight)
    pickle.dump(allImages, open(outname, "wb"))

def store_data_hdf5(train_file, label_file, store_file, store_size=300, train_val=1):
    """ Store the entire dataset in hdf5 dataformat to allow for disk streaming. 
    # Arguments
        train_file: the zip containing the training data.
        label_file: the zip containing the labels .csv file
        store_file: the desired output file name.
        store_size: the desired image size for storage
        train_val: the desired training/validation size 
    """
    if os.path.isfile(store_file):
        print("HDF5 file already exists, continueing")
        return

    labels = pd.read_csv(label_file,
                    compression='zip', 
                    header=0, 
                    sep=',', 
                    quotechar='"')
    labels = labels.as_matrix() # convert to numpy array
    nfiles = labels.shape[0]
    max_train_idx = int(nfiles*train_val)

    img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved
    # check the order of data and chose proper data shape to save images
    data_shape = (0, store_size, store_size, 3)
    # open a hdf5 file and create earrays
    hdf5_file = tables.open_file(store_file, mode='w')
    train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
    val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)

    train_idx = np.arange(0,max_train_idx)
    val_idx = np.arange(max_train_idx,nfiles)

    # create the label arrays and copy the labels data in them
    hdf5_file.create_array(hdf5_file.root, 'train_labels', labels[train_idx,1].tolist())
    hdf5_file.create_array(hdf5_file.root, 'val_labels', labels[val_idx,1].tolist())

    archivezip = ZipFile(train_file, 'r')
    for index in tqdm(range(nfiles), total=nfiles, ncols=70): 
        imagename = 'train/' + labels[index,0] + ".jpg"
        image_buf = BytesIO(archivezip.read(imagename))
        image = PIL.Image.open(image_buf)
        image = image.resize((store_size, store_size))
        image = np.array(image)
        if index < max_train_idx:
            train_storage.append(image[None])
        else:
            val_storage.append(image[None])
    hdf5_file.close()

def hdf5_image_generator(filename, batch_size=32, n_breeds=None, image_preprocessor=None, training=True):
    """ Generates batches of images from hdf5 file """
    with tables.open_file(filename, 'r') as f:
        train_labels = f.root.train_labels
        val_labels = f.root.val_labels
        labels = train_labels[:] + val_labels[:]
        labels = np.array(labels)

        top_breeds = get_top_breeds(labels,n_breeds)
        train_idx = get_breed_indexes(train_labels, top_breeds)
        val_idx = get_breed_indexes(val_labels, top_breeds)

        if training:
            indexes = train_idx
            images = f.root.train_img
        else:
            indexes = val_idx
            images = f.root.val_img
        indexes = np.array(indexes)
        n_indexes = indexes.shape[0]
        residual = n_indexes % batch_size # 

        start = 0
        end = 0
        while end < n_indexes:
            end = min(end+batch_size, n_indexes)
            index_batch = indexes[start:end] 
            if image_preprocessor:
                s = (len(index_batch),) + images.shape[1:] 
                x = np.zeros(s)
                for j, index in enumerate(index_batch):
                    x[j] = image_preprocessor(images[index])
            else:
                x = images[index_batch, :, :, :]
            yield x
            start = end

        f.close()

def get_breed_indexes(labels, breeds):
    return np.flatnonzero(np.in1d(labels, breeds)).tolist()

def get_top_breeds(labels, n_breeds=None):
    labels_freq = itemfreq(labels)
    labels_freq = labels_freq[labels_freq[:, 1].astype(int).argsort()[::-1]] #[::-1] ==> to sort in descending order 
    if n_breeds is None: # take all breeds
        main_labels = labels_freq[:,0][:]
    else: # only take top frequent breeds
        main_labels = labels_freq[:,0][0:n_breeds]
    return main_labels


def preprocess_data(image, nwidth, nheight):
    """ This function creates normalized, square images,
        this is so that all images are the same shape, and
        normalize so that pixel values will be in the range [0,1] """
    image = image.resize((nwidth, nheight))
    image = np.array(image)
    image = np.clip(image/255.0, 0.0, 1.0)

    return image

def get_steps(hdf5_file, batch_size, n_breeds):
    """ Returns the amount of steps that are taken by the image generators, 
        to loop trough all the possible images.
    """
    with tables.open_file(hdf5_file, mode="r") as f:
        train_labels = f.root.train_labels
        val_labels = f.root.val_labels
        labels = train_labels[:] + val_labels[:]
        labels = np.array(labels)

        top_breeds = get_top_breeds(labels,n_breeds)
        train_idx = get_breed_indexes(train_labels, top_breeds)
        val_idx = get_breed_indexes(val_labels, top_breeds)

        train_steps = int(math.ceil(float(len(train_idx))/batch_size))
        val_steps = int(math.ceil(float(len(val_idx))/batch_size))
        f.close()
    return train_steps, val_steps

def get_labels(hdf5_file, n_breeds):
    """ Returns all the labels for the desired breeds. """
    with tables.open_file(hdf5_file, mode="r") as f:
        train_labels = f.root.train_labels
        val_labels = f.root.val_labels
        labels = np.array(train_labels[:] + val_labels[:])

        top_breeds = get_top_breeds(labels,n_breeds)
        train_idx = get_breed_indexes(train_labels, top_breeds)
        val_idx = get_breed_indexes(val_labels, top_breeds)

        _, y_train = np.unique(train_labels[train_idx], return_inverse=True)
        breed_dict, y_val = np.unique(val_labels[val_idx], return_inverse=True)
        f.close()
    return y_train, y_val, breed_dict

def get_images(hdf5_file, indexes, n_breeds, train):
    """ Returns the images belonging to indexes of a subset,
        this can be used when evaluating the dataset.
        # Inputs
        hdf5_file: The file containing the data
        indexes: The indexes of the subset, so not the whole set
        n_breeds: The amount of breeds that have been trained on
        trian: indexes of training set or validation set (boolean)
        # Returns
        images: a list of images
         """
    with tables.open_file(hdf5_file, mode="r") as f:
        train_labels = f.root.train_labels
        val_labels = f.root.val_labels
        labels = np.array(train_labels[:] + val_labels[:])

        top_breeds = get_top_breeds(labels,n_breeds)
        train_idx = get_breed_indexes(train_labels, top_breeds)
        val_idx = get_breed_indexes(val_labels, top_breeds)

        if train:
            ids = np.array(train_idx)[indexes]
            images = f.root.train_img[ids, :, :, :]
        else:
            ids = np.array(val_idx)[indexes]
            images = f.root.val_img[ids, :, :, :]
        f.close()
    return images


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

    # get indexes of main breeds
    main_indexes = labels_raw.index[labels_raw['breed'].isin(main_labels)].tolist()

    main_breeds = labels_raw['breed'][main_indexes]
    # get dictionary of numeric value for each breed
    breed_codes = dict(enumerate(main_breeds.astype('category').cat.categories))

    # translate all breeds to their corresponding code
    label_codes_filtered = main_breeds.astype('category').cat.codes.as_matrix()
    # labels_codes = labels_codes.reshape(labels_codes.shape[0], 1)

    # Get the labels, and get the images
    print("Loading pickle file %s" % image_file)
    images = pickle.load(open(image_file, "rb")) # load the pickle file
    images_filtred = images[main_indexes,:,:,:]   

    return images_filtred, label_codes_filtered, breed_codes

def plot_image_overview(images, cls_true, cls_pred=None):
    """ Plots 12 images and their respective labels """
    assert len(images) == len(cls_true) == 12
    
    fig = plt.figure(1, figsize=(16, 16))
    grid = ImageGrid(fig, 111, nrows_ncols=(3,4), axes_pad=0.05)
    for i, ax in enumerate(grid):
        ax.imshow(images[i,:,:,:])
        ax.text(10, 250, 'True: %s' % cls_true[i], color='k', backgroundcolor='w', alpha=0.8)
        if not cls_pred is None:
            b_color = 'w' if cls_true[i] == cls_pred[i] else 'r'
            ax.text(10, 280, 'Predicted: %s' % (cls_pred[i]), color='k', backgroundcolor=b_color, alpha=0.8)
        ax.axis('off')
    plt.show()

def breed_overview(label_file):
    labels = pd.read_csv(label_file,
                            compression='zip', 
                            header=0, 
                            sep=',', 
                            quotechar='"')    
    yy = pd.value_counts(labels['breed'])

    fig, ax = plt.subplots()
    fig.set_size_inches(15, 9)
    sns.set_style("whitegrid")

    ax = sns.barplot(x = yy.index, y = yy, data = labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize = 8)
    ax.set(xlabel='Dog Breed', ylabel='Count')
    ax.set_title('Distribution of Dog breeds')
    plt.show()

if __name__ == "__main__":
    root_dir = "/home/anne/src/dog_identification/"  
    train_zip = root_dir + "data/train.zip"
    valid_zip = root_dir + "data/valid.zip"
    training_filename_p = root_dir + "data/train_conv.p"
    validation_filename = root_dir + "data/valid.p"
    labels_filename = root_dir + "data/labels.csv.zip"
    hdf5_file = root_dir + "data/images.hdf5"

    print('streaming to HDF5')
    # store_data_hdf5(train_zip, labels_filename, hdf5_file, train_val=0.7)
    # x_train, x_val, y_train, y_val = get_data_hdf5(hdf5_file,n_breeds=3)
    # print x_train.shape
    # x = hdf5_image_generator(hdf5_file, batch_size=15, n_breeds=4, image_preprocessor=preprocessing_function)
    # xception_bottleneck = xception.Xception(weights = "imagenet", include_top=False, pooling='avg')
    # train_steps, val_steps = get_steps(hdf5_file, batch_size=15, n_breeds=4)
    # y_train, y_val, breed_dict = get_labels(hdf5_file, n_breeds=4)
    # # print y_train.shape
    # # print y_val.shape
    
    # indexes = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    # images = get_images(hdf5_file, indexes, 4, train=True)
    # print images.shape

    # plot_image_overview(images, indexes.astype(str), indexes.astype(str))



    
    # train_x_bf = xception_bottleneck.predict_generator(x, steps=train_steps, verbose=1)

    # count = 0
    # for step in range(train_steps):
    #     count +=1
    #     a = next(x)
    #     print count
        # print a.shape
    # serialize_data(train_zip, labels_filename, 'data/train_full.p', img_size=300)

    # breed_overview(labels_filename)

    # serialize_data(train_zip, labels_filename, training_filename_p)
    # x_train, y_train, breed_codes = get_data(training_filename_p, labels_filename, 5)
    # x_batch, y_batch = next_batch(x_train, y_train, 50)

    # print x_train.shape
    # print y_train.shape
    # print x_batch.shape
    # print y_batch.shape

    # plot_images(x_batch[12:24], [breed_codes[code] for code in y_batch[12:24]])
