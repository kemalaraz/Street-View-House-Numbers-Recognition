# -*- coding: utf-8 -*-

from six.moves import range
from six.moves.urllib.request import urlretrieve
from scipy import ndimage
from PIL import Image
import sys
import numpy as np
from numpy import random
import os
from scipy.io import loadmat
import requests
import h5py
import tarfile
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import scipy
import tensorflow as tf

def download(filename):
  """
  Downloads the files from stanford website with the
  svhn url and given file name.

    **Parameters**\n
    ----------\n
    filename: The name of the dataset that will be downloaded
              can be train.tar.gz, test.tar.gz or extra.tar.gz
  """

  ###START CODE###
  url = "http://ufldl.stanford.edu/housenumbers/"
  print("Downloading",filename, "data from ",url+filename,"...")
  full_url=url+filename
  get = requests.get(full_url, allow_redirects=True)
  open(filename, "wb").write(get.content)
  print("Download successful!")
  ###END CODE###



def extract(filename):
  """
  Extract the files from a given filename

    **Parameters**\n
    ----------\n
    filename: The name of the dataset that will be downloaded
              can be train.tar.gz, test.tar.gz or extra.tar.gz.

    **Returns**\n
    -------\n
    folder_name: The name of the folder that all extracted files in.
  """
  ###START CODE###
  print("Extracting data from",filename,"...")
  folder_name = os.path.splitext(os.path.splitext(filename)[0])[0]
  tar = tarfile.open(filename)
  tar.extractall()
  tar.close()
  print("Extraction successful!")
  return folder_name
  ###END CODE###



def bbox_Dims(feature,digit_structure):
  """
  Extract all the attributes of a feature of the bbox for a given image.

    **Parameters**\n
    ----------\n
    feature: One of the features of bbox such as height, width, label,
             top, left.

    **Returns**\n
    -------\n
    att: All attributes of a feature
  """
  ###START CODE###
  if (len(feature) > 1):
    att = [digit_structure[feature.value[j].item()].value[0][0]
                    for j in range(len(feature))]
  else:
    att = [feature.value[0][0]]
  return att
  ###END CODE###

def get_name(n,digitStructBbox,digitStructName,digit_structure):
  """
  Gets the name of an image.

    **Parameters**\n
    ----------\n
    n: the iteration of a for function represents the nth image.

    **Returns**\n
    -------\n
    The name of the image(i.e. 1.png)
  """
  ###START CODE###
  return ''.join([chr(c[0]) for c in digit_structure[digitStructName[n][0]].value])
  ###END CODE###

def get_bbox(n,digitStructBbox,digitStructName,digit_structure):
  """
  Extracts the attributes of an image.

    **Parameters**\n
    ----------\n
    n: the iteration of a for function represents the nth image.

    **Returns**\n
    -------\n
    att: All attributes of an image
  """
  ###START CODE###

  bbox={}
  bbox['height'] = bbox_Dims(digit_structure[digitStructBbox[n].item()]["height"],digit_structure)
  bbox['label'] = bbox_Dims(digit_structure[digitStructBbox[n].item()]["label"],digit_structure)
  bbox['left'] = bbox_Dims(digit_structure[digitStructBbox[n].item()]["left"],digit_structure)
  bbox['top'] = bbox_Dims(digit_structure[digitStructBbox[n].item()]["top"],digit_structure)
  bbox['width'] = bbox_Dims(digit_structure[digitStructBbox[n].item()]["width"],digit_structure)
  return bbox
  ###END CODE###

def get_digit_struct(n,digitStructBbox,digitStructName,digit_structure):
  """
  Gets the bbox attributes and name of an image.

    **Parameters**\n
    ----------\n
    n: the iteration of a for function represents the nth image.

    **Returns**\n
    -------\n
    img_dig_str: A h5py dictionary with all attributes of a single image
  """
  ###START CODE###
  img_dig_str=get_bbox(n,digitStructBbox,digitStructName,digit_structure)
  img_dig_str["name"]=get_name(n,digitStructBbox,digitStructName,digit_structure)
  return img_dig_str
  ###END CODE###

def get_Digit_Structure(digitStructBbox,digitStructName,digit_structure):
  """
  Gets the structures of all images from "digitStruct.mat" file

    **Returns**\n
    -------\n
    A dictionary with all attributes of a all images
  """
  return [get_digit_struct(i,digitStructBbox,digitStructName,digit_structure) for i in range(len(digitStructName))]

def get_Img_Struct(digitStructBbox,digitStructName,digit_structure):
  print("Getting the image structures...")
  dig_str = get_Digit_Structure(digitStructBbox,digitStructName,digit_structure)
  print("Converting to a python dictionary...")
  result = []
  structCnt = 1
  for i in range(len(dig_str)):
    item = { 'filename' : dig_str[i]["name"] }
    figures = []
    for j in range(len(dig_str[i]['height'])):
      figure = {}
      figure['height']=dig_str[i]['height'][j]
      figure['label']=dig_str[i]['label'][j]
      figure['left']=dig_str[i]['left'][j]
      figure['top']=dig_str[i]['top'][j]
      figure['width']=dig_str[i]['width'][j]
      figures.append(figure)
    structCnt = structCnt + 1
    item['boxes'] = figures
    result.append(item)
  print("The structures of images successfully extracted!")
  return result


def generate_dataset(data, folder):

    dataset = np.ndarray([len(data),64,64,3], dtype='float32')
    labels = np.ones([len(data),6], dtype=int) * 10
    for i in np.arange(len(data)):
        filename = data[i]['filename']
        fullname = os.path.join(folder, filename)
        im = Image.open(fullname)
        boxes = data[i]['boxes']
        num_digit = len(boxes)
        labels[i,0] = num_digit
        top = np.ndarray([num_digit], dtype='float32')
        left = np.ndarray([num_digit], dtype='float32')
        height = np.ndarray([num_digit], dtype='float32')
        width = np.ndarray([num_digit], dtype='float32')
        for j in np.arange(num_digit):
            if j < 5:
                labels[i,j+1] = boxes[j]['label']
                if boxes[j]['label'] == 10:
                  labels[i,j+1] = 0
            else: print('#',i,'image has more than 5 digits.')
            top[j] = boxes[j]['top']
            left[j] = boxes[j]['left']
            height[j] = boxes[j]['height']
            width[j] = boxes[j]['width']

        global im_top
        im_top = np.amin(top)
        global im_left
        im_left = np.amin(left)
        im_height = np.amax(top) + height[np.argmax(top)] - im_top
        im_width = np.amax(left) + width[np.argmax(left)] - im_left

        im_top = np.floor(im_top - 0.1 * im_height)
        im_left = np.floor(im_left - 0.1 * im_width)
        global im_bottom
        im_bottom = np.amin([np.ceil(im_top + 1.2 * im_height), im.size[1]])
        global im_right
        im_right = np.amin([np.ceil(im_left + 1.2 * im_width), im.size[0]])

        im = im.crop((im_left, im_top, im_right, im_bottom)).resize([64,64], Image.ANTIALIAS)
        im=np.array(im, dtype='float32')
        dataset[i,:,:,:] = im[:,:,:]
    return dataset, labels


def plotting(train,test,row,column):
  plt.figure(figsize=(10,10)) # Plot first 25 images
  for i in range(row*column):
      plt.subplot(row,column,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(train[i], cmap=plt.cm.binary)
      plt.xlabel(test[i])
  plt.show()

def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(name, shape):
    return tf.get_variable(name,shape=shape,initializer=tf.contrib.layers.xavier_initializer_conv2d())


class Model(object):
    """CNN architecture:
       INPUT -> CONV -> RELU -> CONV -> RELU ->
       POOL -> CONV -> POOL -> FC -> RELU -> 5X SOFTMAX
    """

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Create placeholders for feed data into graph
            self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
            self.y = tf.placeholder(tf.int32, shape=[None, 6])
            self.keep_prob = tf.placeholder(tf.float32)

            # First convolutional layer
            # 16 filters - size(11x11x3)
            W_conv1 = weight_variable("W_c1", [11, 11, 3, 16])
            b_conv1 = bias_variable("B_c1", [16])
            h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
            h_norm1 = tf.nn.local_response_normalization(h_conv1)
            h_drop1 = tf.nn.dropout(h_norm1, self.keep_prob)

            # Second convolutional layer
            # 32 filters - size(7x7x3)
            W_conv2 = weight_variable("W_c2", [9, 9, 16, 32])
            b_conv2 = bias_variable("B_c2", [32])
            h_conv2 = tf.nn.relu(conv2d_2(h_drop1, W_conv2) + b_conv2)
            h_norm2 = tf.nn.local_response_normalization(h_conv2)
            h_pool2 = max_pool_2x2(h_norm2)
            h_drop2 = tf.nn.dropout(h_norm2, self.keep_prob)

            # Third convolutional layer
            # 64 filters - size(5x5x16)
            W_conv3 = weight_variable("W_c3", [5, 5, 32, 64])
            b_conv3 = bias_variable("B_c3", [64])
            h_conv3 = tf.nn.relu(conv2d(h_drop2, W_conv3) + b_conv3)
            h_norm3 = tf.nn.local_response_normalization(h_conv3)
            h_pool3 = max_pool_2x2(h_norm3)
            h_drop3 = tf.nn.dropout(h_pool3, self.keep_prob)

            # Third convolutional layer
            # 64 filters - size(3x3x32)
            W_conv4 = weight_variable("W_c4", [3, 3, 64, 128])
            b_conv4 = bias_variable("B_c4", [128])
            h_conv4 = tf.nn.relu(conv2d_2(h_drop3, W_conv4) + b_conv4)
            h_norm4 = tf.nn.local_response_normalization(h_conv4)
            h_pool4 = max_pool_2x2(h_norm4)


            # Reshape tensor from POOL layer for connection with FC
            h_pool4_flat = tf.reshape(h_pool4, [-1, 13*13*128])
            h_drop4 = tf.nn.dropout(h_pool4_flat, self.keep_prob)

            # Fully connected layer
            W_fc1 = weight_variable("W_fc1", [13 * 13 * 128, 512])
            b_fc1 = bias_variable("B_fc1", [512])
            h_fc1 = tf.nn.relu(tf.matmul(h_drop4, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            # Second Fully connected layer
            W_fc2 = weight_variable("W_fc2", [512, 512])
            b_fc2 = bias_variable("B_fc2", [512])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob)


            # Create variables for 5 softmax classifiers
            W1 = tf.get_variable(shape=[512, 11], name="W1",initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable(shape=[512, 11], name="W2",initializer=tf.contrib.layers.xavier_initializer())
            W3 = tf.get_variable(shape=[512, 11], name="W3",initializer=tf.contrib.layers.xavier_initializer())
            W4 = tf.get_variable(shape=[512, 11], name="W4",initializer=tf.contrib.layers.xavier_initializer())
            W5 = tf.get_variable(shape=[512, 11], name="W5",initializer=tf.contrib.layers.xavier_initializer())

            # Create biases for 5 softmax classifiers
            b1 = bias_variable("B1", [11])
            b2 = bias_variable("B2", [11])
            b3 = bias_variable("B3", [11])
            b4 = bias_variable("B4", [11])
            b5 = bias_variable("B5", [11])

            # Create logits
            self.logits_1 = tf.matmul(h_fc2_drop, W1) + b1
            self.logits_2 = tf.matmul(h_fc2_drop, W2) + b2
            self.logits_3 = tf.matmul(h_fc2_drop, W3) + b3
            self.logits_4 = tf.matmul(h_fc2_drop, W4) + b4
            self.logits_5 = tf.matmul(h_fc2_drop, W5) + b5

            # Define L2 Regularization, lambda == 0.001
            regularizer = (0.005*tf.nn.l2_loss(W_conv1) + 0.005*tf.nn.l2_loss(W_conv2) + \
                             0.005*tf.nn.l2_loss(W_conv3) + 0.005*tf.nn.l2_loss(W_conv4)  + 0.005*tf.nn.l2_loss(W_fc1) +  \
                             0.005*tf.nn.l2_loss(W_fc2) + 0.005*tf.nn.l2_loss(W1) + 0.005*tf.nn.l2_loss(W2) + \
                             0.005*tf.nn.l2_loss(W3) + 0.005*tf.nn.l2_loss(W4) + \
                             0.005*tf.nn.l2_loss(W5))

            # Define cross entropy loss function
            self.loss = (tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.logits_1, labels=self.y[:, 1])) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.logits_2, labels=self.y[:, 2])) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.logits_3, labels=self.y[:, 3])) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.logits_4, labels=self.y[:, 4])) + \
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=self.logits_5, labels=self.y[:, 5])) + regularizer)

            # Define optimizer.
            # Starting learning rate == 0.05, decay_steps == 10000, decay_rate == 0.96
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.96)
            self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(
                                                    self.loss, global_step=global_step)
            # Create saver
            self.saver = tf.train.Saver()

def predictions(logit_1, logit_2, logit_3, logit_4, logit_5):
    """Converts predictions into understandable format.
    For example correct prediction for 2 will be > [2,10,10,10,10]
    """
    first_digits = np.argmax(logit_1, axis=1)
    second_digits = np.argmax(logit_2, axis=1)
    third_digits = np.argmax(logit_3, axis=1)
    fourth_digits = np.argmax(logit_4, axis=1)
    fifth_digits = np.argmax(logit_5, axis=1)
    stacked_digits = np.vstack((first_digits, second_digits, third_digits, fourth_digits, fifth_digits))
    rotated_digits = np.rot90(stacked_digits)[::-1]
    return rotated_digits

def accuracy(logit_1, logit_2, logit_3, logit_4, logit_5, y_):
    """Computes accuracy"""
    correct_prediction = []
    y_ = y_[:, 1:]
    rotated_digits = predictions(logit_1, logit_2, logit_3, logit_4, logit_5)
    for e in range(len(y_)):
        if np.array_equal(rotated_digits[e], y_[e]):
            correct_prediction.append(True)
        else:
            correct_prediction.append(False)
    return (np.mean(correct_prediction))*100.0


def train(train_dataset, train_labels,test_dataset, test_labels, batch_size=64, number_of_iterations=40000):
    """Trains CNN."""
    x_train, y_train, x_test, y_test = train_dataset, train_labels,test_dataset, test_labels
    print ("Data uploaded!")

    model = Model()
    with tf.Session(graph=model.graph) as session:
        z, n = 0, 0
        train_iter_acc=[]
        tf.global_variables_initializer().run()

        for i in range(number_of_iterations):
            indices = np.random.choice(len(y_train), batch_size)
            bat_x = x_train[indices]
            bat_y = y_train[indices]
            _, l = session.run([model.optimizer, model.loss], feed_dict={model.x: bat_x, model.y: bat_y,
                                                                       model.keep_prob: 0.5})

            # Check loss
            if i % 500 == 0:
                log_1, log_2, log_3, log_4, log_5, y_ = session.run([model.logits_1, model.logits_2, model.logits_3,
                                                                    model.logits_4, model.logits_5, model.y],
                                                                    feed_dict={model.x: bat_x, model.y: bat_y,
                                                                               model.keep_prob: 1.0})
                print ("Iteration number: {}".format(i))
                train_iter_acc.append(accuracy(log_1, log_2, log_3, log_4, log_5, y_))
                print ("Loss: {:.3f}".format(l))


        # Evaluate accuracy by parts, if you use GPU and it has low memory.
        z=0
        for el in range(6):
            log_1, log_2, log_3, log_4, log_5, y_ = session.run([model.logits_1, model.logits_2, model.logits_3,
                                                                model.logits_4, model.logits_5, model.y],
                                                                feed_dict={model.x: x_test[n:n+2178],
                                                                           model.y: y_test[n:n+2178],
                                                                           model.keep_prob: 1.0})
            n += 2178

            pred=predictions(log_1, log_2, log_3, log_4, log_5)
            if el==0:
                pred_last=pred
            else:
                pred_last=np.append(pred_last,pred,axis=0)


        test_lbl=test_labels[:,1:6]
        na_zero = np.where(test_lbl.flatten() != 10)
        report = classification_report(pred_last.flatten()[na_zero], test_lbl.flatten()[na_zero], digits=4)
        print("**Classification report** \n",report)


        # Save model in file "try1.ckpt"
        model.saver.save(session, "./tnry1.ckpt")
    #return train_loss



def test(image):
  test_im=np.ndarray([1,64,64,3], dtype='float32')
  im = Image.open(image)
  splt=image.split(".")
  im = im.resize([64,64], Image.ANTIALIAS)
  im=np.array(im, dtype='float32')
  im=(im - 114.00535)/52.501564
  test_im[0,:,:,:] = im[:,:,:]
  print(test_im.shape)

  model = Model()

  with tf.Session(graph=model.graph) as session:

    # Restore model on test data
    model.saver.restore(session, "./tnry1.ckpt")

    for el in range(6):
      log_1, log_2, log_3, log_4, log_5, = session.run([model.logits_1, model.logits_2, model.logits_3,
                                                                model.logits_4, model.logits_5],
                                                                feed_dict={model.x: test_im,
                                                                           model.keep_prob: 1.0})


  # Make predictions on 7 random examples from test data
  pred = predictions(log_1, log_2, log_3, log_4, log_5)
  na_zero_pred = np.where(pred.flatten() != 10)
  plotting(test_im,"-",1,1)
  print(pred.flatten()[na_zero_pred])



def traintest():

  download("train.tar.gz")
  download("test.tar.gz")

  train_folder=extract("train.tar.gz")
  test_folder=extract("test.tar.gz")

  digit_structure=h5py.File("train/digitStruct.mat","r")
  digitStructName=digit_structure['digitStruct']['name']
  digitStructBbox=digit_structure['digitStruct']['bbox']
  train_data=get_Img_Struct(digitStructBbox,digitStructName,digit_structure)

  digit_structure=h5py.File("test/digitStruct.mat","r")
  digitStructName=digit_structure['digitStruct']['name']
  digitStructBbox=digit_structure['digitStruct']['bbox']
  test_data=get_Img_Struct(digitStructBbox,digitStructName,digit_structure)

  # Get Training Image size
  print("Getting the size of all images")
  train_imsize = np.ndarray([len(train_data),2])
  for i in np.arange(len(train_data)):
    filename = train_data[i]['filename']
    fullname = os.path.join(train_folder, filename)
    im = Image.open(fullname)
    train_imsize[i, :] = im.size[:]

# Get Training Image size
  test_imsize = np.ndarray([len(test_data),2])
  for i in np.arange(len(test_data)):
    filename = test_data[i]['filename']
    fullname = os.path.join(test_folder, filename)
    im = Image.open(fullname)
    test_imsize[i, :] = im.size[:]

  print('Success!')

  print('Generating training dataset and labels...')
  train_dataset, train_labels = generate_dataset(train_data, train_folder)
  print('Success! \n Training set: {} \n Training labels: {}'.format(train_dataset.shape, train_labels.shape))

  print('Generating testing dataset and labels...')
  test_dataset, test_labels = generate_dataset(test_data, test_folder)
  print('Success! \n Testing set: {} \n Testing labels: {}'.format(test_dataset.shape, test_labels.shape))

  #Delete the six digit image
  train_dataset = np.delete(train_dataset, 29929, axis=0)
  train_labels = np.delete(train_labels, 29929, axis=0)

  mean = np.mean(train_dataset, dtype='float32')
  std = np.std(train_dataset, dtype='float32')
  train_dataset = (train_dataset - mean) / std
  test_dataset = (test_dataset - mean) / std

  print("25 images from training set after normalization",train_dataset.shape,train_labels.shape)
  (train_dataset,train_labels,5,5)
  print("25 images from test set after normalization",test_dataset.shape,test_dataset.shape)
  plotting(test_dataset,test_labels,5,5)

  print("Training initiated waiting for the data to be uploaded...")
  train(train_dataset, train_labels,test_dataset, test_labels)
