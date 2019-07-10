# Street-View-House-Numbers-Recognition
## Abstract
This document defines, explains and demonstrates results of a model for image recognition of multi- digit numbers through a widely used approach of Deep Learning.

## 1. Model Design
The purpose of this paper is to perform classification task on the Street View House Numbers dataset [1]. 
The dataset contains random images of house numbers which are vary in size, angles that the shot was taken, position etc. 
There were many approaches made during past years to solve this problem. Some approaches used techniques such as localization
to find where are the digits located and after that applied deep learning techniques to classify them. One other technique 
was quite different from those techniques and still considered state-of-art for this dataset [2]. This paper will describe 
their work briefly and follow their path. Rather then using another technique for detecting the location of digits they used 
only convolutional layers, feed forward network connected to parallel softmax layers.<br>The model presented in this paper starts with downloading the full-size images with its link. Then the train and test files extracted to the working directory. Both files not only contain images but also the information about images in a file called digitStruct.mat. That file contains the relevant information about the bounding boxes that used to localize the image to find its location, its name and its label. The bounding box informations are distance from the top and the left of the picture also the height and the width. They are extracted from the digitStruct file as well as the label and the name of the image and stored in an array. After that all bounding box information analyzed. All top, left, height and width information stored separately and then for left and top the minimum for width and height the maximum values found in order to crop the pictures with the biggest bounding box that any picture in the dataset have with that method the images didn’t lose any important information, irrelevant information decreased and computational power saved. Also, for staying on the safe side adjustment made before cropping to extend the bounding boxes. To save more computational time the only picture which have six digits removed from the training set which reduced a softmax layer and normalization took place which was the last step before images 
went into the model.<br>
In addition to that the architecture built also influenced from [3]. 
The architecture described in the [3] was resizing the image to 32x32x3 because of the GPU issues and was using 
3 convolutional layers. First convolution had 16 5x5 feature detectors connected to relu activation and had a dropout, 
second one is also 5x5 with 32 feature detectors with relu activation, 2x2 pooling with 2 stride and a dropout layer, 
third one is the same with second except it has 64 filters. In addition to that all the convolutional layers used padding
to preserve the size of the image and take more information from the edges. Then the last convolution layer connected to a 
fully connected feed forward network with 1024 neurons and a single layer and relu as an activation function which feeds the 
results to 5 separate softmax layers for classifying 5 digits.<br>
The architecture this paper presents (Figure.1) first followed the architecture described and later changed to create a 
better solution. The architecture has 4 convolutional layer connected to a fully connected feed forward network which has
two hidden layers It first uses 16 11x11 feature detector with padding to preserve the image size, connected to a relu activation
function and a dropout layer (0.5), than second convolutional layer had 32 9x9 feature detectors without padding, connected to a
maximum pooling layer with 2 strides. That pooling also has a dropout (0.5) which passes information to the third convolutional layer. 
That layer has 64 filters which has 5x5 size. Later third layer passes information to activation function relu and then to the pooling 
layer same as the other ones described previously which connected to a dropout (0.5). Fourth convolutional layer has 128 3x3 feature 
detectors without padding and as before connected to a relu activation function and pooling layer. Then the information that those 
layers gathered fed into the feedforward network with two hidden layers both have 512 neurons and 0.5 dropout. The last step is to 
construct 5 parallel softmax layers for predicting digits. First softmax layer for first digit, second for second digit and so on. 
If there is no digit to predict the output was indented to be 10. Different hyperparameter settings were tried, lambda for regularization,
learning rate and dropout was chosen 0.05, 0.05 and 0.5 respectively. The images resized to 64x64x3 to get more information from them.
At the start of the model image fed in to the first layer and output had the same size, then after second layer the feature maps size
reduced to 28 then after third layer to 14 and before going into the fully connected network there was 13x13x128 feature maps.

## 2. Results
### A. The architecture of the influenced model
The architecture presented in [3] trained and evaluated. The model reached 82% accuracy on test set however on 
training the accuracy and loss was around %96 and 0.7 respectively which was a clear indicator of overfitting.
Also for normalization the mean and standard deviation calculated separately for both sets which leaked information 
to the test dataset.
