# Hand-Digit-Recognition-using-Deep_learning

Classifying hand-written digits using Convolutional Neural Network MNIST Dataset used for training the model

# About the Dataset
The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.

It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.

The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

It is a widely used and deeply understood dataset and, for the most part, is “solved.” Top-performing models are deep learning convolutional neural networks that achieve a classification accuracy of above 99%, with an error rate between 0.4 %and 0.2% on the hold out test dataset.

# Steps Involved :
IMPORT DATASET

• Tensorflowand Kerasallow us to import and download the MNIST dataset(Modified National Institute of Standards and Technology) directly from their API.

• The MNIST database contains 60,000 training images.

• 10,000 testing images.

PREPROCESS DATA

•Reshaping the array to 4-dims so that it can work with the KerasAPI(greyscale image).

• Need to normalize our data as it is always required in neural network models, by dividing the RGB codes to 255.

• So initially will be converting data to float as we are dividing.

BUILDING THE CONVOLUTIONAL NEURAL NETWORK

• "Sequential model" allows you to build a model layer by layer.model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))#28 number of layers.

• "Max pooling" is a pooling operation that selects the maximum element from the region of the feature map covered by the filter. Thus, the output after max-pooling layer would be a feature map containing the most prominent features of the previous feature map.

• "Flattening" involves transforming the entire pooled feature map matrix into a single column which is then fed to the neural network for processing.

• "DENSE LAYER" , A linear operation in which every input is connected to every output by a weight . It connects neurons in one layer to neurons in another layer. It is used to classify images between different category by training.

• "DROPOUT" : A Simple Way to Prevent Neural Networks from Overfitting.

• "SOFTMAX LAYER" is the last layer of CNN. It resides at the end of FC layer, softmaxis for multi-classification.

COMPILING AND TRAINING THE MODEL

Compiling the model takes three parameters: Optimizer, Loss and Metrics.

• Optimizer : controls the learning rate. We will be using ‘adam’ as our optmizer. Adam is generally a good optimizer to use for many cases. The adamoptimizer adjusts the learning rate throughout training.

• Loss: that can be used to estimate the loss of the model so that the weights can be updated to reduce the loss on the next evaluation.We use fit method then training starts.

EVALUATING THE MODEL

With a simple evaluate function to know the accuracy.
