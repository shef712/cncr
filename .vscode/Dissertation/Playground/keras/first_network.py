from keras.models import Sequential
from keras.layers import Dense
import numpy
import os
import sys

# Set random seed for reproducibility (useful if you need to demonstrate a result, compare algorithms using the same source of randomness or to debug a part of your code)
numpy.random.seed(7)


# We begin by loading the data

# Using patient medical record data for Pima Indians and whether they had an onset of diabetes within five years.

# Binary classification problem (onset of diabetes as 1 or not as 0). 
# All of the input variables that describe each patient are numerical. This makes it easy to use directly with neural networks that expect numerical input and output values, and ideal for our first neural network in Keras. 

# Load pima indians dataset
path_to_file = os.path.join(sys.path[0], "pima-indians-diabetes.csv")
#/home/shaf/workspace/cncr/.vscode/Dissertation/Playground/keras/pima-indians-diabetes.csv
dataset = numpy.loadtxt(path_to_file, delimiter=",")

# Split into input (X) and output (Y) variables
X = dataset[:,0:8] # Get all rows and the column from 0 - 7
Y = dataset[:,8] # Get all rows and column 8


# Personal step of splitting the data into training and testing data
X = dataset[:-10,:8]
Y = dataset[:-10,8]
X_test = dataset[-10:,:8]
Y_test = dataset[-10:,8]


# We now define the (NN) model

# Models in Keras are defined as a sequence of layers
# We create a Sequential model and add layers one at a time until we are happy with our network topology

# Create the model
model = Sequential()

# We define the number of layers, in this case we will use 3 fully connected layers

# Fully connected layers are defined using the Dense class. 
# We can specify the number of neurons in the layer as the first argument, the initialization method as the second argument as init and specify the activation function using the activation argument.

model.add(Dense(12, input_dim=8, activation='relu'))

# This will connect the input to the first layer, we specify number of inputs the first layer should expect (8), as well as the number of neurons in the first layer (12), so the weight matrix in the first layer will be (8x12), 12 weight connections for each of the 8 input neurons

# Initialize the network weights to a small random number generated from a uniform distribution (‘uniform‘), in this case between 0 and 0.05 because that is the default uniform weight initialization in Keras. 
# (Another traditional alternative would be ‘normal’ for small random numbers generated from a Gaussian distribution)

model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# We will use the rectifier (‘relu‘) activation function on the first two layers and the sigmoid function in the output layer. It used to be the case that sigmoid and tanh activation functions were preferred for all layers. These days, better performance is achieved using the rectifier activation function. We use a sigmoid on the output layer to ensure our network output is between 0 and 1 and easy to map to either a probability of class 1 or snap to a hard classification of either class with a default threshold of 0.5.

# Note the number of neurons chosen in the corresponding layer need to be chosen such that the weight matrix computations are compatible! (1) 1x8 * 8x12 = 1x12, (2) 1x12 * 12x8 = 1x8, (3) 1x8 * 8x1 = 1x1

# Now that the model is defined, we can compile it.

# Compiling the model uses the efficient numerical libraries under the covers (the so-called backend) using TensorFlow.

# When compiling, we must specify some additional properties required when training the network. 

# We must specify the loss function to use to evaluate a set of weights, the optimizer used to search through different weights for the network and any optional metrics we would like to collect and report during training (remember training a network means finding the best set of weights to make predictions for this problem)

# In this case, we will use logarithmic loss, and use the efficient gradient descent algorithm “adam”. Finally, because it is a classification problem, we will collect and report the classification accuracy as the metric.

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# We have defined our model and compiled it ready for efficient computation.

# Now it is time to execute the model on some data, i.e. "fit the model"

# We can train or fit our model on our loaded data by calling the fit() function on the model.

model.fit(X, Y, epochs=150, batch_size=10)

# The training process will run for a fixed number of iterations through the dataset called epochs, that we must specify using the nepochs argument. 

# We can also set the number of instances that are evaluated before a weight update in the network is performed, called the batch size and set using the batch_size argument. 
# So 10 sets of inputs are evaluated first before a weight update occurs.
# This training process occurs 150 times over the entire dataset

# For this problem, we will run for a small number of iterations (150) and use a relatively small batch size of 10. Again, these can be chosen experimentally by trial and error.


# The network is now trained.

# We can evaluate the performance of the network on the same dataset.

# We will evaluate the performance on new data. We can do this using the evaluate() function on the model and pass it the same input and output used to train the model.

# This will generate a prediction for each input and output pair and collect scores, including the average loss and any metrics you have configured, such as accuracy.

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Should see a message for each of the 150 epochs printing the loss and accuracy for each, followed by the final evaluation of the trained model on the training dataset.

# Neural networks are a stochastic algorithm, meaning that the same algorithm on the same data can train a different model with different skill. This is a feature, not a bug. 
# We did try to fix the random seed to ensure that the article and I get the same model and therefore the same results, but this does not always work on all systems.



# We can also make predictions

# We can adapt the above example and use it to generate predictions on the training dataset, pretending it is a new dataset we have not seen before.

# Making predictions is as easy as calling model.predict(). We are using a sigmoid activation function on the output layer, so the predictions will be in the range between 0 and 1. We can easily convert them into a crisp binary prediction for this classification task by rounding them.

# # calculate predictions
predictions = model.predict(X_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

# Running this now prints the predictions for each input pattern. We could use these predictions directly in our application if needed.











