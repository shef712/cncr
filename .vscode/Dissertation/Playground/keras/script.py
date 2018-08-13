# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import os
import sys
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt(os.path.join(sys.path[0], "pima-indians-diabetes.csv"), delimiter=",")
# split into input (X) and output (Y) variables, and test variables for evaluation
X = dataset[:-10,:8]
Y = dataset[:-10,8]
X_test = dataset[-10:,:8]
Y_test = dataset[-10:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# # calculate predictions
predictions = model.predict(X_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)