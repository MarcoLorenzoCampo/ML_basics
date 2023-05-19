from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Loading the data
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Class names from tensorflow fashion example 0 to 9
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Pixel value 0-255 greyscale
# print(train_images[8])

# Scale all values capping at 1
train_images = train_images/255.0
test_images = test_images/255.0

'''plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.show()'''

'''
To feed to our NN, we need to flatten the data of the images, this means that we
make something like [[1], [2], [3]] --> [1, 2, 3]
We get an array of 784 elements for each image, compared to 28 arrays os 28 pixels
that we had before flattening
'''

# Create a model for the Neural Network. Sequential --> Sequence of layers:
# First we flatten the data
# We add the first layer that takes 28x28=784 inputs, so 784 input nodes
# We add the middle hidden layer of 128 fully connected nodes (around 20% of input nodes)
# We add the output node of 10 fully connected classes
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Setting some parameters for the NN
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
# epochs = the number of times each image can be seen by the NN
# the order of the data can influence the perception and so the tweaking of params
model.fit(train_images, train_labels, epochs=5)

'''
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Tested accuracy: ', test_acc)
'''

prediction = model.predict(test_images)

'''
print(prediction[0])

[5.4662760e-06 3.5943279e-09 1.3284134e-07 2.9108034e-07 1.3063847e-07
 1.6902683e-03 2.9177670e-06 8.0938593e-02 1.7106835e-05 9.1734517e-01]
 
For each class, it gives a probability the 0 element belongs to it, the highest
is the most likely
'''

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()