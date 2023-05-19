import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# Acquire the 10000 most common words
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# Integer encoded words
print('Review example: ', train_data[0])

_word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in _word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

# Reversing the dictionary, weird behavior of tensorflow
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# We need to use the <PAD> to make all reviews the same length, because of how
# neural networks work. All reviews will be 250 words long.
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding='post',
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=250)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# Model definition
model = keras.Sequential()
# Tries to group words that are similar, more about this below
model.add(keras.layers.Embedding(88000, 16))
# Condenses the sequence of word embeddings into a single 16-dimensional vector
# by computing the average of the embeddings.
# It captures the overall meaning of the input text, reducing dimensionality and
# making it suitable for downstream processing and classification tasks.
model.add(keras.layers.GlobalAveragePooling1D())
# 16 neurons is an arbitrary number. Takes the word vector (similar words), tries
# to pick up where they are used in sentences
model.add(keras.layers.Dense(16, activation='relu'))
# Output node, 1 node saying the review output
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Validation data
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

# Batch size = how many samples are used to update the model's weights in each iteration
model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# 0 = BAD, 1 = GOOD
for x in range(20):
    predict = model.predict([test_data[x]])
    print("Review: ", decode_review(test_data[x]))
    print('Prediction: ', str(predict[x]))
    print('Actual: ', str(test_labels[x]))

results = model.evaluate(test_data, test_labels)
print(results)

# Save the model with h5 extension (for keras and tensorflow)
model.save("model.h5")

# It can be loaded like this:
model = keras.models.load_model('model.h5')

'''
The embedding layer in our model generates 1 word vector for each word it 
analyzes. These word vector serves as representations of individual words within 
the text. Each word is mapped to a 16-dimensional vector, capturing its specific 
characteristics and contextual information.
The word vector is built from the input data itself, using methods such as GloVe
and word2vec to capture these relationships.
This allows the model to better understand the diverse range of words it 
encounters and effectively capture the semantic relationships and context between 
them.
'''
