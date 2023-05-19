import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

# Acquire the 10000 most common words
(train_data, train_labels), (test_data, test_label) = data.load_data(num_words=10000)

# Integer encoded words
print('Review example: ', train_data[0])

word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

# Reversing the dictionary, weird behavior of tensorflow
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return "".join([reverse_word_index.get(i, "?") for i in text])
