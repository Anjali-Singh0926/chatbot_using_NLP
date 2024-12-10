import random
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import nltk
nltk.download('punkt')
import nltk
nltk.download('wordnet')
import nltk
nltk.download('punkt_tab')
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('dataset.json').read())

words = []
classes = []
documents = []
ignore_letters = ['!', '?', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(classes))

classes = sorted(set(classes))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words: bag.append(1) if word in word_patterns else bag.append(0)

    outputRow = list(output_empty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)
# Splitting training data into features (trainX) and labels (trainY)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]
#Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]),activation = 'softmax'))
#compile the model with optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum= 0.9, nesterov=True)
model.compile (loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#train the model
hist = model.fit(np.array(trainX),np.array(trainY),epochs = 200, batch_size = 5, verbose = 1)
#save the model
model.save('chatbot_smartNavigator.h5',hist)
print("Executed")
