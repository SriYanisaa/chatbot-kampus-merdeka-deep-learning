from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense
from tensorflow.keras.models import Model
import pandas as pd
import string
import random

app = Flask(__name__)

# Load preprocessed data and model
with open('kampus_merdeka.json') as content:
    data1 = json.load(content)

tags, inputs, responses, words, classes, documents, ignore_words = [], [], {}, [], [], [], ['?', '!']

for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

data = pd.DataFrame({"patterns": inputs, "tags": tags})
data['patterns'] = data['patterns'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])
x_train = pad_sequences(train)

le = LabelEncoder()
y_train = le.fit_transform(data['tags'])
input_shape = x_train.shape[1]

vocabulary = len(tokenizer.word_index)
output_length = le.classes_.shape[0]

# Creating the model
i = Input(shape=(input_shape,))
x = Embedding(vocabulary + 1, 10)(i)
x = LSTM(10, return_sequences=True, recurrent_dropout=0.2)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Load pre-trained model weights
model.load_weights('model_chatbot_kampus_merdeka.h5')

def predict_chatbot(user_input):
    prediction_input = preprocess_input(user_input)
    prediction_input = tokenizer.texts_to_sequences([prediction_input])
    prediction_input = pad_sequences(prediction_input, maxlen=input_shape)
    output = model.predict(prediction_input)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    bot_response = random.choice(responses[response_tag])
    return bot_response

def preprocess_input(user_input):
    user_input = [letters.lower() for letters in user_input if letters not in string.punctuation]
    user_input = ''.join(user_input)
    return user_input

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        bot_response = predict_chatbot(user_input)
        return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
