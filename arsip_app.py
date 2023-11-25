from flask import Flask, render_template, request
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense
import numpy as np
import pandas as pd
import string
import random

app = Flask(__name__)

# Variabel global
words, classes, tokenizer = [], [], None

def preprocess_data():
    global words, classes, tokenizer
    # Importing the dataset
    with open('kampus_merdeka.json') as content:
        data1 = json.load(content)

    tags = []  # data tag
    inputs = []  # data input atau pattern
    responses = {}  # data respon
    ignore_words = ['?', '!']  # Mengabaikan tanda spesial karakter

    # Tambahkan data intents dalam json
    for intent in data1['intents']:
        responses[intent['tag']] = intent['responses']
        for lines in intent['patterns']:
            inputs.append(lines)
            tags.append(intent['tag'])
            # digunakan untuk pattern atau teks pertanyaan dalam json
            for pattern in intent['patterns']:
                w = nltk.word_tokenize(pattern)
                words.extend(w)
                # tambahkan ke dalam list kelas dalam data
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

    # Konversi data json ke dalam dataframe
    data = pd.DataFrame({"patterns": inputs, "tags": tags})

    # Removing Punctuations (Menghilangkan Punktuasi)
    data['patterns'] = data['patterns'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
    data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    # sorting pada data class
    classes = sorted(list(set(classes)))

    # Tokenize the data (Tokenisasi Data)
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(data['patterns'])
    train = tokenizer.texts_to_sequences(data['patterns'])

    # Melakukan proses padding pada data
    x_train = pad_sequences(train)

    # Melakukan konversi data label tags dengan encoding
    le = LabelEncoder()
    y_train = le.fit_transform(data['tags'])

    # Melihat hasil input pada data teks
    input_shape = x_train.shape[1]

    return x_train, y_train, input_shape, le, responses

def create_model(input_shape, vocabulary, output_length):
    # Creating the model (Membuat Modelling)
    i = Input(shape=(input_shape,))  # Layer Input
    x = Embedding(vocabulary + 1, 10)(i)  # Layer Embedding
    x = LSTM(10, return_sequences=True, recurrent_dropout=0.2)(x)  # Layer Long Short Term Memory
    x = Flatten()(x)  # Layer Flatten
    x = Dense(output_length, activation="softmax")(x)  # Layer Dense
    model = Model(i, x)  # Model yang telah disusun dari layer Input sampai layer Output

    # Compiling the model (Kompilasi Model)
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    return model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global words, classes, tokenizer
    if request.method == 'POST':
        prediction_input = request.form['user_input']

        x_train, y_train, input_shape, le, responses = preprocess_data()
        model = create_model(input_shape, len(words), len(classes))
        
        output = model.predict(pad_sequences(tokenizer.texts_to_sequences([prediction_input]), maxlen=input_shape))
        output = output.argmax()

        response_tag = le.inverse_transform([output])[0]
        bot_response = random.choice(responses[response_tag])

        return render_template('index.html', user_input=prediction_input, bot_response=bot_response)

if __name__ == '__main__':
    app.run(debug=True)
