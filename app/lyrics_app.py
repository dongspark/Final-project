#import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.preprocessing.text import Tokenizer
#from  tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dropout
#from tensorflow.keras.optimizers import Adam
import numpy as np 

import pandas as pd
#from sklearn.preprocessing import OneHotEncoder

#import re

import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from PIL import Image
import pickle

from rdkit import Chem
from rdkit.Chem import Descriptors

model = keras.models.load_model('models/bi_lstm')
word_index = pickle.load(open('word_index.p','rb'))
input_sequences = pickle.load(open('input_sequences.p','rb'))
max_sequence_len = max([len(x) for x in input_sequences])
corpus_cleaned = pd.read_csv('corpus_cleaned.csv')
tokenizer = pickle.load(open('tokenizer.p','rb'))

st.write("""
# Lyrics and image Web App
This app counts the nucleotide composition of query DNA!
***
""")

seed_text='the blue sky'

st.sidebar.header('User Input Features')
lyrics=st.sidebar.text_area('lytics input',seed_text)




st.header('Generated lyrics')

next_words = 20

output_index = [0,0,0]
output_word = [0,0,0]
  
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    print(predicted[0])
    print(output_index[-1], output_index[-2])
    
    # if-clauses I added, because otherwise the model sometimes gives the same word several times in a row:
    if predicted[0] == output_index[-1]:
        predicted = np.argsort(model.predict(token_list), axis=-1)[0][-2]
    
    # also this. Otherwise it sometimes gives a pair of two words several times in a row:
    if (predicted[0] == output_index[-2]) & (output_word[-1]==output_word[-3]):
        predicted = np.argsort(model.predict(token_list), axis=-1)[0][-2]
    #print(model.predict(token_list))
     
    
    #output_word = [0,0,0]
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word.append(word)
            output_index.append(index)
            break
    seed_text += " " + output_word[-1]
#print(seed_text)

seed_text


st.header('Generated image')


