import keras
from keras import regularizers
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Conv1D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from sklearn.metrics import cohen_kappa_score, accuracy_score
import numpy as np
import os
from PIL import ImageFile,Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy
import json
from keras.models import load_model
import gensim
from bs4 import BeautifulSoup
import requests
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('stopwords')
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import streamlit as st
from io import BytesIO
import urllib.request

st.title('CFMI - Clickbait Detector')

def process_title(title):
    title = title.lower()
    stemmer = PorterStemmer()
    stopwords_english = STOP_WORDS 
    # remove hashtags
    # only removing the hash # sign from the word
    title = re.sub(r'#', '', title)
    # tokenize tweets
    eng_tokenize = English()
    title_doc =  eng_tokenize(title)
    
    token_list = []
    title_clean = []
    for token in title_doc:
        token_list.append(token.text)
        
    for word in token_list:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            title_clean.append(stem_word)

    title_clean = [*set(title_clean)] #Remove duplicate words
    return title_clean

def mean_embed(tokens,model):
    vector_list = []
    
    for x in tokens:
        if x in model.wv:
            val = model.wv[x]
            vector_list.append(val)
            
    return np.mean(np.array(vector_list), axis=0)

def get_comments_count(link):
    r = requests.get(link, headers={'User-Agent': ''})
    r = r.text[r.text.find('commentCount'):r.text.find('commentCount')+40:].split("\"")
    if 'K' in r[4]:
        comments = float(r[4].strip('K').strip('.'))*1000
    elif 'M' in r[4]:
        comments = float(r[4].strip("M").strip('.')) * 1000000
    else:
        comments = int(r[4].replace(",",""))
    return int(comments)


def imageToArray(image_path):
  # Load the image and resize it to the desired dimensions
  width, height = 240, 240  # Replace with the dimensions required by your model

  image = Image.open(image_path)
  image = image.resize((width, height))
  print(image.width)
  # Convert the image to a NumPy array and normalize the pixel values (if necessary)
  image_array = np.asarray(image)
  image_array = image_array / 255.0  # Normalize the pixel values between 0 and 1

  print(image_array.shape)
  # Reshape the image array to match the input shape of your model
  image_array = image_array.reshape(1, width, height, 3)  # Assumes the input shape is (width, height, 3)

  return image_array


main_model = load_model('model_85acc.h5')
ftt_model = gensim.models.Word2Vec.load("FastTest.kv")


link = st.text_input(label = 'Enter a Youtube Link Here....')

if link is not '':
    response = requests.request("GET", link)
    soup = BeautifulSoup(response.text, "html.parser")
    body = soup.find_all("body")[0]
    scripts = body.find_all("script")
    result = json.loads(scripts[0].string[30:-1])

    title = result['videoDetails']['title']
    title_main = title
    title_mean = mean_embed(title,ftt_model)
    views = result['videoDetails']['viewCount']
    views = int(views.replace(",",''))
    thumbnail_link = result['videoDetails']['thumbnail']['thumbnails'][-1]['url']

    urllib.request.urlretrieve(thumbnail_link, "thumbnail.jpg")
    st.image("thumbnail.jpg")

    r = requests.get(link, headers={'User-Agent': ''})
    likes = r.text[:r.text.find(' likes"')]
    likes = likes[likes.rfind('"') +1:]
    likes = int(likes.replace(",",''))

    dislikes = r.text[:r.text.find(' dislikes"')]
    dislikes = dislikes[dislikes.rfind('"') + 0:]
    dislikes = dislikes.split(" ")
    dislikes = dislikes[4]
    dislikes = int(dislikes[:-2]) * 10

    comments_count = get_comments_count(link)

    title = process_title(title)
    st.markdown(f"<h3 color: white;'>Title : {title_main}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 color: white;'>Total View Count : {str(views)}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 color: white;'>Total Likes : {str(likes)}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 color: white;'>Total Dislikes : {str(dislikes)}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 color: white;'>Comment Count : {str(comments_count)}</h3>", unsafe_allow_html=True)

    img_array = imageToArray("thumbnail.jpg")
    numerical_data = np.concatenate(([views, likes, dislikes, comments_count], title_mean))

    images = tf.convert_to_tensor(np.array(img_array, dtype=np.float32) / 255.0)
    numerical_data = tf.convert_to_tensor(np.array(numerical_data), dtype=np.float32)

    image_data = np.expand_dims(images, axis=0)
    numerical_data = np.expand_dims(numerical_data, axis=0)

    ans = main_model.predict([images, numerical_data])
    print(ans)

    if ans>0.375:
        st.markdown("<h2 style='text-align: center; color: green;'>This video is not a clickbait</h2>", unsafe_allow_html=True)

    else:
        st.markdown("<h2 style='text-align: center; color: red;'>This video is a clickbait</h2>", unsafe_allow_html=True)

