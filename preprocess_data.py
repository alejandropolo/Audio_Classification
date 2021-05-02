import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler




"""

GENERAMOS LA CONFIGURACIÓN DE LOS LOGS

"""

import logging

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Create handlers
f_handler = logging.FileHandler('{}.log'.format(__name__),mode='w+')
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(f_handler)




"""

DECLARAMOS LAS FUNCIONES

"""

def load_data(genres):
    ## Cargamos los géneros
    logger.debug('Cargando los datos...')

    song_names = []
    song_genres = []

    for genre in genres:
        directory = './genres/' + genre + '/'
        songs = os.listdir(directory)
        songs.sort()
        for song in songs:
            song_names.append(song)
            song_genres.append(genre)
            
    list_of_tuples = list(zip(song_names, song_genres)) 

    df = pd.DataFrame(list_of_tuples,
                    columns = ['Song', 'Genre']) 
    logger.debug('Datos cargados en el dataframe...')

    return df 



"""

Pasamos a la extraccion de features

"""


########## ZERO CROSSING RATE

def get_zcr(song_name):
    logger.debug('Extrayendo los zero crossing rate...')

    genre = song_name.split(".")[0]
    filename = './genres/' + genre + '/' + song_name
    x,sample_rate = librosa.load(filename)
    return np.mean(librosa.feature.zero_crossing_rate(x))
    logger.debug('Extrayendo los zero crossing rate...')

def extract_zcr(df):
    df['MFCC'] = df['Song'].apply(get_mfcc)
    return df

########## SPECTRAL CENTROID



########## SPECTRAL FLATNESS



########## SPECTRAL BANDWITH


########## SPECTRAL ROLLOFF


########## CHROMA VECTOR



########## MEL

def get_mfcc(song_name):

    genre = song_name.split(".")[0]
    filename = './genres/' + genre + '/' + song_name
    x,sample_rate = librosa.load(filename)
    mfccs = librosa.feature.mfcc(x, sr = sample_rate)
    return np.mean(mfccs, axis = 1)

def extract_mfcc(df):
    logger.debug('Extrayendo los MFCC...')

    df['MFCC'] = df['Song'].apply(get_mfcc)
    logger.debug('MFCC extraidos...')

    return df



###### PREPROCESADO



