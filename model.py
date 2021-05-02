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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense,LSTM,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU, BatchNormalization,MaxPooling2D,GlobalAveragePooling2D


from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow
import datetime


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

GENERAMOS LA CLASE DEL MODELO

"""

class Model():

    def __init__(self,model_name,df,epochs,verbose=1,patience=50):
        self.model_name=model_name
        self._model=None
        self._X_train=None
        self._X_validation=None
        self._X_test=None
        self._y_train=None
        self._y_validation=None
        self._y_test=None
        self._df=df
        self._epochs=epochs
        self.verbose=verbose
        self.patience=patience


    
    def preprocess_DENSE(self):
        logger.debug('Preprocesando los datos para la DENSE..')

        encoder = LabelEncoder()
        labels = encoder.fit_transform(self._df.iloc[:, 1])
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(self._df.iloc[:, 2:], dtype = float))
        
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20)
        

        ## Pasamos a categórico con el número de clases posibles

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        y_validation = to_categorical(y_validation, 10)
    
        ### Cargamos los atributos dentro de nuestro modelo

        self._X_train=X_train
        self._X_validation=X_validation
        self._X_test=X_test
        self._y_train=y_train
        self._y_validation=y_validation
        self._y_test=y_test
        logger.debug('Datos preprocesados..')

    def preprocess_LSTM(self):
        logger.debug('Preprocesando los datos para la LSTM..')
        ## Cargamos los datos del df
        encoder = LabelEncoder()
        labels = encoder.fit_transform(self._df.iloc[:, 1])
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(self._df.iloc[:, 2:], dtype = float))

        ### Separamos en train,validation y test

        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20)
        
        X_train=X_train.reshape((600, 20,1))
        X_test=X_test.reshape((250, 20,1))
        X_validation=X_validation.reshape((150, 20,1))
        ## Pasamos a categórico con el número de clases posibles

        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        y_validation = to_categorical(y_validation, 10)
    
        ### Cargamos los atributos dentro de nuestro modelo

        self._X_train=X_train
        self._X_validation=X_validation
        self._X_test=X_test
        self._y_train=y_train
        self._y_validation=y_validation
        self._y_test=y_test
        logger.debug('Datos preprocesados..')

    def preprocess_CNN(self):
        logger.debug('Comenzamos a preprocesar los datos para la CNN...')
        max_pad_len = 1320

        def get_mfcc_CNN(song_name):
            genre = song_name.split(".")[0]
            file_name = './genres/' + genre + '/' + song_name
            
            try:
                audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
                pad_width = max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
                
            except Exception as e:
                print("Error encountered while parsing file: ", file_name)
                return None 
            
            return mfccs
        features = []
        
        for i in range(len(self._df.Song)):
                
            class_label = self._df.Genre[i] 
            data = get_mfcc_CNN(self._df.Song[i])
            
            features.append([data, class_label])

        featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

        ## Lo pasamos a vector

        X = np.array(featuresdf.feature.tolist(), dtype = "object")
        y = np.array(featuresdf.class_label.tolist(), dtype = "object")

        # Encoding
        encoder = LabelEncoder()
        labels = to_categorical(encoder.fit_transform(y)) 

        # Splitting
        from sklearn.model_selection import train_test_split 

        x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size = 0.2, random_state = 42)
        x_train = np.asarray(x_train).astype(np.float32)
        x_test = np.asarray(x_test).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)

        ### hacemos un reshape

        num_rows = x_train.shape[1] 
        num_columns = 1320 #frames (ver get_mfcc_CNN)
        num_channels = 1 

        x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
        x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

        
        self._X_train=x_train
        self._X_validation=x_test
        self._X_test=x_test
        self._y_train=y_train
        self._y_validation=y_test
        self._y_test=y_test
        logger.debug('Datos preprocesados para la CNN..')

    def _model_lstm(self):
        logger.debug('Generamos el modelo LSTM...')
        
        input_shape = (self._X_train.shape[1], self._X_train.shape[2])

        model = Sequential()

        # 2 LSTM layers
        model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(0.5))
        #model.add(LSTM(64))

        # dense layer
        model.add(Dense(64, activation='elu'))
        model.add(Dropout(0.3))

        # output layer
        model.add(Dense(10, activation='softmax'))
        plot_model(model)
        logger.debug('LSTM Generado...')
     
        return model

    def _model_cnn(self):
        
        logger.debug('CNN ...')

        # Construct model 
        model2 = Sequential()

        

        model2.add(Conv2D(filters = 16, kernel_size = 2, input_shape = (self._X_train.shape[1], 1320, 1)))
        model2.add(BatchNormalization())
        model2.add(LeakyReLU(alpha = 0.01))
        model2.add(MaxPooling2D(pool_size = 2))
        model2.add(Dropout(0.5))

        

        model2.add(Conv2D(filters = 32, kernel_size = 2))
        model2.add(BatchNormalization())
        model2.add(LeakyReLU(alpha = 0.01))
        model2.add(MaxPooling2D(pool_size = 2))
        model2.add(Dropout(0.4))

        

        model2.add(Conv2D(filters = 64, kernel_size = 2))
        model2.add(BatchNormalization())
        model2.add(LeakyReLU(alpha = 0.01))
        model2.add(MaxPooling2D(pool_size = 2))
        model2.add(Dropout(0.3))

        

        model2.add(Conv2D(filters = 128, kernel_size = 2))
        model2.add(BatchNormalization())
        model2.add(LeakyReLU(alpha = 0.01))
        model2.add(MaxPooling2D(pool_size = 2))
        model2.add(Dropout(0.3))

        


        model2.add(Flatten())
        model2.add(Dropout(0.4))

        

        model2.add(Dense(10, activation = 'softmax')) 
        plot_model(model2,to_file='CNN.png')

        logger.debug('CNN Generado...')

        return model2

    def _model_dense(self):
        logger.debug('Inicializando el modelo dense...')
        model = Sequential()

        model.add(Dense(256, input_shape=(self._X_train.shape[1],)))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha = 0.01))
        model.add(Dropout(0.5))


        model.add(Dense(128))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha = 0.01))
        model.add(Dropout(0.3))

        model.add(Dense(64))
        model.add(BatchNormalization()) 
        model.add(LeakyReLU(alpha = 0.01))
        model.add(Flatten())
        model.add(Dropout(0.3))

        model.add(Dense(10,activation='softmax'))
        logger.debug('modelo dense Inicializado...')
        return model

    def _model_lstm_cnn(self):
        logger.debug('Generamos el modelo LSTM...')
        
        input_shape = (self._X_train.shape[1], self._X_train.shape[2])

        model = Sequential()

        # 2 LSTM layers
        model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
        model.add(Dropout(0.5))
        #model.add(LSTM(64))

        # dense layer
        model.add(Dense(64, activation='elu'))
        model.add(Dropout(0.3))

        # output layer
        model.add(Dense(10, activation='softmax'))
        plot_model(model)
        logger.debug('LSTM Generado...')
     
        return model


    def train(self):

        logger.debug('Iniciando el modelo...')

        ###Elige entre los modelos posibles

        if(self.model_name=='LSTM'):
            model_train=self._model_lstm()
        elif(self.model_name=='CNN'):
            model_train=self._model_cnn()
        elif(self.model_name=='DENSE'):
            model_train=self._model_dense()

        
        optimiser = Adam(learning_rate=0.0001)
        model_train.compile(optimizer=optimiser,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        # To launch TensorBoard type the following in a Terminal window: tensorboard --logdir /path/to/log/folder
        logdir = os.path.join("logs_{}".format(self.model_name), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
            logdir, histogram_freq=0,
            write_graph=True, write_grads=False,
            write_images=False, embeddings_freq=0,
            embeddings_layer_names=None, embeddings_metadata=None,
            embeddings_data=None, update_freq='epoch'
        )

        ## AÑADIMOS DOS CALLBACKS
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=self.patience)
        mc = ModelCheckpoint('best_model_{}.h5'.format(self.model_name), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        callbacks = [tensorboard_callback,es,mc]
        history = model_train.fit(self._X_train, self._y_train, batch_size=64, epochs=self._epochs,verbose=self.verbose,validation_data=(self._X_validation,self._y_validation),callbacks=callbacks)


        logger.debug('Modelo entrenado...')
        # Plot model training history
        if self._epochs > 1:
            self._plot_training(history)

    @staticmethod
    def _plot_training(history):
        """Plots the evolution of the accuracy and the loss of both the training and validation sets.

        Args:
            history: Training history.

        """
        training_accuracy = history.history['accuracy']
        validation_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(training_accuracy))

        # Accuracy
        plt.figure()
        plt.plot(epochs, training_accuracy, 'r', label='Training accuracy')
        plt.plot(epochs, validation_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()
