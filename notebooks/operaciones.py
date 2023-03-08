###
from numpy import array
from keras.models import Sequential
# from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import csv
import numpy as np
import matplotlib.pyplot as plt
import bqplot.pyplot as bqplt
from sklearn.model_selection import train_test_split
from tensorflow import keras

"""Documentación del módulo.
Esta es una anotación la cual debe de encontrarse
en la parte superior de nuestro script.
Esta anotación tiene cómo objetivo describir el módulo"""

__author__ = "Alberto Reyes "
__copyright__ = "Copyright 2023, INEEL"
__credits__ = ["GCEC", "DTH", "GGIP"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "ARB"
__email__ = "areyes@ineel.mx"
__status__ = "Preparación"

class operaciones :
    """Clase que contiene los métodos para el pronóstico de procesos basado en deep learning"""

    # def __init__(self):
    #     print('arranca clase')

    def split_sequence(self, sequence, n_steps_in, n_steps_out):
        """Split a univariate sequence into samples
           Args:
            sequence -- parámetro (list)
            n_steps_in -- parámetro (int)
            n_steps_out -- parámetro (int)
           Returns:
            Tuple[ndarray, ndarray]
        """
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    # Demuestra el uso de split_sequence() con el conjunto de datos de ejemplo.
    # def ejemplo():
    #     raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    #     n_steps_in, n_steps_out = 3, 2
    #     # dividimos en ejemplos X , y
    #     X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
    #     # visualizamos los resultados
    #     for i in range(len(X)):
    #         print(X[i], y[i])
    #     print('Antes', X)
    #     # remodela de [muestras, timesteps] a [muestras, timesteps, características]
    #     n_features = 1
    #     X = X.reshape((X.shape[0], X.shape[1], n_features))
    #     print('Después', X)

    def dataFile2rawSeq(self, data_path):
        """
        Args:
            data_path (str)
        Returns:
            Secuencia cruda (ndarray)
  """
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            rs = np.array(list(reader)).astype(float)
            rs = [row[0] for row in rs]
        return rs

    def grafica_serie(self, secuencia,etiquetaEjeY) :
        """ Método que grafica una serie de tiempo
        Args:
            secuencia -- Secuencia de datos (list)
            etiquetaEjeY -- Etiqueta de las ordenadas (str)
  """
        bqplt.figure(title='Serie de tiempo ')
        bqplt.plot(secuencia)
        bqplt.xlabel('tiempo de registro')
        bqplt.ylabel(etiquetaEjeY)
        bqplt.show()

    def formatea_serie(self, raw_seq,n_steps_in,n_steps_out):
        """Método que formatea una serie de tiempo
        Args:
            raw_seq -- Secuencia de datos (list)
            n_steps_in -- Long de secuencia de entrada (int)
            n_steps_out -- Long de secuencia de salida (int)
        Returns:
            X -- secuencia de entrada formateada (ndarray)
            y -- secuencia de salida formateada (ndarray)
        """
        # escoger el número de pasos de timpo (time steps)
        #n_steps_in, n_steps_out = 7, 7
        # separar en muestras (preparación)
        X,y = operaciones.split_sequence(self, raw_seq, n_steps_in, n_steps_out)
        # remodelación del formato [muestras, timesteps] a [muestras, timesteps, características]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        return X,y

    def separa_ejemplos(self, X,y,num_test):
        """Separacion de ejemplos en subconjuntos de entrenamiento y prueba.
        Args:
            X -- Conjunto de datos de entrada formateada (ndarray)
            y -- Conjunto de datos de salida formateada (ndarray)
            num_test -- Fracción de datos de prueba (float)
        Returns: Tuple X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=num_test)

    def genera_modelo(self, n_steps_in,n_steps_out,n_features,X_train, y_train):
        """Generación del modelo.
        Args:
            n_steps_in -- Long de secuencia de entrada (int)
            n_steps_out -- Long de secuencia de salida (int)
            n_features -- Número de características (int)
            X_train -- Datos de entrenamiento - entradas (list)
            y_train -- Datos de entrenamiento - salidas (list)
        Returns: Modelo de pronóstico (Sequential)
        """
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
        model.add(LSTM(100, activation='relu'))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mse')
        # fit model
        # model.fit(X_train, y_train, epochs=1050, verbose=0)
        model.fit(X_train, y_train, epochs=1050)
        #model.save(data_path)
        return model

    def guardar_modelo(self, modelo,data_path):
        """Archivado del modelo.
            Args:
                modelo -- Modelo de pronóstico (Sequential)
                data_path -- ruta a directorio (str)
        """
        modelo.save(data_path)

    def recupera_modelo(self, data_path):
        """Recuperación del modelo.
            Args:
                data_path -- ruta a directorio (str)
            Returns: Modelo de pronóstico (Sequential)
        """
        modelo=keras.models.load_model(data_path)
        return modelo

    def ver_modelo(self, model,data_path):
        """Guarda y visualiza imagen con estructura del modelo en formato .png.
        Args:
            model -- Modelo de pronóstico (Sequential)
            data_path -- ruta a directorio (str)
        """
        plot_model(model, to_file=data_path, show_shapes=True, show_layer_names=True)

    def demo_prediccion(self, X_test,y_test,n_steps_in,n_features,model,etiquetaEjeY):
        """Demuestra la predicción.
            Args:
                data_path -- ruta a directorio (str)
                X_test -- Conjunto de datos de prueba entradas (List)
                y_test -- Conjunto de datos de prueba salidas (List)
                n_steps_in -- Long de secuencia de entrada (int)
                n_features -- Número de características (int)
                model -- Modelo de pronóstico (Sequential)
                etiquetaEjeY -- Etiqueta de las ordenadas (str)
        """
        # demonstrate prediction
        l = X_test.reshape((X_test.shape[0], X_test.shape[1]))
        i_sample = np.random.randint(len(y_test))  # Tomar un ejemplo aleatorio
        x_sample = l[i_sample]

        pred=operaciones.predice_secuencia(self, x_sample,model,n_steps_in,n_features)

        print('secuencia de entrada:', x_sample)
        print('secuencia de salida (predicha):', pred)
        print('secuencia de salida (real):', y_test[i_sample])

        operaciones.grafica_prediccion(self, x_sample, y_test[i_sample], pred, etiquetaEjeY)


    def predice_secuencia(self, x_sample,model,n_steps_in, n_features):
        """Predice una secuencia de entrada.
            Args:
                x_sample -- Secuencia de entrada muestra (List)
                model -- Modelo de pronóstico (Sequential)
                n_steps_in -- Long de secuencia de entrada (int)
                n_features -- Número de características (int)
            Returns: y -- secuencia de pronóstico (List)
        """
        x_sample = x_sample.reshape((1, n_steps_in, n_features))
        return model.predict(x_sample, verbose=0)


    def grafica_prediccion(self, x_sample, y_real,pred,etiquetaEjeY):
        trama_real = np.append(x_sample, y_real)
        trama_predicha = np.append(x_sample, pred)
        bqplt.figure(title='Pronóstico')
        bqplt.xlabel('tiempo de registro')
        bqplt.ylabel(etiquetaEjeY)
       # bqplt.plot(trama_predicha, color='green', linestyle='dashed')
        bqplt.plot(trama_predicha, linestyle='dashed')
        bqplt.plot(trama_real)
        bqplt.show()

    # Definimos una función simple para calcular el MAPE
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Guardamos todos los vectores de salida pronosticados para posteriormente compararlos con los vectores de salida reales.

    #import numpy as np
    def lista_predicciones_tplusn(self, X_test,n_steps_in, n_features,model):
        """Lista de predicciones hasta el tiempo t + n.
            Args:
                X_test -- Conjunto de datos de prueba (List)
                n_steps_in -- Long de secuencia de entrada (int)
                n_features -- Número de características (int)
                model -- Modelo de pronóstico (Sequential)
            Returns: y -- secuencia de pronóstico (ndarray)
        """
        l = X_test.reshape((X_test.shape[0], X_test.shape[1]))

        lista_tplusn = array([])
        for i in l:
            # i = i.reshape((1, n_steps_in, n_features))
            # yhat = model.predict(i, verbose=0)

            yhat = operaciones.predice_secuencia(self, i,model,n_steps_in, n_features)

            lista_tplusn = np.append(lista_tplusn, yhat[0, :])

        lista_tplusn = np.reshape(lista_tplusn, (len(l), -1))
        return lista_tplusn

    #print(lista_tplusn)

    def muestra_error(self, n_steps_out,lista_tplusn,y_test,etiquetaEjeY):
        """Muestra error MAPE.
            Args:
                n_steps_out -- Long de secuencia de salida (int)
                lista_tplusn -- secuencia de pronóstico (ndarray)
                y_test --
                etiquetaEjeY --
        """
        for i in range(n_steps_out):
            mape = operaciones.mean_absolute_percentage_error(lista_tplusn[:, i], y_test[:, i])
            print(f'MAPE tiempo t+{i + 1}: {mape:.3}%')
            t = i + 1
            bqplt.figure(title='Error MAPE al tiempo: t+%i' % t)
            bqplt.plot(lista_tplusn[:, i])
            bqplt.plot(y_test[:, i])
            bqplt.xlabel('tiempo de registro')
            bqplt.ylabel(etiquetaEjeY)
            # plt.annotate(f'MAPE tiempo t+{i + 1}: {mape:.3}%',10,10)
            #bqplt.annotate(mape, (10, 10))
            bqplt.show()