# Audio_Classification
Audio_Classification
ESTRUCTURA DEL PROYECTO:
Nuestro objetivo con este proyecto era tratar de generar varios tipos de modelos para probar las distintas estructuras de cara a la predicción de audio. Las limitaciones con las que nos encontramos son claras dado que contamos con un dataset bastante pequeño y aún más si tenemos en cuenta que se trata de 10 clases distintas.
Por otro lado, hemos tratado de generar un código que pudiese ser utilizado en un entorno profesional productivo, es por ello que hemos generado una clase amplia que contiene todo lo concerniente al modelo y además se han añadido gran cantidad de logs para luego poder depurar el código en caso de correr el programa en un despliegue fuera del local de nuestro ordenador.
Por ello, el trabajo consiste de cuatro documentos:


    1) Introducción al proyecto: En este notebook lo que se hace referencia es a las distintas funciones de preprocesado disponibles para su utilización
    2) preprocess_data.py: Este documento contiene las distitnas funciones de preprocesado de forma que luego podamos hacer referencias a ellas para a partir de los audios extraer su espectograma, MFCC, etc..
    3) model.py: Aquí contiene todo lo relativo a los distintos modelos que hemos probado (DENSE, CNN, LSTM)
    4) Audio_final: Por último, en este notebook tenemos la llamada a cada una de los modelos embebidos en el archivo model.py

Es importante entender igualmente los archivos que se generan tras la ejecución del notebook Audio_final.ipynb. Por un lado, model.log y preprocess_data.log hacen referencia a los logs que se generan tras la ejecución de dichos ficheros. Igualmente, se generan varios directorios como pueden ser logs_CNN que luego permitirán dar servicio a la interfaz de tensorboard. Por último, en model.png y en CNN.png tenemos una representación gráfica de nuestros modelos. Para poder la interfaz de tensorboard es necesario correr el código completo.
Por último, mencionar que en la última parte se ha dejado planteada la estructura de una CNN combinada con una LSTM pero por falta de tiempo no hemos podido llegar a implementarla.
