Modelo para la clasificacion nuronal de IDS (Sistemma de deteccion de intrusos)
usando el algoritmo descenso del gradiente estocastico mejorado.
El algoritmo funciona para redes neuronales de 1 o 2 capas ocultas.
Con la configuracion actual entrega un accuracy del 92% (no es el mejor pero al menos pasa el 90 xd)
==================================Archivos.csv======================================
config.csv: 
contiene la configuracion para el algoritmo donde cada casilla (fila) corresponde a
1 = numero de epocas
2 = tamaño del batch
3 = numero de capas ocultas (especifica si se usa 1 capa oculta o 2) 
4 = nodos capa oculta 1
5 = nodos capa oculta 2 (en caso de usar solo una capa oculta otorgar valor de 0)
6 = tipo de activacion (1° sigmoide; 2° Tanh, 3° RELU, 4° ELU, 5° SELU)
7 = tasa de aprendizaje

xtrain.csv / ytrain.csv:
contiene datos para el entrenamiento

xtest.csv / ytest.csv:
contiene datos para el testeo

cmatrix.csv:
matriz de confusion resultante

costo_avg.csv:
costo promedio

fscores.csv:
metrica de desempeño

==================================Archivos.py======================================
data_param.py:
carga los datos

nnetwork.py:
funciones para la red neuronal

trn.py:
funciones para ejecutar el training

tst.py:
funciones para ejecutar el testing
