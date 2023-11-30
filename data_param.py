# Data and Parameters
import numpy  as np
import pandas as pd 
#cargar parametros
def load_config(ruta_archivo='config.csv'):

    with open(ruta_archivo, 'r') as archivo_csv:

        conf = [int(i) if '.' not in i else float(i)
                for i in archivo_csv if i != '\n']

    return conf
#data para el training
def load_dtrain(path_csv_x='xtrain.csv',path_csv_y='ytrain.csv'):

    x = np.genfromtxt(path_csv_x, delimiter=',').T
    y = np.genfromtxt(path_csv_y, delimiter=',').T

    return x , y
def load_data(path_csv_x='xtest.csv',path_csv_y='ytest.csv'):
    
    x = np.genfromtxt(path_csv_x, delimiter=',')
    y = np.genfromtxt(path_csv_y, delimiter=',')
         
    return x , y
#se guarda peso y costo
def save_ws_costo(W,Costos):
    np.savez('Ws.npz', *W)

    df = pd.DataFrame( Costos )
    df.to_csv('costo_avg.csv',index=False, header = False )

    return

#se carga peso pre entrenado
def load_ws():

    ws = np.load('Ws.npz')

    ws = [ws[i] for i in ws.files]

    return ws

#save metrics
def save_metric(cm,Fsc):

    cm = pd.DataFrame( cm )
    cm.to_csv('cmatrix.csv',index=False, header = False )
    
    Fsc_array = np.array([Fsc])
    Fsc_df = pd.DataFrame( Fsc_array )
    Fsc_df.to_csv('fscores.csv',index=False, header = False )

    return