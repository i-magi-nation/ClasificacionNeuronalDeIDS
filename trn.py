# MLP's Trainig 
import numpy       as np
import nnetwork    as nn
import data_param  as dpar
import matplotlib.pyplot as plt

# Training by use miniBatch iSGD

def train_miniBatch(X,Y,W,V,Param):
    Costo=0
    Batch_size = Param[1]
    gW = []
    B = int(np.floor(len(X)/Batch_size))
   
    for i in range(B):
        
        xe,ye = X[Batch_size*i:(Batch_size*i)+Batch_size].T, Y[Batch_size*i:(Batch_size*i)+Batch_size].T
        
        Act = nn.forward(xe, W, Param)
        gW,C = nn.gradWs(Act,ye, W, Param)
        Costo += C
       
        W,V = nn.updWs(W, gW, V, Param, i, B)
        
    
    return W,Costo/B


# mlp's training 
def train_mlp(x,y,param):        
    W,V = nn.iniWs(x.shape[1], param)
    
    variable1_values = []
    variable2_values = []
        
    Costos = []               
    for Iter in range(1,param[0]+1):        
        xe,ye = nn.randpermute(x,y)
        
        W,Costo = train_miniBatch(xe,ye,W,V,param)
        Costos.append(Costo)
       
        if ((Iter %20)== 0):
            print('Iter={} Cost={:.5f}'.format(Iter,Costos[-1]))  
            variable1_values.append(Iter)
            variable2_values.append(Costo)
        
    # Grafica las variables
    plt.plot(variable2_values, label='Costos')
    
    # Agrega etiquetas y leyenda
    plt.xlabel('Iteración')
    plt.ylabel('Costo')
    plt.legend()
    
    # Muestra la gráfica
    plt.show()
    return W, Costos

# Beginning ...
def main():
    param       = dpar.load_config()        
    x,y         = dpar.load_dtrain()   
    W,costo     = train_mlp(x,y,param)         
    dpar.save_ws_costo(W,costo)
       
if __name__ == '__main__':   
	 main()

