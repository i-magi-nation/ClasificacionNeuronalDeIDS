# Neural Network: functions
import numpy  as np
    
#Inicializar pesos
def iniWs(inshape, Param):
    #1°ra capa oculta
    W1 = randW(Param[3], inshape)
    
    if Param[2] == 2:
        #2°da capa oculta
        W2 = randW(Param[4], Param[3])
        #3°capa salida
        W3 = randW(2, Param[4])
        W = list((W1, W2, W3)) 
    else:
        #capa salida
        W3 = randW(2, Param[3])
        W = list((W1,W3))
        
    V = []
    for i in range(len(W)):
        V.append(np.zeros(W[i].shape))

    return W, V

# Rand values for W    
def randW(next,prev): 
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# Random location for data
def randpermute(X, Y): 
    
    indices_filas = np.arange(X.shape[0])
    
    np.random.shuffle(indices_filas)
    
    
    X = X[indices_filas, :]
    Y = Y[indices_filas, :]

    
    return X , Y

#Activation function 
def act_functions(x, act):
    # Sigmoid

    if act == 1:
        return 1 / (1 + np.exp(-1*x))

    # tanh

    if act == 2:
        
        return np.tanh(x)


    # Relu

    if act == 3:
        
        return np.maximum(0, x)

    # ELU

    if act == 4:

        return np.where(x > 0, x, 0.01 * (np.exp(x) - 1))

    # SELU

    if act == 5:
          
        return np.where(x <= 0, 1.0507 * (1.6732 * (np.exp(x) - 1)), 1.0507 * x)
    

    return x

#Derivadas de las funciones de activacion 
def deriva_act(x, act):
    # Derivada de Sigmoid
    if act == 1:
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * (1 - sigmoid)

    # Derivada de tanh
    if act == 2:
        return 1 - np.tanh(x)**2

    # Derivada de Relu
    if act == 3:
        return np.where(x > 0, 1, 0)

    # Derivada de ELU
    if act == 4:
        alpha = 0.01
        return np.where(x > 0, 1, alpha * np.exp(x))

    # Derivada de SELU
    if act == 5:
        lambda_ = 1.0507
        alpha = 1.6732
        return np.where(x > 0, lambda_, lambda_ * alpha * np.exp(x))

    # Para cualquier otro caso, asumimos que la derivada es 1
    return 1


#Feed-forward 
def forward(X, W, Param):   #BIEN
     
    # cambiar activaciones por config
    act_encoder = Param[5]

    A = []
    z = []
    Act = []
   
    # data input
    z.append(X)
    A.append(X)

    # iter por la cantidad de pesos
    for i in range(len(W)): 
       
        X = np.dot(W[i], X)
        z.append(X)
        if i == len(W)-1: 
            X = act_functions(X, act=1)
        else:
            X= act_functions(X, act_encoder)

        A.append(X)

    Act.append(A)
    Act.append(z)

    return Act
# Feed-Backward 
def gradWs(Act,Y, W, Param):
    
    act_encoder = Param[5]
    L = len(Act[0])-1
    N = Param[1]
    e = Act[0][L] - Y
    #Cost = np.sum(np.sum(np.square(e), axis=0)/2)/N
    Cost = (1 / (2 * N)) * np.sum((e) ** 2)
    #Cost = (1 / (2 * N)) * np.sum(np.square(e), axis=0)
    #gradiente del error inicial
    #######
    delta = np.multiply(e, deriva_act(Act[0][L], act=1))

    gW_l = np.dot(delta, Act[0][L-1].T)/N
    gW = []
    gW.append(gW_l)

    # grad para pesos ocultos
    for l in reversed(range(1,L)):
        
        g_capa_oculta = np.dot(W[l].T, delta)
        
        #derivada capa oculta
        d_capa_oculta = deriva_act(Act[1][l], act_encoder)
        
        #gradiente del error para las capas siguientes
        delta = np.multiply(g_capa_oculta, d_capa_oculta)
        
        #Se usa la traspuesta de las activaciones de la capa anterior a la actual 
        gradiente = Act[0][l-1].T 

        gW_l = np.dot(delta, gradiente)
        gW.append(gW_l)
        
    #Se invierte la lista de gradientes para que coincida con el orden de las capas    
    gW.reverse()
    
    return gW, Cost        

# Update MLP's weigth using iSGD
def updWs(W, gW, V, Param, ite, T):
    
    u = Param[6]

    t = 1-(ite/T)
    beta = (0.9*t)/(0.1+(0.9*t))

    for i in range(len(W)):
        V[i] = (beta * V[i]) - (u*gW[i])
        W[i] = W[i] + V[i]

    return W, V

# Measure
def metricas(x,y):
    cm     = confusion_matrix(x,y)
    
    TP = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TN = cm[1,1]
        
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    Fsc = (( 2 * Precision * Recall ) / ( Precision + Recall ))
     
    return cm , Fsc

    
#Confusion matrix
def confusion_matrix(z, y):
    y,z = y.T,z.T
    
    m= y.shape[0]
    c = y.shape[1]
    
    y = np.argmax(y, axis=1)
    z = np.argmax(z, axis=1)
   
    cm = np.zeros((c,c))
    
    for i in range(m):
         cm[z[i] ,y[i]] +=1
    
    return cm

#

