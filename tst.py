# MLP's tesi
#import numpy       as np
import nnetwork    as nn
import data_param  as dpar

# Begin
def main():			
    Param       = dpar.load_config() 
    x,y    = dpar.load_data()
    W      = dpar.load_ws()
    zv     = nn.forward(x,W,Param)      		
    cm,Fsc = nn.metricas(zv[0][len(zv[0])-1],y) 		
    dpar.save_metric(cm,Fsc)
    print(Fsc*100)
    print('Fsc-mean {:.5f}'.format(Fsc.mean()*100))
	

if __name__ == '__main__':   
	 main()

