import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape


pricesFile="./prices250.txt"
prcAll = loadPrices(pricesFile)
print(type(prcAll))
print ("Loaded %d instruments for %d days" % (nInst, nt))
#Running correlations 
my_array = np.zeros((100,100))
###print(my_array)s


i = 0
j = 0


for stock in prcAll:
    j = 0
    #outer for loop (testing stock 1 against all the others...)
    for iterate in prcAll:
        correlation = np.corrcoef(stock, iterate)
        #add the correlation to numpy array
        rounded = round(correlation[0,1],3)
        my_array[i,j] = rounded
        j += 1
        
    i+=1
    

print(my_array)  
#converting to pandas dataframe and dumping in txt file
np.savetxt("test.csv", my_array, delimiter=",")
