import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from statsmodels.tsa.stattools import coint, adfuller 
import statsmodels.api as sm
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)
def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return (df.values).T

pricesFile="./prices250.txt"
prcAll = loadPrices(pricesFile)
print ("Loaded %d instruments for %d days" % (nInst, nt))

my_array = np.ones((100,100), dtype="f") #dtype="f,f"
pair_list = []
i = 0
j = 0
for stock in prcAll:
    j = 0
    #outer for loop (testing stock 1 against all the others...)
    for iterate in prcAll:
        correlation = np.corrcoef(stock, iterate)
        if i != j: 
            #print("debug1")
            score, pvalue_granger, array = coint(stock, iterate)
            stock_expensive = stock
            stock_cheaper = iterate 

            stock_cheaper_constant = sm.add_constant(stock_cheaper, has_constant='add')
            results = sm.OLS(stock_expensive, stock_cheaper_constant).fit()
            b = results.params[1]
            theo_spread = results.params[0]
            spread = stock_expensive - b*stock_cheaper 
            pvalue_fuller = adfuller(spread)[1]
            my_pair_a = (i, j)
            my_pair_b = (j, i)
            print(my_pair_a)
            #print("debug2")
            if (pvalue_granger <= 0.01 and pvalue_fuller <= 0.01):
                    my_array[i,j] = pvalue_granger
                    if (my_pair_a not in pair_list) and (my_pair_b not in pair_list):
                        pair_list.append(my_pair_a)
                    
        #add the correlation and coint to numpy array
       # rounded = round(correlation[0,1],3)       
        j += 1   

    i+=1

#print(my_array)
print(pair_list)
#converting to pandas dataframe and dumping in txt file
np.savetxt("pvalue1.csv", my_array, delimiter=",")