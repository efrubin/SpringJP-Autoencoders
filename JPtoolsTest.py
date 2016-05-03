#!/usr/bin/env python

from tools import JPtools
import numpy as np
def makePsf(x, sigma1=1.0, b=0.0, sigma_ratio=2, xc=0):
    I = np.exp(-0.5*((x - xc)/sigma1)**2) + b*np.exp(-0.5*((x - xc)/(sigma_ratio*sigma1))**2)
    I /= np.sum(I)*(x[1] - x[0])
    
    return I
## put back x inputs 
def manyPsf(bList, samples=30):
    X = []
    for b in bList:
        x0 = np.zeros(samples)
        #x0[0:samples] = np.linspace(-1, 1, samples)
        #x0[samples:2*samples] = makePsf(np.linspace(-1,1,samples), b=b)
        x0[0:samples] = makePsf(np.linspace(-1,1,samples), b=b)
        #x0[-1] = b
        X.append(x0)
    return X


from sklearn.cross_validation import train_test_split
from scipy.special import expit
X = manyPsf(np.linspace(0, 10, 5000))
X = np.vstack(X)
X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)


jae = JPtools.JPAutoEncoder([30, 10, 5])
jae.pretrain(X_train, 10)
jae.fine_train(X_train, 10)
jae.predict(X_test)