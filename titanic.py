# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:54:38 2020

@author: 815364
"""



import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import OneHotEncoder
# =============================================================================
# Complete the code for linear regression
# 1a, 1b, 2, 3, 4
# Code should be vectorized when complete
# =============================================================================

def createData():
    fileName = 'Titanic_Train.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    #loadtxt defaults to floats

    data = np.loadtxt(raw_data, usecols = (2,5,6,7,8,10,1), delimiter=",", skiprows=1 , dtype=np.str )
#1a
   
    rows,cols = data.shape
    X = data[:,0:-1]
    Y = data[:,-1]
    Y = Y.astype(np.float)
    X = processData(X)
    sub = X[:,3:]
    meanX = sub.mean(axis=0)
    rangeX = sub.max(axis=0)-sub.min(axis=0)
    X[:,3:] = (sub-meanX)/rangeX
    X = np.hstack((np.ones((rows, 1)),X))   
    return X,Y

def processData(X):
    X[:, 1][X[:, 1] == 'female'] = 1
    X[:, 1][X[:, 1] == 'male'] = 0 
    X[:, 2][X[:,2] == ''] = np.nan
    X[:, 5][X[:,5] == ''] = np.nan
    X = X.astype(np.float)
    
    nonnan = np.nanmean(X[:,2])
    X[:,2][np.isnan(X[:,2])] = nonnan

    
    nonnan = np.nanmean(X[:,5])
    X[:,5][np.isnan(X[:,5])] = nonnan
    
    
    classCol = X[:,0:1]
    ohe = OneHotEncoder(categories = 'auto')
    classCols = ohe.fit_transform(classCol).toarray().astype(np.float)
    X = np.column_stack((classCols , X[: , 1:]))
    return X


# ===================================
#2
def calcCost(X,W,Y):
    p = sigmoid(np.dot(X,W))
    J = Y * np.log(p) + (1-Y)*np.log(1-p)
    J = -J.mean()
    return J
#4
def sigmoid(Z):
    return 1/(1+np.exp(-Z))
def calcGradient(X,Y,W):
    predmat = np.dot(X,W)
    predmat=sigmoid(predmat)
    predymat = predmat-Y
    gradsum = np.dot(predymat,X)
    return np.array(gradsum/len(X))
    
def standardize(X):
  #use the equation given to standardize the matrix X
  Xbar = np.mean(X,axis=0)
  stdev = np.std(X,axis=0)
  
  return (X-Xbar)/stdev, Xbar, stdev
def meanNormalize(X):
  #use the equation given to meanNormalize the matrix X
  mean = np.mean(X, axis=0)
  colmax = np.max(X, axis=0)
  colmin = np.min(X, axis=0)
  return (X-mean)/(colmax-colmin)

############################################################
# Create figure objects
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])# [left, bottom, width, height]

#1b
# =============================================================================
#  X,Y use createData method to create the X,Y matrices
# Weights - Create initial weight matrix to have the same weights as features
# Weights - should be set to 0
# =============================================================================
# 

X,Y = createData()

numRows,numCols = X.shape
W=np.zeros(numCols)





# set learning rate - the list is if we want to try multiple LR's
# We are only doing one of them today
lrList = [.2,.01]
lr = lrList[0]

#set up the cost array for graphing
costArray = []
costArray.append(calcCost(X, W, Y))

#initalize while loop flags
finished = False
count =0
minVectorDiff = .0001
while (not finished and count <20000):
    gradient = calcGradient(X,Y,W)
    
    #5 update weights
    W=W-lr*gradient
    costArray.append(calcCost(X, W, Y))
    lengthOfGradientVector = np.linalg.norm(gradient)
    if (lengthOfGradientVector < minVectorDiff): 
        finished=True
    count+=1
        
    


print(count)
ax.plot(np.arange(len(costArray)), costArray, "ro", label = "cost")


ax.set_title("Cost as weights are changed")
ax.set_xlabel("iteration")
ax.set_ylabel("Costs")
ax.legend()          

# probability
testcertainty = sigmoid(np.dot(X,W))      
threshold = 0.5
neg = tuple([Y[:]== 0])
pos = tuple([Y[:]== 1])

tn = np.count_nonzero([testcertainty[neg] < threshold])
fn = np.count_nonzero([testcertainty[pos] < threshold])

tp = np.count_nonzero([testcertainty[pos] >= threshold])
fp = np.count_nonzero([testcertainty[neg] >= threshold])


print("True Positives: "+ str(tp)) 
print("Percent Positives correct: "+ str(tp/(tp+fp)*100) + "%") 
print("False Positives: "+ str(fp))
print("Percent Positives Incorrect: "+ str(fp/(tp+fp)*100) +"%")
print("True Negatives: "+ str(tn))
print("Percent Negatives correct: "+ str(tn/(tn+fn)*100) +"%")
print("False Negatives: "+ str(fn))
print("Percent Negatives incorrect: "+ str(fn/(tn+fn)*100) +"%")

print("Precision: " + str(tp / (tp+fp)*100)+"%")
print("Recall: " + str(tp / (tp+fn)*100)+"%")

print("F1: " + str(tp/(tp+0.5*(fp+fn))*100)+"%")
