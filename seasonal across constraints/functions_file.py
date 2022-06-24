#%% Libraries 

import os.path
import pypsa
import math
import numpy as np
from numpy.linalg import eig
import pandas as pd

import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.ticker as tick
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

#%% SavePlot

def SavePlot(fig,path,title):
    """

    Parameters
    ----------
    path : string
        folder path in which the figures needs to be saved in. This needs to be created before
    title: string
        Name of the figure, this is also what the figures is saved as. This does not need to be created

    Returns
    -------
    Nothing

    """
    
    # Check if path exist
    assert os.path.exists(path), "Path does not exist"
    
    # Check if figure is already existing there
    fig.savefig(path+title+".png", bbox_inches='tight')
    
    return fig
    

#%% PCA
def PCA(X):
    """
    Input:
        - X: Matrix for performing PCA on
    
    Output:
        - eigen_values:         Eigen values for the input
        - eigen_vectors:        Eigen vectors for the input
        - variance_explained:   How much of the variance is explained by each principal component
        - norm_const:           Normalization constant
        - T:                    Principle component amplitudes (a_k from report)
        
    """
    # Average value of input matrix
    X_avg = np.mean(X, axis=0)
    
    # Mean center data (subtract the average from original data) 
    B = X.values - X_avg.values
    
    # Normalisation constant
    normConst = (1 / (np.sqrt( np.sum( np.mean( ( (B)**2 ), axis=0 ) ) ) ) )
        
    # Covariance matrix (A measure of how much each of the dimensions varies from the mean with respect to each other)
    C = np.cov((B*normConst).T, bias=True)
    
    # Stops if C is larger than [30 x 30] 
    assert np.size(C) <= 900, "C is too big"
    
    # Eigen vector and values
    eigenValues, eigenVectors = eig(C)
        
    # Variance described by each eigen_value
    varianceExplained = (eigenValues * 100 ) / eigenValues.sum()
    
    # Principle component amplitudes
    T = np.dot((B*normConst), eigenVectors)
         
    # Cumulative variance explained
    #variance_explained_cumulative = np.cumsum(variance_explained)
    
    return (eigenValues, eigenVectors, varianceExplained, normConst, T)

#%% AlignEigenVectors

def AlignEigenVectors(EigenVector, dataNames, country="ES"):
    
    # Correction
    beforeCorrection = EigenVector
    correction = np.zeros(len(EigenVector))
    for i in range(len(beforeCorrection)):
        if beforeCorrection[list(dataNames).index(country)][i] > 0: # 9 = Spain
            correction[i] = 1
        else:
            correction[i] = -1
    eigenVectorsNew = beforeCorrection * correction
    
    return(eigenVectorsNew, correction)

#%% Coherence

def Coherence(mismatch, electricityNodalPrices, decimals=3):
    """
    Parameters
    ----------
    mismatch : matrix [8760 x 30] floating point
        mismatch between generation and load for each country [MWh].
        
    electricityNodalPrices : matrix [8760 x 30] floating point
        prices of electricity for each country [€/MWh]

    Returns
    -------
    c1 : matrix [30 x 30]
        Coherence method 1 - orthogonality between eigen vectors.
        
    c2 : matrix [30 x 30]
        Coherence method 2 - weighed (eigen values) orthogonality between eigen vectors.
        
    c3 : matrix [30 x 30]
        Coherence method 3 - mean time-varying (principle component amplitudes) orthognoality between eigen vectors.

    """
    
    # Stops if mismatch have the wrong format
    #assert mismatch.shape == (8760,30), "'mismatch' matrix has wrong dimensions. It must be [8760 x 30] !"
    
    # Stops if electricity prices have the wrong format
    #assert electricityNodalPrices.shape == (8760,30), "'electricityNodalPrices' matrix has wrong dimensions. It must be [8760 x 30] !"
    
    # PCA for mismatch
    eigenValuesMismatch, eigenVectorsMismatch, varianceExplainedMismatch, normConstMismatch, akMismatch = PCA(mismatch)
    
    # Correction
    eigenVectorsMismatch, correction = AlignEigenVectors(eigenVectorsMismatch, mismatch.columns.str.slice(0,2).unique(), country="ES")
    akMismatch = akMismatch*correction # Timeseries
    
    # PCA for electricity nodal prices
    eigenValuesPrice, eigenVectorsPrice, varianceExplainedPrice, normConstPrice, akPrice = PCA(electricityNodalPrices)

    # Correction
    eigenVectorsPrice, correction = AlignEigenVectors(eigenVectorsPrice, electricityNodalPrices.columns.str.slice(0,2).unique(), country="ES")
    akPrice = akPrice*correction # Timeseries

    matrixSize = np.zeros(eigenVectorsMismatch.shape)
    c1 = np.zeros(matrixSize.shape)
    c2 = np.zeros(matrixSize.shape)
    c3 = np.zeros(matrixSize.shape)
    
    for n in range(matrixSize.shape[0]):
        for k in range(matrixSize.shape[1]):
            
            # Eigen vectors
            p_k1 = eigenVectorsMismatch[:,n]
            p_k2 = eigenVectorsPrice[:,k]
            
            # Principal component weights (because some of the smalles values can become negative, it is chosen to use the aboslute value)
            lambda1 = np.abs(eigenValuesMismatch[n])
            lambda2 = np.abs(eigenValuesPrice[k])
    
            # Time amplitude
            a_k1 = akMismatch[:,n]
            a_k2 = akPrice[:,k]
            
            # Coherence calculations
            c1[n,k] = np.abs(np.dot(p_k1, p_k2))
            c2[n,k] = np.sqrt(lambda1 * lambda2) * np.abs(np.dot(p_k1, p_k2))
            # OLD METHOD! c3[n,k] = (a_k1 * a_k2).mean() * np.abs(np.dot(p_k1, p_k2))
            c3[n,k] = ( (a_k1 * a_k2).mean() / np.sqrt(lambda1 * lambda2) )
            
    # Round off coherence matrices
    c1 = np.around(c1, decimals)
    c2 = np.around(c2, decimals)
    c3 = np.around(c3, decimals)
    
    return (c1, c2, c3)

#%% FilterPrice

def FilterPrice(prices, maxVal):
    """
    Parameters
    ----------
    prices : DataFrame
        Pandas dataframe containing .

    maxVal : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    pricesCopy = prices.copy()
    
    for name in pricesCopy.columns:
        for i in np.arange(len(pricesCopy.index)):
            
            # Check if the current value is larger than the maxVal
            if pricesCopy[name][i] > maxVal:
                
                # If i is the first element
                if i == 0:
                    position = 0
                    value = maxVal + 1
                    
                    # Replace the current value with the first value that is less than the max allowable value
                    while value > maxVal:
                        value = pricesCopy[name][i + position]
                        pricesCopy[name][i] = value
                        position +=1
                
                # If i is the last element
                elif i == (len(pricesCopy.index)-1):
                    pricesCopy[name][i] = pricesCopy[name][i-1]
            
                # Average the current element with its closest neighbouring elements
                else:
                    
                    # Forward looking
                    position = 0
                    valueForward = maxVal + 1
                    while valueForward > maxVal:
                        valueForward = pricesCopy[name][i + position]
                        position +=1
                        
                        # If the end of the array is reached
                        if i + position == (len(pricesCopy.index)-1):
                            valueForward = np.inf
                            break
                    
                    # Backward looking
                    position = 0
                    valueBackward = maxVal + 1
                    while valueBackward > maxVal:
                        valueBackward = pricesCopy[name][i - position]
                        position +=1
                        
                        # If the beginning of the array is reached
                        if i - position == 0:
                            valueBackward = np.inf
                            break
                    
                    
                    # Determine the value to insert into the array
                    value = 0
                    
                    # If the position of the array resulted in being out of bound, the value to insert is determined on only a one of them or the maxVal
                    if valueForward == np.inf and valueBackward == np.inf:
                        value = maxVal
                    
                    # If only one of the val
                    elif valueForward == np.inf:
                        value = valueBackward
                    
                    elif valueBackward == np.inf:
                        value = valueForward
                    
                    else:
                        value = (valueForward + valueBackward) / 2
                    
                    pricesCopy[name][i] = value
    return(pricesCopy)

    