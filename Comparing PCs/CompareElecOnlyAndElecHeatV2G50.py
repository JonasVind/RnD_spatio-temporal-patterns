# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:18:40 2022

@author: jones
"""

#%% Function setup

# Import libraries
import os
import sys
import pypsa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs

# Timer
t0 = time.time() # Start a timer

# Import functions file
sys.path.append(os.path.split(os.getcwd())[0])
from functions_file import *    

#%% Setup paths                     

# Directory of files
directory = os.path.split(os.getcwd())[0] + "\\Data\\"

# Figure path
figurePath = os.getcwd() + "\\Figures\\"

# File name
filename_ElecOnly = [
                #"postnetwork-elec_only_0.125_0.6.h5",
                #"postnetwork-elec_only_0.125_0.5.h5",
                "postnetwork-elec_only_0.125_0.4.h5",
                "postnetwork-elec_only_0.125_0.3.h5",
                "postnetwork-elec_only_0.125_0.2.h5",
                "postnetwork-elec_only_0.125_0.1.h5",
                "postnetwork-elec_only_0.125_0.05.h5"]
filename_ElecSector = [
                #"postnetwork-elec_heat_v2g50_0.125_0.6.h5",
                #"postnetwork-elec_heat_v2g50_0.125_0.5.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.4.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.3.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.2.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.1.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.05.h5"]


#%% Import elec_only data

# file name
file = filename_ElecOnly[4]

# Import network
network = pypsa.Network(directory + "elec_only\\" + file)

# Get the names of the data
dataNames = network.buses.index.str.slice(0,2).unique()

# Get index
timeIndex = network.loads_t.p_set.index

# Electricity load for each country
loadElec = network.loads_t.p_set[dataNames]

# Solar PV generation
generationSolar = network.generators_t.p[dataNames + " solar"]
generationSolar.columns = generationSolar.columns.str.slice(0,2)

# Onshore wind generation
generationOnwind = network.generators_t.p[[country for country in network.generators_t.p.columns if "onwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()

# Offshore wind generation
# Because offwind is only for 21 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
# Create empty array of 8760 x 30, add the offwind generation and remove 'NaN' values.
generationOffwind = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationOffwind += network.generators_t.p[[country for country in network.generators_t.p.columns if "offwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
generationOffwind = generationOffwind.replace(np.nan,0)

# RoR generations
# Because RoR is only for 27 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
# Create empty array of 8760 x 30, add the RoR generation and remove 'NaN' values.
generationRoR = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationRoR += network.generators_t.p[[country for country in network.generators_t.p.columns if "ror" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
generationRoR = generationRoR.replace(np.nan,0)

# Combined generation for electricity
generationElec = generationSolar + generationOnwind + generationOffwind + generationRoR

# Mismatch electricity
mismatchElec = generationElec - loadElec

# PCA on mismatch for electricity
eigenValuesElecOnly, eigenVectorsElecOnly, varianceExplainedElecOnly, normConstElecOnly, TElecOnly = PCA(mismatchElec)

# Correction
eigenVectorsElecOnly, correction = AlignEigenVectors(eigenVectorsElecOnly, mismatchElec.columns, country="ES")
TElecOnly = TElecOnly*correction # Timeseries

# Contribution
dircConElecOnly = Contribution(network, "elec")
lambdaCollectedConElecOnly = ConValueGenerator(normConstElecOnly, dircConElecOnly, eigenVectorsElecOnly)

# Response
dircResElecOnly = ElecResponse(network,True)
lambdaCollectedResElecOnly = ConValueGenerator(normConstElecOnly, dircResElecOnly, eigenVectorsElecOnly)

#%% Import elec_heat_v2g50 data

# file name
file = filename_ElecSector[4]

# Import network
network = pypsa.Network(directory + "elec_heat_v2g50\\" + file)

# Get the names of the data
dataNames = network.buses.index.str.slice(0,2).unique()

# Get the names of the data
dataNames = network.buses.index.str.slice(0,2).unique()

# Get index
timeIndex = network.loads_t.p_set.index

# Electricity load for each country
loadElec = network.loads_t.p_set[dataNames]

# Solar PV generation
generationSolar = network.generators_t.p[dataNames + " solar"]
generationSolar.columns = generationSolar.columns.str.slice(0,2)

# Onshore wind generation
generationOnwind = network.generators_t.p[[country for country in network.generators_t.p.columns if "onwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()

# Offshore wind generation
# Because offwind is only for 21 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
# Create empty array of 8760 x 30, add the offwind generation and remove 'NaN' values.
generationOffwind = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationOffwind += network.generators_t.p[[country for country in network.generators_t.p.columns if "offwind" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
generationOffwind = generationOffwind.replace(np.nan,0)

# RoR generations
# Because RoR is only for 27 countries, additional methods have to be implemented to make it at 8760 x 30 matrix
# Create empty array of 8760 x 30, add the RoR generation and remove 'NaN' values.
generationRoR = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationRoR += network.generators_t.p[[country for country in network.generators_t.p.columns if "ror" in country]].groupby(network.generators.bus.str.slice(0,2),axis=1).sum()
generationRoR = generationRoR.replace(np.nan,0)

# Combined generation for electricity
generationElec = generationSolar + generationOnwind + generationOffwind + generationRoR

# Mismatch electricity
mismatchElec = generationElec - loadElec

# PCA on mismatch for electricity
eigenValuesElecSector, eigenVectorsElecSector, varianceExplainedElecSector, normConstElecSector, TElecSector = PCA(mismatchElec)

# Correction
eigenVectorsElecSector,correction = AlignEigenVectors(eigenVectorsElecSector, mismatchElec.columns, country="ES")
TElecSector = TElecSector*correction # Timeseries

# Contribution
dircConElecSector = Contribution(network, "elec")
lambdaCollectedConElecSector = ConValueGenerator(normConstElecSector, dircConElecSector, eigenVectorsElecSector)

# Response
dircResElecSector = ElecResponse(network,True)
lambdaCollectedResElecSector = ConValueGenerator(normConstElecSector, dircResElecSector, eigenVectorsElecSector)

#%% Setting up grid structure for figure

# Create figure
fig = plt.figure(figsize=(12,12),dpi=200) # Figure size and quality
gs = fig.add_gridspec(22, 24) # Grid size

# Setup gridspace to according figures
axs = []
axs.append( fig.add_subplot(gs[6:10,0:12]) )    # Plot  7: Daily        - Elec Only
axs.append( fig.add_subplot(gs[6:10,12:24]) )   # Plot  8: Daily        - Elec Sector Coupled
axs.append( fig.add_subplot(gs[11:16,0:12]) )   # Plot  9: Seasonal     - Elec Only
axs.append( fig.add_subplot(gs[11:16,12:24]) )  # Plot 10: Seasonal     - Elec Sector Coupled
axs.append( fig.add_subplot(gs[18:22,0:6]) )    # Plot 11: Contribution - Elec Only
axs.append( fig.add_subplot(gs[18:22,6:12]) )   # Plot 12: Response     - Elec Sector Coupled
axs.append( fig.add_subplot(gs[18:22,12:18]) )  # Plot 13: Contribution - Elec Only
axs.append( fig.add_subplot(gs[18:22,18:24]) )  # Plot 14: Response     - Elec Sector Coupled
# axs.append( fig.add_subplot(gs[18:22,0:12]) )    # Plot 11: Contribution - Elec Only
# axs.append( fig.add_subplot(gs[22:26,0:12]) )   # Plot 12: Response     - Elec Sector Coupled
# axs.append( fig.add_subplot(gs[18:22,12:24]) )  # Plot 13: Contribution - Elec Only
# axs.append( fig.add_subplot(gs[22:26,12:24]) )  # Plot 14: Response     - Elec Sector Coupled

# Color palet
color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']


#%% Plotting map

# Data
eigenVector = [eigenVectorsElecOnly, eigenVectorsElecOnly, eigenVectorsElecSector,eigenVectorsElecOnly, eigenVectorsElecSector,eigenVectorsElecSector]
eigenValues = [eigenValuesElecOnly, eigenValuesElecOnly, eigenValuesElecSector,eigenValuesElecOnly, eigenValuesElecSector, eigenValuesElecSector]
PCNO = [1,2,3,1,2,3]

# Settings
title_plot="none"
filename_plot="none"

for i in range(6):
    
    eigen_vectors = eigenVector[i]
    eigen_values = eigenValues[i]
    data_names = dataNames
    PC_NO = PCNO[i]
    
    
    VT = pd.DataFrame(data=eigen_vectors, index=data_names)
    
    # Variance described by each eigen_value
    variance_explained = (eigen_values * 100 ) / eigen_values.sum()
    if i == 0:
        ax = fig.add_subplot(gs[0:5,0:4],projection=cartopy.crs.TransverseMercator(20))
    elif i == 1:
        ax = fig.add_subplot(gs[0:5,4:8],projection=cartopy.crs.TransverseMercator(20))
    elif i == 2:
        ax = fig.add_subplot(gs[0:5,8:12],projection=cartopy.crs.TransverseMercator(20))
    elif i == 3:
        ax = fig.add_subplot(gs[0:5,12:16],projection=cartopy.crs.TransverseMercator(20))
    elif i == 4:
        ax = fig.add_subplot(gs[0:5,16:20],projection=cartopy.crs.TransverseMercator(20))
    elif i == 5:
        ax = fig.add_subplot(gs[0:5,20:24],projection=cartopy.crs.TransverseMercator(20))
    
    ax.axis('off')
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=1)
    ax.coastlines(resolution='10m')
    ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.6,0.8,1), alpha=0.30)
    ax.set_extent ((-9.5, 32, 35, 71), cartopy.crs.PlateCarree())
    ax.gridlines()

    # List of european countries not included in the data (ISO_A2 format)
    europe_not_included = {'AD', 'AL','AX','BY', 'FO', 'GG', 'GI', 'IM', 'IS', 
                            'JE', 'LI', 'MC', 'MD', 'ME', 'MK', 'MT', 'RU', 'SM', 
                            'UA', 'VA', 'XK'}
    
    # Create shapereader file name
    shpfilename = shpreader.natural_earth(resolution='10m',
                                          category='cultural',
                                          name='admin_0_countries')
    
    # Read the shapereader file
    reader = shpreader.Reader(shpfilename)
    
    # Record the reader
    countries = reader.records()
        
    # Determine name_loop variable
    name_loop = 'start'
    
    # Start for-loop
    for country in countries:
        
        # If the country is in the list of the european countries, but not 
        # part of the included european countries: color it gray
        if country.attributes['ISO_A2'] in europe_not_included:
            ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                              facecolor=(0.8, 0.8, 0.8), alpha=0.50, linewidth=0.15, 
                              edgecolor="black", label=country.attributes['ADM0_A3'])
        
        # If the country is in the region Europe
        elif country.attributes['REGION_UN'] == 'Europe':
            
            # Account for Norway and France bug of having no ISO_A2 name
            if country.attributes['NAME'] == 'Norway':
                name_loop = 'NO'
                
            elif country.attributes['NAME'] == 'France':
                name_loop = 'FR'
                
            else:
                name_loop = country.attributes['ISO_A2']
            
            # Color country
            for country_PSA in VT.index.values:
                
                # When the current position in the for loop correspond to the same name: color it
                if country_PSA == name_loop:
                    
                    # Determine the value of the eigen vector
                    color_value = VT.loc[country_PSA][PC_NO-1]
                    
                    # If negative: color red
                    if color_value <= 0:
                        color_value = np.absolute(color_value)*1.5
                        ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                              facecolor=(1, 0, 0), alpha=(np.min([color_value, 1])), linewidth=0.15, 
                              edgecolor="black", label=country.attributes['ADM0_A3'])
                        
                    
                    # If positive: # Color green
                    else:
                        
                        color_value = np.absolute(color_value)*1.5
                        ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                              facecolor=(0, 1, 0), alpha=(np.min([color_value, 1])), linewidth=0.15, 
                              edgecolor="black", label=country.attributes['ADM0_A3'])

                
        # Color any country outside of Europe gray        
        else:
            ax.add_geometries(country.geometry, ccrs.PlateCarree(), 
                              facecolor=(0.8, 0.8, 0.8), alpha=0.50, linewidth=0.15, 
                              edgecolor="black", label=country.attributes['ADM0_A3'])
                    
    # if (variance_explained[PC_NO-1] < 0.1):
    #     plt.legend([r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],2)) + '%'], loc='upper left', fontsize=14, framealpha=1)
    
    # elif (variance_explained[PC_NO-1] < 0.01):
    #     plt.legend([r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],3)) + '%'], loc='upper left', fontsize=14, framealpha=1)

    # else:
    #     plt.legend([r'$\lambda_{'+ str(PC_NO) + '}$ = ' + str(round(variance_explained[PC_NO-1],1)) + '%'], loc='upper left', fontsize=14, framealpha=1)

    # if title_plot != "none":
    #     plt.title(title_plot)
    #     plt.suptitle(filename_plot, fontsize=16, x=.51, y=0.94)
    
# Matrix to determine maximum and minimum values for color bar
color_matrix = np.zeros([2,2])
color_matrix[0,0]=-1
color_matrix[-1,-1]=1

cmap = LinearSegmentedColormap.from_list('mycmap', [(1,0,0),(1,0,0),(1,0.333,0.333),(1,0.666,0.666), 'white',(0.666,1,0.666),(0.333,1,0.333),(0,1,0),(0,1,0)])

cax = fig.add_axes([0.105, 0.728, 0.013, 0.135])
cax.yaxis.set_label_position('left')
im = ax.imshow(color_matrix,cmap=cmap)               
plt.colorbar(im,cax=cax, ticklocation='left', ticks=[-1, -0.5, 0, 0.5, 1])

#%% Daily plot

data = [TElecOnly, TElecSector]
timeIndex = [network.loads_t.p_set.index, network.loads_t.p_set.index]
eigenValues = [eigenValuesElecOnly, eigenValuesElecSector]

for i in range(2):
    
    # Calculate Time
    T = pd.DataFrame(data=data[i],index=timeIndex[i])
    T_avg_hour = T.groupby(timeIndex[i].hour).mean() # Hour
    T_avg_day = T.groupby([timeIndex[i].month,timeIndex[i].day]).mean() # Day
    
    # Eigenvalue used
    eigenValue = eigenValues[i]
    
    # Plot 
    axs[i].hlines(0,-2,25 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
    for j in range(3): # Visible PC
        axs[i].plot(T_avg_hour[j],marker='.',color=color[j], label = '$\lambda_{'+str(j+1)+'}$ = '+str(round(eigenValue[j].sum()*100,1))+'%') # Plot
    # for j in range(6-len(PC_NO)):
    #     axs[0].plot(T_avg_hour[j+len(PC_NO)],color="k",alpha=0.1) # Plot
    axs[i].set(xlim = [-0.5,23.5],
               xticks= range(0,24,2),
               ylim = [-1.5,1.5])
    #axs[0].set_title(label="Daily Average", fontweight="bold", size=13) # Title
    axs[i].set_xlabel('Hours', fontsize = 13) # X label
    axs[i].set_ylabel("$a_k$", fontsize = 14,rotation=0) # Y label
    axs[i].legend(loc="upper left",fontsize = 11.5, ncol=3, bbox_to_anchor=(-0.015,1.3),columnspacing=1.45)
    axs[i].set_yticks([-1,-0.5,0,0.5,1])
    axs[i].tick_params(axis='both',
                       labelsize=12)
    
    if i != 0:
        axs[i].set_yticklabels([])
        axs[i].set_ylabel("") # Y label



#%% Seasonal

for i in range(2):
    
    # Calculate Time
    T = pd.DataFrame(data=data[i],index=timeIndex[i])
    T_avg_hour = T.groupby(timeIndex[i].hour).mean() # Hour
    T_avg_day = T.groupby([timeIndex[i].month,timeIndex[i].day]).mean() # Day

    # offset i    
    i += 2

    # Year plot
    x_ax = range(len(T_avg_day[0])) # X for year plot
    maxOffset = 0 # find max space between values
    
    # Value to offset the PCs 
    offsetValue = 3
    
    # add zero value
    axs[i].hlines(offsetValue,-50,400 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
    axs[i].hlines(0,-50,400 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
    axs[i].hlines(-offsetValue,-50,400 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
    # Mid axis
    axs[i].hlines(1.5,-50,400 ,colors="k",linewidth=1,)
    axs[i].hlines(-1.5,-50,400 ,colors="k",linewidth=1,)
    
    # plot PC1
    T_avg_day[0] = T_avg_day[0]+offsetValue # add offset
    T_avg_day[0].values[T_avg_day[0].values >  6] =  4.5 # Remove values higher than 0
    T_avg_day[0].values[T_avg_day[0].values <  2] =  1.5 # Remove values lower than 0
    axs[i].plot(x_ax,T_avg_day[0],color=color[0]) # Plot
    # plot PC2
    T_avg_day[1] = T_avg_day[1]
    T_avg_day[1].values[T_avg_day[1].values >  2] =  1.5 # Remove values higher than 0
    T_avg_day[1].values[T_avg_day[1].values < -2] = -1.5 # Remove values lower than 0
    axs[i].plot(x_ax,T_avg_day[1],color=color[1]) # Plot
    # plot PC2
    T_avg_day[2] = T_avg_day[2]-offsetValue
    T_avg_day[2].values[T_avg_day[2].values > -2] = -1.5 # Remove values higher than 0
    T_avg_day[2].values[T_avg_day[2].values < -6] = -4.5 # Remove values lower than 0
    axs[i].plot(x_ax,T_avg_day[2],color=color[2]) # Plot
    
    
    # Plot setting
    axs[i].set(#xlabel = "Day",
               #ylabel = "$a_k$ seasonal",
               xticks = range(0,370,50),
               ylim = [-4.5,4.5],
               xlim = [-10,370])
    #axs[2].set_title(label="Seasonal", fontweight="bold", size=13) # Title
    axs[i].set_ylabel("$a_k$", fontsize = 14,rotation=0) # Y label
    axs[i].tick_params(axis='both',
                       labelsize=12)
    
    # y-axis
    axs[i].set_yticks([3.75, 3.00, 2.25, 1.50, 0.75, 0, -0.75, -1.50, -2.25, -3.00 , -3.75])
    axs[i].set_yticklabels([0.75,0,-0.75,"",0.75,0,-0.75,"",0.75,0,-0.75])
    # x-axis 
    axs[i].set_xticks(np.array([0,31,59,90,120,151,181,212,243,273,304,334])+14) # y-axis label (at 14th in the month)
    axs[i].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],rotation=-90) # y-axis label

    if i != 2:
        axs[i].set_yticklabels([])
        axs[i].set_ylabel("") # Y label

#%% Contribution & Response - Elec Only

lambdaContribution = lambdaCollectedConElecOnly
lambdaResponse = lambdaCollectedResElecOnly
eigenValues = eigenValuesElecOnly
FFTCount = 4
PC_NO = [0,1,2]
depth = 2
subtitle = ["Contribution","Response","Covariance"]


for k in range(2):
    # Choose which type of plot
    if k == 0:
        lambdaCollected = lambdaContribution
        ylabel = 'Influance [%]'
    elif k == 1:
        lambdaCollected = lambdaResponse
        ylabel = ""
    elif k == 2:
        #lambdaCollected = lambdaCovariance
        ylabel = ""
    # Find the highest values
    highestCollected = []
    for j in PC_NO:
        highest = abs(lambdaCollected.iloc[j,:]).sort_values(ascending=False)[0:depth]
        highest = lambdaCollected[highest.index].iloc[j].sort_values(ascending=False)[0:depth] # Sort by value
        highestCollected.append(highest)
    # Counter for loop
    counter = 0
    for l, j in enumerate(PC_NO):
        # Create a zero matrix for the percentage values
        percent = np.zeros([depth])
        # Loop to calculate and plot the different values    
        for i in range(depth):
            # absolute percentage
            percent[i] = lambdaCollected[highestCollected[l].index[i]][j]/eigenValues[j]*100
            # Plot
            if i == 0:
                axs[k+FFTCount].bar(counter,percent[i],color=color[j], label = '$\lambda_{'+str(j+1)+'}$ = '+str(round(eigenValues[j].sum()*100,1))+'%')
            else:
                axs[k+FFTCount].bar(counter,percent[i],color=color[j])
            # Insert text into bar
            if percent[i] > 0:
                if percent[i] >= 100:
                    v = 10
                else:
                    v = percent[i]+10
            else:
                    v = 10
            axs[k+FFTCount].text(x=counter,y=v,s=str(round(float(percent[i]),1))+'%',ha='center',size=12,rotation='vertical')
            # Count up
            counter += 1
    # x axis label
    xLabel = []
    for j in range(len(PC_NO)):
        xLabel += list(highestCollected[j].index)
    
    # General plot settings   
    axs[k+FFTCount].set_xticks(np.arange(0,depth*len(PC_NO)))
    axs[k+FFTCount].set_yticks([-50,0,50,100,150])
    axs[k+FFTCount].set_xticklabels(xLabel,rotation=90,fontsize=12)
    axs[k+FFTCount].set(ylabel=ylabel,
                 ylim = [-75,175],
                 title = subtitle[k])
    axs[k+FFTCount].tick_params(axis='both',
                       labelsize=12)
    #axs[k+FFTCount].set_title(label=subtitle[k], fontweight="bold", size=13) # Title
    axs[k+FFTCount].set_ylabel(ylabel, fontsize = 12) # Y label
    axs[k+FFTCount].grid(axis='y',alpha=0.5)
    #axs[k+FFTCount].text(-1.6,165,letter[k],fontsize=13, fontweight="bold")

    if k != 0:
        axs[k+FFTCount].set_yticklabels([])


#%% Contribution & Response - Elec sector coupled

lambdaContribution = lambdaCollectedConElecSector
lambdaResponse = lambdaCollectedResElecSector
eigenValues = eigenValuesElecSector
FFTCount = 6
PC_NO = [0,1,2]
depth = 2
subtitle = ["Contribution","Response","Covariance"]


for k in range(2):
    # Choose which type of plot
    if k == 0:
        lambdaCollected = lambdaContribution
        #ylabel = 'Influance [%]'
    elif k == 1:
        lambdaCollected = lambdaResponse
        ylabel = ""
    elif k == 2:
        #lambdaCollected = lambdaCovariance
        ylabel = ""
    # Find the highest values
    highestCollected = []
    for j in PC_NO:
        highest = abs(lambdaCollected.iloc[j,:]).sort_values(ascending=False)[0:depth]
        highest = lambdaCollected[highest.index].iloc[j].sort_values(ascending=False)[0:depth] # Sort by value
        highestCollected.append(highest)
    # Counter for loop
    counter = 0
    for l, j in enumerate(PC_NO):
        # Create a zero matrix for the percentage values
        percent = np.zeros([depth])
        # Loop to calculate and plot the different values    
        for i in range(depth):
            # absolute percentage
            percent[i] = lambdaCollected[highestCollected[l].index[i]][j]/eigenValues[j]*100
            # Plot
            if i == 0:
                axs[k+FFTCount].bar(counter,percent[i],color=color[j], label = '$\lambda_{'+str(j+1)+'}$ = '+str(round(eigenValues[j].sum()*100,1))+'%')
            else:
                axs[k+FFTCount].bar(counter,percent[i],color=color[j])
            # Insert text into bar
            if percent[i] > 0:
                if percent[i] >= 100:
                    v = 10
                else:
                    v = percent[i]+10
            else:
                    v = 10
            axs[k+FFTCount].text(x=counter,y=v,s=str(round(float(percent[i]),1))+'%',ha='center',size=12,rotation='vertical')
            # Count up
            counter += 1
    # x axis label
    xLabel = []
    for j in range(len(PC_NO)):
        xLabel += list(highestCollected[j].index)
    
    # General plot settings   
    axs[k+FFTCount].set_xticks(np.arange(0,depth*len(PC_NO)))
    axs[k+FFTCount].set_yticks([-50,0,50,100,150])
    axs[k+FFTCount].set_xticklabels(xLabel,rotation=90,fontsize=12)
    axs[k+FFTCount].set(ylabel=ylabel,
                 ylim = [-75,175],
                 title = subtitle[k])
    axs[k+FFTCount].tick_params(axis='both',
                       labelsize=12)
    #axs[k+FFTCount].set_title(label=subtitle[k], fontweight="bold", size=13) # Title
    axs[k+FFTCount].set_ylabel(ylabel, fontsize = 12) # Y label
    axs[k+FFTCount].grid(axis='y',alpha=0.5)
    #axs[k+FFTCount].text(-1.6,165,letter[k],fontsize=13, fontweight="bold")

    axs[k+FFTCount].set_yticklabels([])


#%% Titles

axs[0].text(x=11.5,y=6.2,s="Electricity Only (95% $CO_2$)",ha='center',size=18)
axs[1].text(x=11.5,y=6.2,s="Electricity Coupled (95% $CO_2$)",ha='center',size=18)





#%% Plot and save
path = figurePath
title = "Comparison between Electricity Only and Electricity Sector"

SavePlot(fig, path, title)

plt.show(all)

