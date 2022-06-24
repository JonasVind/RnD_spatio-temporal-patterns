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

filename_ElecSector = [
                #"postnetwork-elec_heat_v2g50_0.125_0.6.h5",
                #"postnetwork-elec_heat_v2g50_0.125_0.5.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.4.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.3.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.2.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.1.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.05.h5"]


#%% Import elec_heat_v2g50 heat price data

# file name
file = filename_ElecSector[4]

# Import network
network = pypsa.Network(directory + "elec_heat_v2g50\\" + file)

# Get the names of the data
dataNames = network.buses.index.str.slice(0,2).unique()

# Get time stamps
timeIndex = network.loads_t.p_set.index

# Heat load for each country
loadHeat = network.loads_t.p_set[[country for country in network.loads_t.p_set.columns if "heat" in country]].groupby(network.loads.bus.str.slice(0,2),axis=1).sum()

# Heat generators for each country (solar collectors)
# Because some countries have urban collectors, while other have central collectors, 
# additional methods have to be implemented to make it at 8760 x 30 matrix
# Create empty array of 8760 x 30, add the heat generators and remove 'NaN' values.
generationHeatSolar = network.generators_t.p[dataNames + " solar thermal collector"]
generationHeatSolar.columns = generationHeatSolar.columns.str.slice(0,2)

# Urban heat
generationHeatUrbanSingle = network.generators_t.p[[country for country in network.generators_t.p.columns if "urban" in country]]
generationHeatUrbanSingle.columns = generationHeatUrbanSingle.columns.str.slice(0,2)
generationHeatUrban = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationHeatUrban += generationHeatUrbanSingle
generationHeatUrban = generationHeatUrban.replace(np.nan,0)

# Central heat
generationHeatCentralSingle = network.generators_t.p[[country for country in network.generators_t.p.columns if "central" in country]]
generationHeatCentralSingle.columns = generationHeatCentralSingle.columns.str.slice(0,2)
generationHeatCentral = pd.DataFrame(np.zeros([8760,30]),index=timeIndex, columns=dataNames)
generationHeatCentral += generationHeatCentralSingle
generationHeatCentral = generationHeatCentral.replace(np.nan,0)

# Combine generation for heat
generationHeat = generationHeatSolar + generationHeatUrban + generationHeatCentral

# Mismatch electricity
mismatchHeat = generationHeat - loadHeat

# catagories seasons
spring = mismatchHeat.index[1416:3624]
summer = mismatchHeat.index[3624:5832]
fall = mismatchHeat.index[5832:8016]
winter1 = mismatchHeat.index[0:1416]
winter2 = mismatchHeat.index[8016:8760]

# Delete seasons
mismatchHeat = mismatchHeat.drop(index=spring)
mismatchHeat = mismatchHeat.drop(index=fall)
mismatchHeat = mismatchHeat.drop(index=winter1)
mismatchHeat = mismatchHeat.drop(index=winter2)

# PCA on mismatch for electricity
eigenValuesHeatMis, eigenVectorsHeatMis, varianceExplainedHeatMis, normConstHeatMis, THeatMis = PCA(mismatchHeat)

# Correction
eigenVectorsHeatMis, correction = AlignEigenVectors(eigenVectorsHeatMis, mismatchHeat.columns, country="ES")
THeatMis = THeatMis*correction # Timeseries

# Contribution Heat
dircConHeat = Contribution(network, "heat")
lambdaCollected = ConValueGenerator(normConstHeatMis, dircConHeat, eigenVectorsHeatMis)

# Response Heat
dircResHeat = HeatResponse(network,True)
lambdaCollected = ConValueGenerator(normConstHeatMis, dircResHeat, eigenVectorsHeatMis)

# Covariance Heat
covMatrix = CovValueGenerator(dircConHeat, dircResHeat , True, normConstHeatMis,eigenVectorsHeatMis)


#%% Import elec_heat_v2g50 heat price data

# file name
file = filename_ElecSector[4]

# Import network
network = pypsa.Network(directory + "elec_heat_v2g50\\" + file)

# Get the names of the data
priceHeat = network.buses_t.marginal_price[[x for x in network.buses_t.marginal_price.columns if ("heat" in x) or ("cooling" in x)]]
priceHeat = priceHeat.groupby(priceHeat.columns.str.slice(0,2), axis=1).sum()
priceHeat.columns = priceHeat.columns + " heat"

# Prices for electricity for each country (restricted to 1000 €/MWh)
#priceElecSector = FilterPrice(network.buses_t.marginal_price[dataNames], 465).drop('BA', axis=1)
priceHeat = FilterPrice(priceHeat,465)

# catagories seasons
spring = priceHeat.index[1416:3624]
summer = priceHeat.index[3624:5832]
fall = priceHeat.index[5832:8016]
winter1 = priceHeat.index[0:1416]
winter2 = priceHeat.index[8016:8760]

# Delete seasons
priceHeat = priceHeat.drop(index=spring)
priceHeat = priceHeat.drop(index=fall)
priceHeat = priceHeat.drop(index=winter1)
priceHeat = priceHeat.drop(index=winter2)

# PCA on nodal prices for electricity
eigenValuesHeatPrice, eigenVectorsHeatPrice, varianceExplainedHeatPrice, normConstHeatPrice, THeatPrice = PCA(priceHeat)

# Correction
eigenVectorsHeatPrice, correction = AlignEigenVectors(eigenVectorsHeatPrice, priceHeat.columns, country="ES heat")
THeatPrice = THeatPrice*correction # Timeseries

#%% Setting up grid structure for figure

# Create figure
fig = plt.figure(figsize=(10,9),dpi=200) # Figure size and quality
gs = fig.add_gridspec(14, 12) # Grid size

# Setup gridspace to according figures
axs = []
axs.append( fig.add_subplot(gs[5:9,0:6]) )      # Plot 10: Seasonal - Real Price
axs.append( fig.add_subplot(gs[5:9,6:12]) )     # Plot 11: Seasonal - Elec Only
axs.append( fig.add_subplot(gs[10:14,0:6]) )    # Plot 13: Daily    - Real Price
axs.append( fig.add_subplot(gs[10:14,6:12]) )   # Plot 14: Daily    - Elec Only

# Color palet
color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']


#%% Plotting map
eigenValuesHeatMis
eigenValuesHeatPrice
# Data
eigenVector = [eigenVectorsHeatPrice, eigenVectorsHeatPrice, eigenVectorsHeatMis, eigenVectorsHeatMis]
eigenValues = [eigenValuesHeatPrice, eigenValuesHeatPrice, eigenValuesHeatMis, eigenValuesHeatMis]
PCNO = [1,2,1,2]
dataNames =  dataNames

# Settings
title_plot="none"
filename_plot="none"

for i in range(4):
    
    eigen_vectors = eigenVector[i]
    eigen_values = eigenValues[i]
    data_names = dataNames
    PC_NO = PCNO[i]
    
    
    VT = pd.DataFrame(data=eigen_vectors, index=data_names)
    
    # Variance described by each eigen_value
    variance_explained = (eigen_values * 100 ) / eigen_values.sum()
    if i == 0:
        ax = fig.add_subplot(gs[0:4,0:3],projection=cartopy.crs.TransverseMercator(20))
    elif i == 1:
        ax = fig.add_subplot(gs[0:4,3:6],projection=cartopy.crs.TransverseMercator(20))
    elif i == 2:
        ax = fig.add_subplot(gs[0:4,6:9],projection=cartopy.crs.TransverseMercator(20))
    elif i == 3:
        ax = fig.add_subplot(gs[0:4,9:12],projection=cartopy.crs.TransverseMercator(20))
        
    
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

cax = fig.add_axes([0.11, 0.672, 0.012, 0.205])
cax.yaxis.set_label_position('left')
im = ax.imshow(color_matrix,cmap=cmap)               
plt.colorbar(im,cax=cax, ticklocation='left', ticks=[-1, -0.5, 0, 0.5, 1])


#%% Daily plot

data = [THeatPrice, THeatMis]
timeIndex = [mismatchHeat.index,priceHeat.index]
eigenValues = [eigenValuesHeatPrice, eigenValuesHeatMis]

for i in range(2):
    
    # Calculate Time
    T = pd.DataFrame(data=data[i],index=timeIndex[i])
    T_avg_hour = T.groupby(timeIndex[i].hour).mean() # Hour
    T_avg_day = T.groupby([timeIndex[i].month,timeIndex[i].day]).mean() # Day
    
    # Eigenvalue used
    eigenValue = eigenValues[i]
    
    # Plot 
    axs[i].hlines(0,-2,25 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
    for j in range(2): # Visible PC
        axs[i].plot(T_avg_hour[j],marker='.',color=color[j], label = '$\lambda_{'+str(j+1)+'}$ = '+str(round(eigenValue[j].sum()*100,1))+'%') # Plot
    # for j in range(6-len(PC_NO)):
    #     axs[0].plot(T_avg_hour[j+len(PC_NO)],color="k",alpha=0.1) # Plot
    axs[i].set(xlim = [-0.5,23.5],
               xticks= range(0,24,2),
               ylim = [-1.4,1.4])
    #axs[0].set_title(label="Daily Average", fontweight="bold", size=13) # Title
    axs[i].set_xlabel('Hours', fontsize = 13) # X label
    axs[i].set_ylabel("$a_k$", fontsize = 14,rotation=0) # Y label
    axs[i].legend(loc="upper left",fontsize = 11.5, ncol=2, bbox_to_anchor=(0.01,1.26),columnspacing=4.5)
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
    offsetValue = 4
    
    # add zero value
    axs[i].hlines(2,-50,120 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
    #axs[i].hlines(0,-50,400 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
    axs[i].hlines(-2,-50,120 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
    # Mid axis
    axs[i].hlines(0,-50,120 ,colors="k",linewidth=1,)
    #axs[i].hlines(-2,-50,400 ,colors="k",linewidth=1,)
    
    # plot PC1
    T_avg_day[0] = T_avg_day[0]+2 # add offset
    T_avg_day[0].values[T_avg_day[0].values >  4] =  4 # Remove values higher than 0
    T_avg_day[0].values[T_avg_day[0].values <  0] =  0 # Remove values lower than 0
    axs[i].plot(x_ax,T_avg_day[0],color=color[0]) # Plot
    # plot PC2
    T_avg_day[1] = T_avg_day[1]-2
    T_avg_day[1].values[T_avg_day[1].values >  0] =  0 # Remove values higher than 0
    T_avg_day[1].values[T_avg_day[1].values < -4] = -4 # Remove values lower than 0
    axs[i].plot(x_ax,T_avg_day[1],color=color[1]) # Plot
    # # plot PC3
    # T_avg_day[2] = T_avg_day[2]-offsetValue
    # T_avg_day[2].values[T_avg_day[2].values > -2] = -2 # Remove values higher than 0
    # T_avg_day[2].values[T_avg_day[2].values < -6] = -6 # Remove values lower than 0
    # axs[i].plot(x_ax,T_avg_day[2],color=color[2]) # Plot
    
    
    # Plot setting
    axs[i].set(#xlabel = "Day",
               #ylabel = "$a_k$ seasonal",
               xticks = range(0,90,10),
               ylim = [-4,4],
               xlim = [-5,95])
    #axs[2].set_title(label="Seasonal", fontweight="bold", size=13) # Title
    axs[i].set_ylabel("$a_k$", fontsize = 14,rotation=0) # Y label
    axs[i].tick_params(axis='both',
                       labelsize=12)
    
    # y-axis
    axs[i].set_yticks([3,2,1,0,-1,-2,-3])
    axs[i].set_yticklabels([1,0,-1,"",1,0,-1])
    # x-axis 
    axs[i].set_xticks(np.array([0,30,61])+14) # y-axis label (at 14th in the month)
    axs[i].set_xticklabels(["Jun","Jul","Aug"],rotation=-90) # y-axis label

    if i != 2:
        axs[i].set_yticklabels([])
        axs[i].set_ylabel("") # Y label

#%% Titles

axs[0].text(x=11.5,y=5.35,s="Heating Prices (95% $CO_2$)",ha='center',size=18)
axs[1].text(x=11.5,y=5.35,s="Heating Mismatch (95% $CO_2$)",ha='center',size=18)


#%% Plot and save
path = figurePath
title = "Comparison between Heating Prices and Mismatch (summer)"

SavePlot(fig, path, title)

plt.show(all)

