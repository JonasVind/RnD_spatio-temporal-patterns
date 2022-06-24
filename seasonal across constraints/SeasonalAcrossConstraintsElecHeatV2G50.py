# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:40:58 2022

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

# Timer
t0 = time.time() # Start a timer

# Import functions file
sys.path.append(os.path.split(os.getcwd())[0])
from functions_file import *    

#%% Setup paths                     

# Directory of files
directory = os.path.split(os.getcwd())[0] + "\\Data\\elec_heat_v2g50\\"

# Figure path
figurePath = os.getcwd() + "\\Figures\\"

# File name
filename_CO2 = [#"postnetwork-elec_heat_v2g50_0.125_0.6.h5",
                #"postnetwork-elec_heat_v2g50_0.125_0.5.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.4.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.3.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.2.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.1.h5",
                "postnetwork-elec_heat_v2g50_0.125_0.05.h5"]

#%% Setup constrain names

# List of constraints
#constraints = ["40%", "50%", "60%", "70%", "80%", "90%", "95%"]
constraints = ["60%", "70%", "80%", "90%", "95%"]

networkType = "elec_heat_v2g50"

#%% Setup Variables
seasonalElecPC1 = np.zeros([8760,5]) # Electricity price PC 1
seasonalElecPC2 = np.zeros([8760,5]) # Electricity price PC 2
seasonalHeatPC1 = np.zeros([8760,5]) # Heating price PC 1
seasonalHeatPC2 = np.zeros([8760,5]) # Heating price PC 2


#%% Calculate mismatch

for i, file in enumerate(filename_CO2):
    # --------------------------- Electricity -------------------------------#
    # Network
    network = pypsa.Network(directory + file)
    
    # Get the names of the data
    dataNames = network.buses.index.str.slice(0,2).unique()
    
    # Get time stamps
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
    
    # Electricity Price
    priceElec = FilterPrice(network.buses_t.marginal_price[dataNames],465)

    # PCA on mismatch for electricity
    eigenValuesElec, eigenVectorsElec, varianceExplainedElec, normConstElec, TElec = PCA(mismatchElec)

    # Correction
    eigenVectorsElec, correction = AlignEigenVectors(eigenVectorsElec, mismatchElec.columns, country="ES")
    TElec = TElec*correction # Timeseries

    # --------------------------- Heat -------------------------------#
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
    
    # Heat price
    priceHeat = network.buses_t.marginal_price[[x for x in network.buses_t.marginal_price.columns if ("heat" in x) or ("cooling" in x)]]
    priceHeat = priceHeat.groupby(priceHeat.columns.str.slice(0,2), axis=1).sum()
    priceHeat.columns = priceHeat.columns + " heat"
    priceHeat = FilterPrice(priceHeat,465)
    
    # PCA on mismatch for heating
    eigenValuesHeat, eigenVectorsHeat, varianceExplainedHeat, normConstHeat, THeat = PCA(mismatchHeat)

    # Correction
    eigenVectorsHeat, correction = AlignEigenVectors(eigenVectorsHeat, mismatchHeat.columns, country="ES")
    THeat = THeat*correction # Timeseries

    # Append
    seasonalElecPC1[:,i] = TElec[:,0]
    seasonalElecPC2[:,i] = TElec[:,1]
    seasonalHeatPC1[:,i] = THeat[:,0]
    seasonalHeatPC2[:,i] = THeat[:,1]


#%% Plot seasonal prices

title = ["Electricity Price Seasonal Across Constraints (PC 1)",
          "Electricity Price Seasonal Across Constraints (PC 2)",
          "Heating Price Seasonal Across Constraints (PC 1)",
          "Heating Price Seasonal Across Constraints (PC 2)"]
data = [seasonalElecPC1,
        seasonalElecPC2,
        seasonalHeatPC1,
        seasonalHeatPC2]

for k in range(4):
    # Calculate Time
    timeIndex = network.loads_t.p_set.index
    T = pd.DataFrame(data=data[k],index=timeIndex)
    T_avg_hour = T.groupby(timeIndex.hour).mean() # Hour
    T_avg_day = T.groupby([timeIndex.month,timeIndex.day]).mean() # Day
    
    
    # Create figure
    fig = plt.figure(figsize=(12,8),dpi=200) # Figure size and quality
    gs = fig.add_gridspec(5, 2) # Grid size
    
    # Setup gridspace to according figures
    axs = []
    for i in range(2):
        for j in range(5):
            axs.append( fig.add_subplot(gs[j,i]) )
    axs.append( fig.add_subplot(gs[0,1]) ) # seasonal plot
    
    # Color palet
    color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    
    # Plot daily
    for j in range(5): # plot different constaints
        axs[j].hlines(0,-2,25 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
        axs[j].plot(T_avg_hour[j],marker='.',color=color[j], label = constraints[j]) # Plot
        axs[j].set(xlim = [-0.5,23.5],
                   xticks= range(0,24,2),
                   ylim = [-1.5,1.5])
        axs[j].set_ylabel("$a_k$", fontsize = 14,rotation=0) # Y label
        axs[j].legend(loc="upper left",fontsize = 11.5, ncol=3, bbox_to_anchor=(-0.015,1),columnspacing=1.45)
        axs[j].tick_params(axis='both',
                           labelsize=12)
        if j!=4:
            axs[j].set_xticklabels([])
    
    # for Year plot
    x_ax = range(len(T_avg_day[0])) # X for year plot
    maxOffset = 0 # find max space between values
    
    
    # Plot seasonal
    for j in range(5): # plot different constaints
        axs[j+5].hlines(0,-50,400 ,colors="k", linestyles= "--", linewidth=1,alpha=0.5)
        axs[j+5].plot(x_ax,T_avg_day[j],color=color[j], label = constraints[j]) # Plot
        axs[j+5].set(xticks = range(0,370,50),
                     yticks = [-1,1],
                     ylim = [-2,2],
                     xlim = [-10,370])
        axs[j+5].tick_params(axis='both',
                           labelsize=12)
        if j!=4:
            axs[j+5].set_xticklabels([])
        else:
            axs[j+5].set_xticks(np.array([0,31,59,90,120,151,181,212,243,273,304,334])+14) # y-axis label (at 14th in the month)
            axs[j+5].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],rotation=-90) # y-axis label
    
    # Title
    plt.title(title[k],fontsize=18, x=-0.1, y=1.1)
    
    # Adjust spacing
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # save coherence figure
    saveTitle = title[k]
    SavePlot(fig, figurePath, saveTitle) 
    plt.show(all)
