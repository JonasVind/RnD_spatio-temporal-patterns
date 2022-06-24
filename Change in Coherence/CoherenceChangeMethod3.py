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
coherenceMismatchC3_1 = []  # Coherence between Electricity and Heating mismatch
coherenceMismatchC3_2 = []  # Coherence between Electricity and Transport mismatch
coherenceMismatchC3_3 = []  # Coherence between Heating and Transport mismatch
coherencePriceC3_1 = []     # Coherence between Electricity and Heating Price
coherencePriceC3_2 = []     # Coherence between Electricity and Transport Price
coherencePriceC3_3 = []     # Coherence between Heating and Transport Price
coherenceElecC3 = []        # Coherence between Electricity mismatch and price
coherenceHeatC3 = []        # Coherence between Heating mismatch and price
coherenceTransC3 = []       # Coherence between Transport mismatch and price

TESTER1 = []
TESTER2 = []

#%% Calculate mismatch

for file in filename_CO2:
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
    
    # ----------------------------- Transport --------------------------------#
    # Transport load for each country
    loadTransport = network.loads_t.p_set[dataNames + ' transport']
    
    # Generation transport
    generationTransport = pd.DataFrame(data=np.zeros([8760,30]), index=timeIndex, columns=(dataNames + ' transport'))
    
    # Mismatch transport
    mismatchTransport = generationTransport - loadTransport
    
    # Transport Price
    priceTrans = FilterPrice(network.buses_t.marginal_price[dataNames + " EV battery"],465)
    
    
    # ----------------------------- Coherence --------------------------------#
    # Mismatch
    c1Mismatch1, c2Mismatch1, c3Mismatch1 = Coherence(mismatchElec, mismatchHeat)
    c1Mismatch2, c2Mismatch2, c3Mismatch2 = Coherence(mismatchElec, mismatchTransport)
    c1Mismatch3, c2Mismatch3, c3Mismatch3 = Coherence(mismatchHeat, mismatchTransport)
    # Price
    c1Price1, c2Price1, c3Price1 = Coherence(priceElec, priceHeat)
    c1Price2, c2Price2, c3Price2 = Coherence(priceElec, priceTrans)
    c1Price3, c2Price3, c3Price3 = Coherence(priceHeat, priceTrans)
    # Mismatch and price
    c1Elec, c2Elec, c3Elec = Coherence(mismatchElec, priceElec)
    c1Heat, c2Heat, c3Heat = Coherence(mismatchHeat, priceHeat)
    c1Trans, c2Trans, c3Trans = Coherence(mismatchTransport, priceTrans)
    
    # Append
    coherenceMismatchC3_1.append(c3Mismatch1) 
    coherenceMismatchC3_2.append(c3Mismatch2)
    coherenceMismatchC3_3.append(c3Mismatch3)
    coherencePriceC3_1.append(c3Price1)
    coherencePriceC3_2.append(c3Price2)
    coherencePriceC3_3.append(c3Price3)
    coherenceElecC3.append(c3Elec)
    coherenceHeatC3.append(c3Heat)
    coherenceTransC3.append(c3Trans)


#%% Coherence

# ---------------------- Coherence mehtod 3 (amplitude) --------------------------- #  
coherenceC2 = np.zeros([3,5])

for j in range(len(coherenceElecC3)):
    coherenceC2[0,j] = coherenceElecC3[j][0,0]
    coherenceC2[1,j] = coherenceHeatC3[j][0,0]
    coherenceC2[2,j] = coherencePriceC3_1[j][0,0]

# Plot figure
fig = plt.figure(figsize=(10,3), dpi=300)
plt.title("Amplitude Coherence (PC 1 vs PC 1)",fontsize=16)

plt.xticks(np.arange(len(constraints)), constraints, fontsize=12)
plt.yticks(ticks=np.linspace(-1,1,9), fontsize=12)
plt.grid(alpha=0.3)
plt.plot(coherenceC2[0], marker='o', markersize=5, label = "Electricity mismatch\n& price coherence")
plt.plot(coherenceC2[1], marker='o', markersize=5, label = "Heating mismatch\n& price coherence")
plt.plot(coherenceC2[2], marker='o', markersize=5, label = "Electricity & heating\nprice coherence")
plt.ylabel("Coherence [%]", fontsize=14)
plt.ylim([-1,1])
plt.legend(loc="lower left", fontsize=12, bbox_to_anchor =(0.03, -0.4), labelspacing=1, ncol=3)

# save coherence figure
saveTitle = "Amplitude Coherence (PC 1 vs PC 1)"
SavePlot(fig, figurePath, saveTitle) 
plt.show(all)

#%% Coherence

# ---------------------- Coherence mehtod 3 (amplitude) --------------------------- #  
coherenceC2 = np.zeros([3,5])

for j in range(len(coherenceElecC3)):
    coherenceC2[0,j] = coherenceElecC3[j][1,1]
    coherenceC2[1,j] = coherenceHeatC3[j][1,1]
    coherenceC2[2,j] = coherencePriceC3_1[j][1,1]

# Switch sign
coherenceC2[1,2] = coherenceC2[1,2]*-1
coherenceC2[2,2] = coherenceC2[2,2]*-1

# Plot figure
fig = plt.figure(figsize=(10,3), dpi=300)
plt.title("Amplitude Coherence (PC 2 vs PC 2)",fontsize=16)

plt.xticks(np.arange(len(constraints)), constraints, fontsize=12)
plt.yticks(ticks=np.linspace(-1,1,9), fontsize=12)
plt.grid(alpha=0.3)
plt.plot(coherenceC2[0], marker='o', markersize=5, label = "Electricity mismatch\n& price coherence")
plt.plot(coherenceC2[1], marker='o', markersize=5, label = "Heating mismatch\n& price coherence")
plt.plot(coherenceC2[2], marker='o', markersize=5, label = "Electricity & heating\nprice coherence")
plt.ylabel("Coherence [%]", fontsize=14)
plt.ylim([-1,1])
plt.legend(loc="lower left", fontsize=12, bbox_to_anchor =(0.03, -0.4), labelspacing=1, ncol=3)

# save coherence figure
saveTitle = "Amplitude Coherence (PC 2 vs PC 2)"
SavePlot(fig, figurePath, saveTitle) 
plt.show(all)

