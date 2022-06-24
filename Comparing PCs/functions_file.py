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

#%% MeanAndCenter
def MeanAndCenter(value):
    """

    Parameters
    ----------
    Value : Panda Dataframe
        The given value, either a load or generator to be centered around the mean value

    Returns
    -------
    valueCenter : Panda Dataframe
        The calulated value

    """    
    
    # Mean
    valueMean = np.mean(value, axis=0)
    
    # Center
    valueCenter = np.subtract(value,valueMean.T)
    
    return valueCenter

#%% CovNameGenerator

def CovNameGenerator(conNames):
    """
    

    Parameters
    ----------
    conNames : array
        array of strings with names of either generaters, load or reponsetypes

    Returns
    -------
    con : array
        output an larger array than original where the the covariance versions of the original are attached

    """
    
    amount = int(len((conNames)*(len(conNames)+1))/2)
    
    con = [""]*amount
    
    for j in range( amount ):
        
        if j < len(conNames):
            con[j] = conNames[j]
        if j < len(conNames)-1:
            con[j+len(conNames)] = conNames[0] + "/ \n" + conNames[j+1]
        elif j < len(conNames)-1 + len(conNames)-2:
            con[j+len(conNames)] = conNames[1] + "/ \n" + conNames[j-len(conNames)+1+2]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3:
            con[j+len(conNames)] = conNames[2] + "/ \n" + conNames[j-len(conNames)*2 + 1+2+3]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4:
            con[j+len(conNames)] = conNames[3] + "/ \n" + conNames[j-len(conNames)*3 + 1+2+3+4]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5:
            con[j+len(conNames)] = conNames[4] + "/ \n" + conNames[j-len(conNames)*4 + 1+2+3+4+5]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6:
            con[j+len(conNames)] = conNames[5] + "/ \n" + conNames[j-len(conNames)*5 + 1+2+3+4+5+6]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6 + len(conNames)-7:
            con[j+len(conNames)] = conNames[6] + "/ \n" + conNames[j-len(conNames)*6 + 1+2+3+4+5+6+7]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6 + len(conNames)-7 + len(conNames)-8:
            con[j+len(conNames)] = conNames[7] + "/ \n" + conNames[j-len(conNames)*7 + 1+2+3+4+5+6+7+8]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6 + len(conNames)-7 + len(conNames)-8 + len(conNames)-9:
            con[j+len(conNames)] = conNames[8] + "/ \n" + conNames[j-len(conNames)*8 + 1+2+3+4+5+6+7+8+9]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6 + len(conNames)-7 + len(conNames)-8 + len(conNames)-9 + len(conNames)-10:
            con[j+len(conNames)] = conNames[9] + "/ \n" + conNames[j-len(conNames)*9 + 1+2+3+4+5+6+7+8+9+10]
        elif j < len(conNames)-1 + len(conNames)-2 + len(conNames)-3 + len(conNames)-4 + len(conNames)-5 + len(conNames)-6 + len(conNames)-7 + len(conNames)-8 + len(conNames)-9 + len(conNames)-10 + len(conNames)-11:
            con[j+len(conNames)] = conNames[10] + "/ \n" + conNames[j-len(conNames)*10 + 1+2+3+4+5+6+7+8+9+10+11]
        else:
            assert True, "something is wrong"
            
    return con

#%% LinksEff
    
def LinksEff(network):
    """
    

    Parameters
    ----------
    network : PyPSA network type
        input network

    Returns
    -------
    linkseff : array
        an array of the types of links that have an efficency. Used to calculate the response values

    """
    linkseff = network.links # Save link data
    linkseff = linkseff.drop(linkseff.index[np.where(linkseff["bus1"].str.contains("H2"))]) # Delete bus1 = H2
    linkseff = linkseff.drop(linkseff.index[np.where(linkseff["bus1"].str.contains("battery"))]) # Delete bus1 = battery
    linkseff = linkseff.drop(linkseff.index[np.where(linkseff["bus1"].str.contains("water"))]) # Delete bus1 = water tanks
    linkseff = pd.DataFrame(data={"eff": linkseff.efficiency.values}, index=linkseff.index.str.slice(3))
    linkseff = linkseff[~linkseff.index.duplicated(keep='first')]
    linkseff = linkseff[:np.where(linkseff.index.str.len()==2)[0][0]]
    
    return linkseff

#%% Contribution

def Contribution(network,contributionType):

    # Save index with only countries
    country_column = network.loads.index[:30]
    
    # For contribution of the electric sector
    if contributionType == "elec":
        # Load
        load     = network.loads_t.p_set.filter(items=country_column)
        # Generators
        wind     = network.generators_t.p.filter(regex="wind").groupby(network.generators.bus,axis=1).sum()
        solar    = network.generators_t.p.filter(items=country_column+" solar").groupby(network.generators.bus,axis=1).sum()
        ror = network.generators_t.p[[country for country in network.generators_t.p.columns if "ror" in country]]
        ror = pd.DataFrame(np.zeros([8760,30]), index=network.loads_t.p.index, columns=(network.buses.index.str.slice(0,2).unique() + ' ror')).add(ror, fill_value=0)
        
        # Collect terms
        collectedContribution = {"Load Electric":               - load, 
                                 "Wind":                        + wind,
                                 "Solar PV":                    + solar,
                                 "RoR":                         + ror
                                 }
     
    # For contribution of the heating sector    
    elif contributionType == "heat":
        # Load
        load        = pd.DataFrame(data=network.loads_t.p_set.filter(items=country_column+" heat").values,columns = country_column, index = network.generators_t.p.index)
        loadUrban = network.loads_t.p_set[[country for country in network.loads_t.p_set.columns if "urban heat" in country]]
        loadUrban = pd.DataFrame(np.zeros([8760,30]), index=network.loads_t.p_set.index, columns=(network.buses.index.str.slice(0,2).unique() + ' urban heat')).add(loadUrban, fill_value=0)
        #loadUrban   = pd.DataFrame(data=network.loads_t.p_set.filter(items=country_column+" urban heat").values,columns = country_column, index = network.generators_t.p.index)
        
        # Check if brownfield network
        isBrownfield = len([x for x in network.links_t.p0.columns if "nuclear" in x]) > 0
        if isBrownfield == False: 
        
            # Generators
            SolarCol = pd.DataFrame(data=network.generators_t.p.filter(items=country_column+" solar thermal collector").groupby(network.generators.bus,axis=1).sum().values,columns = country_column, index = network.generators_t.p.index)
            centralUrbanSolarCol = pd.DataFrame(data=network.generators_t.p.filter(regex=" solar").groupby(network.generators.bus,axis=1).sum().filter(regex=" urban").values,columns = country_column, index = network.generators_t.p.index)
            
            # Collect terms
            collectedContribution = {"Load Heat":                   - load,
                                     "Load Urban Heat":             - loadUrban,
                                     "Solar Collector":             + SolarCol,
                                     "Central-Urban\nSolar Collector": + centralUrbanSolarCol
                                     }
            
        elif isBrownfield == True: 
        
            loadCooling = network.loads_t.p_set[[country for country in network.loads_t.p_set.columns if "cooling" in country]]
            loadCooling = pd.DataFrame(np.zeros([8760,30]), index=network.loads_t.p_set.index, columns=(network.buses.index.str.slice(0,2).unique() + ' cooling')).add(loadCooling, fill_value=0)    
        
            # Collect terms
            collectedContribution = {"Load Heat":                   - load,
                                     "Load Urban Heat":             - loadUrban,
                                     "Load Cooling":                - loadCooling
                                     }
            
    else: # Error code if wrong input
        assert False, "choose either elec or heat as a contributionType"
    
    return collectedContribution

#%% ElecResponse

def ElecResponse(network, collectTerms=True):
    
    # Save index with only countries
    country_column = network.loads.index[:30]
    
    # Determin size of loads to know if the system includes heat and/or transport
    size = network.loads.index.size
    
    # Find efficency of links
    busEff = LinksEff(network)
    
    # Check if brownfield network
    isBrownfield = len([x for x in network.links_t.p0.columns if "nuclear" in x]) > 0
    if isBrownfield == True: 
        size = 0 # Makes use that the sector coupled links are correct
        
    # General Reponse
    H2                          = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    battery                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    PHS                         = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    importExport                = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    OCGT                        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    hydro                       = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)

    if isBrownfield == False:
        # Heating related response:
        if size >= 90:
            groundHeatPump              = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
            centralUrbanHeatPump        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
            ResistiveHeater             = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
            centralUrbanResistiveHeater = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
            CHPElec                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        
        # Transportation related response
        if size == 60 or size == 120:
            EVBattery                   = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)

    elif isBrownfield == True:
        
        # new types of backup
        CCGT                        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        coal                        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        lignite                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        nuclear                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        oil                         = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        biomassEOP                  = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        
        # Heat related
        groundHeatPump              = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        centralUrbanHeatPump        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        ResistiveHeater             = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        centralUrbanResistiveHeater = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        coolingPump                 = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        gasCHPElec                  = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        biomassCHPElec              = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)

    for country in country_column:
        # Storage
        H2[country] = (- network.links_t.p0.filter(regex=country).filter(regex="Electrolysis")
            + network.links_t.p0.filter(regex=country).filter(regex="H2 Fuel Cell").values
            * busEff.loc["H2 Fuel Cell"].values[0]
            ).values 
        
        battery[country] = (- network.links_t.p0.filter(regex=country).filter(regex="battery charger")
                  + network.links_t.p0.filter(regex=country).filter(regex="battery discharger").values
                  * busEff.loc["battery discharger"].values[0]
                  ).values
        
        if (+ network.storage_units_t.p.filter(regex=country).filter(regex="PHS")).size > 0:
            PHS[country] = (+ network.storage_units_t.p.filter(regex=country).filter(regex="PHS")).values
        
        # Import/export
        if network.links_t.p0.filter(regex=country).filter(regex=str(country)+"-").groupby(network.links.bus0.str.slice(0,2), axis=1).sum().size > 0:
            Import = (+ network.links_t.p0.filter(regex=country).filter(regex=str(country)+"-").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()).values
        else:
            Import = np.zeros([8760,1])
        
        if network.links_t.p0.filter(regex=country).filter(regex="-"+str(country)).groupby(network.links.bus0.str.slice(0,2), axis=1).sum().size > 0:
            Export = (- network.links_t.p0.filter(regex=country).filter(regex="-"+str(country)).groupby(network.links.bus1.str.slice(0,2), axis=1).sum()).values
        else:
            Export = np.zeros([8760,1])  
            
        importExport[country] = Import + Export
    
        # Backup generator
        OCGT[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="OCGT").values
                * busEff.loc["OCGT"].values[0]
                )
        # dispathable backup generator
        if (+ network.storage_units_t.p.filter(regex=country).filter(regex="hydro")).size > 0:
            hydro[country] = (+ network.storage_units_t.p.filter(regex=country).filter(regex="hydro")).values
        
        
        # Heating related response:
        if size >= 90:
            # Sector links to Heat
            groundHeatPump[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="ground heat pump").groupby(network.links.bus0.str.slice(0,2), axis=1).sum() ).values
            
            if (network.links_t.p1.filter(regex=country).filter(regex="central heat pump")).size > 0:
                centralUrbanHeatPump[country] = (network.links_t.p0.filter(regex=country).filter(regex="central heat pump")).values
            else:
                centralUrbanHeatPump[country] = (network.links_t.p0.filter(regex=country).filter(regex="urban heat pump")).values
            
            ResistiveHeater[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" resistive heater") ).values
           
            if (network.links_t.p1.filter(regex=country).filter(regex="central resistive heater")).size > 0:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex="central resistive heater")).values
            else:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex="urban resistive heater")).values

            # CHP
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="CHP electric")).size > 0:
                CHPElec[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central CHP electric").values
                       * busEff.loc["central CHP electric"].values[0]
                       )        
        
        
        # Transportation related response
        if size == 60 or size == 120:
        
            # Sector links to vehicle
            EVBattery[country] = (- network.links_t.p0.filter(regex=str(country)+" ").filter(regex="BEV")
                - network.links_t.p1.filter(regex=country).filter(regex="V2G").values
                #* busEff.loc["V2G"].values[0]
                ).values      

        # Brownfield related response
        if isBrownfield == True: 
            # Backup generator
            CCGT[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="CCGT").values
                    * busEff.loc["CCGT"].values[0]
                    )
            oil[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=" oil").values
                    * busEff.loc["oil"].values[0]
                    )
            coal[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="coal").values
                    * busEff.loc["coal"].values[0]
                    )
            lignite[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="lignite").values
                    * busEff.loc["lignite"].values[0]
                    )
            nuclear[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="nuclear").values
                    * busEff.loc["nuclear"].values[0]
                    )
            biomassEOP[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="EOP").values
                    * busEff.loc["biomass EOP"].values[0]
                    )
            # Sector links to Heat
            groundHeatPump[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="decentral heat pump").groupby(network.links.bus0.str.slice(0,2), axis=1).sum() ).values
            
            if (network.links_t.p1.filter(regex=country).filter(regex="central heat pump")).size > 0:
                centralUrbanHeatPump[country] = (network.links_t.p0.filter(regex=country).filter(regex="central heat pump")).values
            else:
                centralUrbanHeatPump[country] = (network.links_t.p0.filter(regex=country).filter(regex="urban heat pump")).values
            
            ResistiveHeater[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" decentral resistive heater") ).values
           
            if (network.links_t.p1.filter(regex=country).filter(regex="central resistive heater")).size > 0:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex="central resistive heater")).values
            else:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex="urban resistive heater")).values

            coolingPump[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="cooling pump").groupby(network.links.bus0.str.slice(0,2), axis=1).sum() ).values
                
            # CHP
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="gas CHP electric")).size > 0:
                gasCHPElec[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central gas CHP electric").values
                       * busEff.loc["central gas CHP electric"].values[0]
                       )        
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="biomass CHP electric")).size > 0:
                biomassCHPElec[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central biomass CHP electric").values
                       * busEff.loc["central biomass CHP electric"].values[0]
                       )  
        
    # Collecting terms        
    storage = + H2 + battery.values + PHS.values
    importExport = - importExport
    backupGenerator = OCGT
    dispBackupGenerator = hydro
    
    if size >= 90:
        heatCouple = - groundHeatPump - centralUrbanHeatPump.values - ResistiveHeater.values - centralUrbanResistiveHeater.values
        CHP = CHPElec
    if size == 60 or size == 120:
        transCouple = EVBattery
    
    if isBrownfield == True:  
        backupGenerator = OCGT + CCGT + coal.values + lignite.values + nuclear.values + oil.values + biomassEOP.values
        heatCouple = - groundHeatPump - centralUrbanHeatPump.values - ResistiveHeater.values - centralUrbanResistiveHeater.values - coolingPump.values
        CHP = gasCHPElec + biomassCHPElec.values
    
    if isBrownfield == False:
        if collectTerms == True: # Collecting into general terms
            if size == 60: # elec + transport
                collectedResponse = {"Storage":                         + storage, 
                                     "Import-Export":                   + importExport,
                                     "Backup Generator":                + backupGenerator,
                                     "Hydro Reservoir":                 + dispBackupGenerator,
                                     "Transport Couple":                + transCouple
                                     }
            elif size == 90: # elec + heating
                collectedResponse = {"Storage":                         + storage, 
                                     "Import-Export":                   + importExport,
                                     "Backup Generator":                + backupGenerator,
                                     "Hydro Reservoir":                 + dispBackupGenerator,
                                     "Heat Couple":                     + heatCouple,
                                     "CHP Electric":                    + CHP
                                     }
            elif size == 120: # elec + heating + transport
                collectedResponse = {"Storage":                         + storage, 
                                     "Import-Export":                   + importExport,
                                     "Backup Generator":                + backupGenerator,
                                     "Hydro Reservoir":                 + dispBackupGenerator,
                                     "Heat Couple":                     + heatCouple,
                                     "Transport Couple":                + transCouple,
                                     "CHP Electric":                    + CHP
                                     }   
            else: # elec only
                collectedResponse = {"Storage":                         + storage, 
                                     "Import-Export":                   + importExport,
                                     "Backup Generator":                + backupGenerator,
                                     "Hydro Reservoir":                 + dispBackupGenerator
                                     }
    
        elif collectTerms == False: # Collecting into general terms
            if size == 60: # elec + transport
                collectedResponse = {"H2":                              + H2, 
                                     "Battery":                         + battery,
                                     "PHS":                             + PHS,
                                     "Import-Export":                   + importExport,
                                     "OCGT":                            + OCGT,
                                     "Hydro Reservoir":                 + hydro,
                                     "EV Battery":                      + EVBattery
                                     }
            elif size == 90: # elec + heating
                collectedResponse = {"H2":                              + H2, 
                                     "Battery":                         + battery,
                                     "PHS":                             + PHS,
                                     "Import-Export":                   + importExport,
                                     "OCGT":                            + OCGT,
                                     "Hydro Reservoir":                 + hydro,
                                     "Ground Heat Pump":                - groundHeatPump,
                                     "Central-Urban Heat Pump":         - centralUrbanHeatPump,
                                     "Resistive Heater":                - ResistiveHeater,
                                     "Central-Urban Resistive Heater":  - centralUrbanResistiveHeater,
                                     "CHP Electric":                    + CHPElec
                                     }
            elif size == 120: # elec + heating + transport
                collectedResponse = {"H2":                              + H2, 
                                     "Battery":                         + battery,
                                     "PHS":                             + PHS,
                                     "Import-Export":                   + importExport,
                                     "OCGT":                            + OCGT,
                                     "Hydro Reservoir":                 + hydro,
                                     "Ground Heat Pump":                - groundHeatPump,
                                     "Central-Urban Heat Pump":         - centralUrbanHeatPump,
                                     "Resistive Heater":                - ResistiveHeater,
                                     "Central-Urban Resistive Heater":  - centralUrbanResistiveHeater,
                                     "EV Battery":                      + EVBattery,
                                     "CHP Electric":                    + CHPElec
                                    }   
            else: # elec only
                collectedResponse = {"H2":                              + H2, 
                                     "Battery":                         + battery,
                                     "PHS":                             + PHS,
                                     "Import-Export":                   + importExport,
                                     "OCGT":                            + OCGT,
                                     "Hydro Reservoir":                 + hydro
                                    }
    if isBrownfield == True:
        if collectTerms == True: # Collecting into general terms
            collectedResponse = {"Storage":                         + storage, 
                                 "Import-Export":                   + importExport,
                                 "Backup Generator":                + backupGenerator,
                                 "Hydro Reservoir":                 + dispBackupGenerator,
                                 "Heat Couple":                     + heatCouple,
                                 "CHP Electric":                    + CHP
                                 }
        
        elif collectTerms == False: # Collecting into general terms
            collectedResponse = {"H2":                              + H2, 
                                 "Battery":                         + battery,
                                 "PHS":                             + PHS,
                                 "Import-Export":                   + importExport,
                                 "OCGT":                            + OCGT,
                                 "CCGT":                            + CCGT,
                                 "Coal":                            + coal,
                                 "Lignite":                         + lignite,
                                 "Nuclear":                         + nuclear,
                                 "Oil":                             + oil,
                                 "Biomass EOP":                     + biomassEOP,
                                 "Hydro Reservoir":                 + hydro,
                                 "Ground Heat Pump":                - groundHeatPump,
                                 "Central-Urban Heat Pump":         - centralUrbanHeatPump,
                                 "Resistive Heater":                - ResistiveHeater,
                                 "Central-Urban Resistive Heater":  - centralUrbanResistiveHeater,
                                 "Cooling Pump":                    - coolingPump,
                                 "Gas CHP Electric":                + gasCHPElec,
                                 "Biomass CHP Electric":            + biomassCHPElec,
                         }

    return collectedResponse

#%% ConValueGenerator

def ConValueGenerator(norm_const, dirc, eigen_vectors):
    """
    

    Parameters
    ----------
    norm_const : float64
        normilazation constant from PCA
    dirc : dirt
        a dictionary with the different generators in
    eigen_vectors : Array of float64
        eigen vectors from PCA

    Returns
    -------
    lambdaCollected : DataFrame
        Colelcted lambda values for each component

    """
    
    dictTypes = list(dirc.keys())
    
    moveToEnd = []
    types = []
    
    for j in range(len(dictTypes)):
        if dictTypes[j].split()[0] == "Load":
            moveToEnd.append(dictTypes[j])
        else:
            types.append(dictTypes[j])
  
    types.extend(moveToEnd)    
  
    for j in types:
    
        # Mean and centered
        centered = MeanAndCenter(dirc[j])
        
        # Projection
        projection = np.dot(centered,eigen_vectors)
    
        dirc[j] = projection  
    
    conNames = CovNameGenerator(types)
    
    amount = len(conNames)
    
    lambdaCollected =  pd.DataFrame( columns = conNames)
    
    for j in range(amount):
        if j < len(types):
            lambdaCollected[conNames[j]] = (norm_const**2)*(np.mean((dirc[types[j]]**2),axis=0))
        elif j < len(types)*2 -1:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[0]]*dirc[types[j+0-(len(types)*1-1)]]),axis=0))
        elif j < len(types)*3 -3:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[1]]*dirc[types[j+1-(len(types)*2-2)]]),axis=0))
        elif j < len(types)*4 -6:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[2]]*dirc[types[j+3-(len(types)*3-3)]]),axis=0))
        elif j < len(types)*5 -10:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[3]]*dirc[types[j+6-(len(types)*4-4)]]),axis=0))  
        elif j < len(types)*6 -15:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[4]]*dirc[types[j+10-(len(types)*5-5)]]),axis=0))  
        elif j < len(types)*7 -21:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[5]]*dirc[types[j+15-(len(types)*6-6)]]),axis=0))  
        elif j < len(types)*8 -28:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[6]]*dirc[types[j+21-(len(types)*7-7)]]),axis=0))  
        elif j < len(types)*9 -36:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[7]]*dirc[types[j+28-(len(types)*8-8)]]),axis=0))  
        elif j < len(types)*10 -45:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[8]]*dirc[types[j+36-(len(types)*9-9)]]),axis=0))  
        elif j < len(types)*11 -55:
            lambdaCollected[conNames[j]] = (norm_const**2)*2*(np.mean((dirc[types[9]]*dirc[types[j+45-(len(types)*10-10)]]),axis=0))     
            
        else:
            assert True, "something is wrong"
    
    return lambdaCollected

#%% HeatResponse

def HeatResponse(network, collectTerms=True):
    
    # Save index with only countries
    country_column = network.loads.index[:30]
    
    # Find efficency of links
    busEff = LinksEff(network)
    
    # Check if brownfield network
    isBrownfield = len([x for x in network.links_t.p0.columns if "nuclear" in x]) > 0
    
    # General Reponse
    waterTanks                  = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    centralUrbanWaterTanks      = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    GasBoiler                   = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    centralUrbanGasBoiler       = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    groundHeatPump              = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    centralUrbanHeatPump        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    ResistiveHeater             = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    centralUrbanResistiveHeater = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    CHPHeat                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
    
    if isBrownfield == True:
        biomassCHPHeat                     = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        HOP                                = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)
        coolingPump                        = pd.DataFrame(data=np.zeros([8760,30]),columns=country_column)

    for country in country_column:
        
        # Storage
        waterTanks[country] = (- network.links_t.p0.filter(regex=country).filter(regex=str(country)+" water tanks charger")
            + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" water tanks discharger").values
            * busEff.loc["water tanks discharger"].values[0]
            ).values 
        
        if network.links_t.p0.filter(regex=country).filter(regex="central water tanks charger").size > 0:
            centralUrbanWaterTanks[country] = (- network.links_t.p0.filter(regex=country).filter(regex="central water tanks charger").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            + network.links_t.p0.filter(regex=country).filter(regex="central water tanks discharger").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            * busEff.loc["water tanks discharger"].values[0]
                            ).values
        else:
            centralUrbanWaterTanks[country] = (- network.links_t.p0.filter(regex=country).filter(regex="urban water tanks charger").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            + network.links_t.p0.filter(regex=country).filter(regex="urban water tanks discharger").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            * busEff.loc["water tanks discharger"].values[0]
                            ).values
        
        # Backup generator
        if isBrownfield == False:
            GasBoiler[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" gas").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                       * busEff.loc["central gas boiler"].values[0]
                      ).values
        elif isBrownfield == True:
            GasBoiler[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" decentral gas")
                       * busEff.loc["central gas boiler"].values[0]
                      ).values
            if network.links_t.p0.filter(regex=country).filter(regex="HOP").size > 0:
                HOP[country] = ( + network.links_t.p0.filter(regex=country).filter(regex="HOP")
                                * busEff.loc["central biomass HOP"].values[0]
                                ).values
    
        if network.links_t.p0.filter(regex=country).filter(regex=" central gas").size > 0:
            centralUrbanGasBoiler[country] = (+ network.links_t.p1.filter(regex=country).filter(regex=" central gas boiler").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            #* busEff.loc["central gas boiler"].values[0]
                            * -1
                         ).values
        else:
            centralUrbanGasBoiler[country] = (+ network.links_t.p0.filter(regex=country).filter(regex="urban gas").groupby(network.links.bus0.str.slice(0,2), axis=1).sum()
                            * busEff.loc["central gas boiler"].values[0]
                         ).values
    
        # Sector links
        if isBrownfield == False:
            groundHeatPump[country] = ( - network.links_t.p1.filter(regex=country).filter(regex="ground heat pump").groupby(network.links.bus0.str.slice(0,2), axis=1).sum() ).values

        elif isBrownfield == True:
            groundHeatPump[country] = ( - network.links_t.p1.filter(regex=country).filter(regex="decentral heat pump").groupby(network.links.bus0.str.slice(0,2), axis=1).sum() ).values

        if (network.links_t.p1.filter(regex=country).filter(regex="central heat pump")).size > 0:
            centralUrbanHeatPump[country] = (- network.links_t.p1.filter(regex=country).filter(regex=" central heat pump")).values
        else:
            centralUrbanHeatPump[country] = (- network.links_t.p1.filter(regex=country).filter(regex="urban heat pump")).values
        
        if isBrownfield == False:
            ResistiveHeater[country] = ( + network.links_t.p0.filter(regex=country).filter(regex=str(country)+" resistive heater") 
                                    * (busEff.loc["central resistive heater"].values[0])
                                    ).values
        elif isBrownfield == True:
           ResistiveHeater[country] = ( + network.links_t.p1.filter(regex=country).filter(regex=str(country)+" decentral resistive heater") 
                                    #* (busEff.loc["decentral resistive heater"].values[0])
                                    * -1
                                    ).values
        if isBrownfield == False:
            if (network.links_t.p1.filter(regex=country).filter(regex=" central resistive heater")).size > 0:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex=" central resistive heater")
                                                        * (busEff.loc["central resistive heater"].values[0])
                                                        ).values
            else:
                centralUrbanResistiveHeater[country] = (network.links_t.p0.filter(regex=country).filter(regex="urban resistive heater")
                                                        * (busEff.loc["central resistive heater"].values[0])
                                                        ).values
        elif isBrownfield == True:
            if (network.links_t.p1.filter(regex=country).filter(regex=" central resistive heater")).size > 0:
                centralUrbanResistiveHeater[country] = (network.links_t.p1.filter(regex=country).filter(regex=" central resistive heater")
                                                        #* (busEff.loc["central resistive heater"].values[0])
                                                        * -1
                                                        ).values
        if isBrownfield == True:
            coolingPump[country] = ( - network.links_t.p1.filter(regex=country).filter(regex="cooling pump") ).values
                
                
                
        # CHP
        if isBrownfield == False:
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="CHP heat")).size > 0:
                CHPHeat[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central CHP heat").values
                       * busEff.loc["central CHP heat"].values[0]
                       )      
        elif isBrownfield == True:
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="gas CHP heat")).size > 0:
                CHPHeat[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central gas CHP heat").values
                       * busEff.loc["central gas CHP heat"].values[0]
                       )     
            if ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="biomass CHP heat")).size > 0:
                biomassCHPHeat[country] = ( + network.links_t.p0.filter(regex=str(country)+" ").filter(regex="central biomass CHP heat").values
                       * busEff.loc["central biomass CHP heat"].values[0]
                       )   

    if isBrownfield == False:
        storage = + waterTanks + centralUrbanWaterTanks.values
        backupGenerator = GasBoiler + centralUrbanGasBoiler.values
        elecCouple = + groundHeatPump + centralUrbanHeatPump.values + ResistiveHeater.values + centralUrbanResistiveHeater.values 
        CHP = CHPHeat
        
        if collectTerms == True: # Collecting into general terms
            collectedResponse = {"Storage":                         + storage,
                                 "Backup Generator":                + backupGenerator,
                                 "Electricity Couple":              + elecCouple,
                                 "CHP Heat":                        + CHP
                                 }
        
        elif collectTerms == False: # Collecting into general terms
            collectedResponse = {"Water Tanks":                     + waterTanks, 
                                 "Central-Urban Water Tanks":       + centralUrbanWaterTanks,
                                 "Gas Boiler":                      + GasBoiler,
                                 "Central-Urban Gas Boiler":        + centralUrbanGasBoiler,
                                 "Ground Heat Pump":                + groundHeatPump,
                                 "Central-Urban Heat Pump":         + centralUrbanHeatPump,
                                 "Resistive Heater":                + ResistiveHeater,
                                 "Central-Urban Resistive Heater":  + centralUrbanResistiveHeater,
                                 "CHP Heat":                        + CHPHeat
                                 }
            
    elif isBrownfield == True:
        storage = waterTanks + centralUrbanWaterTanks.values
        backupGenerator = GasBoiler + centralUrbanGasBoiler.values + HOP.values
        elecCouple = + groundHeatPump + centralUrbanHeatPump.values + ResistiveHeater.values + centralUrbanResistiveHeater.values + coolingPump.values
        CHP = CHPHeat + biomassCHPHeat.values
        
        if collectTerms == True: # Collecting into general terms
            collectedResponse = {"Storage":                         + storage,
                                 "Backup Generator":                + backupGenerator,
                                 "Electricity Couple":              + elecCouple,
                                 "CHP Heat":                        + CHP
                                 }
        
        elif collectTerms == False: # Collecting into general terms
            collectedResponse = {"Water Tanks":                     + waterTanks,
                                 "Central Water Tanks":             + centralUrbanWaterTanks, 
                                 "Decentral Gas Boiler":            + GasBoiler,
                                 "Central Gas Boiler":              + centralUrbanGasBoiler,
                                 "Biomass HOP":                     + HOP,
                                 "Decentral Heat Pump":             + groundHeatPump,
                                 "Central Heat Pump":               + centralUrbanHeatPump,
                                 "Decentral Resistive Heater":      + ResistiveHeater,
                                 "Central Resistive Heater":        + centralUrbanResistiveHeater,
                                 "Cooling Hump":                    + coolingPump,
                                 "Gas CHP Heat":                    + CHPHeat,
                                 "Biomass CHP Heat":                + biomassCHPHeat
                                 }
            
                   

    return collectedResponse

#%% CovNames

def CovNames(dirc1, dirc2):
    
    # Convert to list
    covNames1 = list(dirc1.keys())
    covNames2 = list(dirc2.keys())
    
    # Determin size
    covSize = len(covNames1) * len(covNames2)
    
    covNames = pd.DataFrame(np.empty((covSize, 3), dtype = np.str), columns = ["Contribution","Response","Covariance"])
    
    # index counter
    i = 0
    
    for j in covNames1: # List of dirc1 names
        for k in covNames2: # List of dirc2 names
            
            # Save names
            covNames["Contribution"][i] = j
            covNames["Response"][i] = k
            covNames["Covariance"][i] = j + "/\n" + k
    
            # add one to index counter
            i += 1

    return covNames

#%% CovValueGenerator

def CovValueGenerator(dirc1, dirc2, meanAndCenter,normConst, eigenVectors):

    # Generate name list
    covNames = CovNames(dirc1, dirc2)
    
    # Empty dataframe
    covMatrix = pd.DataFrame(np.zeros([len(covNames),30]), index = covNames["Covariance"])
    
    if meanAndCenter == False: # If values have not been meanséd and cented before
        # Mean, center and projection of 1. dirc
        for j in list(dirc1.keys()):
        
            # Mean and centered
            centered = MeanAndCenter(dirc1[j])
            
            # Projection
            projection = np.dot(centered,eigenVectors)
        
            # Save value agian
            dirc1[j] = projection
        
        # Mean, center and projection of 2. dirc
        for j in list(dirc2.keys()):
        
            # Mean and centered
            centered = MeanAndCenter(dirc2[j])
            
            # Projection
            projection = np.dot(centered,eigenVectors)
        
            # Save value agian
            dirc2[j] = projection
            
    # Main calulation
    for i in range(len(covNames)):
        
        # Calculate covariance
        covMatrix.loc[covNames["Covariance"][i]] = - (normConst**2)*(np.mean((dirc1[covNames["Contribution"][i]]*dirc2[covNames["Response"][i]]),axis=0))

    return covMatrix

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
    
    
    
    
    
    
    