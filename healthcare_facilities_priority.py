#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as ctx
import os, sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D  # for legend handle
from matplotlib_scalebar.scalebar import ScaleBar
from sklearn.metrics.pairwise import haversine_distances

#External package from:
#https://github.com/yquilcaille/FWI_CMIP6/blob/main/functions_support.py
sys.path.append(os.path.join(os.path.dirname("src/")))
import functions_support as fsupport

import warnings 
from shapely import wkt



def healthcare_prioritization(df_exposed_population, #Read exposed population
                              df_health_infrastructure, #Read exposed healthcare infrastructure
                              df_danger_variables, #Read danger (flood, landslides, etc) variables
                              factor_variables #List of variables to take into account in @df_danger_variables
                             ):
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #
    #  DATA PREPARTION
    #
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    #  Compute proportion of total and exposed population
    df_exposed_population['total pop'] = df_exposed_population['Exposed Pop'] + df_exposed_population['Not Exposed Pop']
    df_exposed_population['Exposed Pop'] = df_exposed_population['Exposed Pop'] / df_exposed_population['total pop']
    df_exposed_population = df_exposed_population[['Administration 3 Code', 'Exposed Pop', 'Not Exposed Pop','total pop']]

    #Subsamplig the columns to take into account
    df_danger_variables = df_danger_variables[factor_variables]
    #Verifying NaN  THERE SHOULD NOT BE NAN NUMBERS
    count_na = df_health_infrastructure.isna().sum().sum()
    if (count_na > 0):
        # warnings.warn(f"WARNING! the health infrstructure information hss NaN. Please verify for consistency of results. We filll all NA as zero.")
        df_health_infrastructure.fillna(0,inplace=True)
        # df_health_infrastructure.replace(math.nan, 0, inplace=True)
    
    #Cast to int
    df_health_infrastructure['Administration 1 Code'] = df_health_infrastructure['Administration 1 Code'].astype(int) 
    df_health_infrastructure['Administration 2 Code'] = df_health_infrastructure['Administration 2 Code'].astype(int) 
    df_health_infrastructure['Administration 3 Code'] = df_health_infrastructure['Administration 3 Code'].astype(int) 
    
    #################################################
    #
    #         THIS MUST BE GENERALIZED !!!!!!!
    #
    #################################################
    #replace danger column
    danger = df_health_infrastructure.columns[-1]
    # print(df_health_infrastructure.head())
    df_health_infrastructure[danger].replace("not_floods", 0, inplace=True)
    df_health_infrastructure[danger].replace("floods", 1, inplace=True)
    ##################################################
    
    #Adding variables to health information 
    df_health_infrastructure['Exposed Pop'] =  df_health_infrastructure['Administration 3 Code'].map(df_exposed_population.set_index('Administration 3 Code')['Exposed Pop'])
    df_health_infrastructure['Per Indigenous Pop'] =  df_health_infrastructure['Administration 3 Code'].map(df_danger_variables.set_index('Administration 3 Code')['Per Indigenous Pop'])
    df_health_infrastructure['Per Pobrezanbi'] =  df_health_infrastructure['Administration 3 Code'].map(df_danger_variables.set_index('Administration 3 Code')['Per Pobrezanbi'])
    df_health_infrastructure['Per Desnutricion'] =  df_health_infrastructure['Administration 3 Code'].map(df_danger_variables.set_index('Administration 3 Code')['Per Desnutricion'])
    df_health_infrastructure['ValorConcentracióndeEESSdel'] =  df_health_infrastructure['Administration 3 Code'].map(df_danger_variables.set_index('Administration 3 Code')['ValorConcentracióndeEESSdel'])
    df_health_infrastructure['MEDICOX10000HABITANTENORMAL'] =  df_health_infrastructure['Administration 3 Code'].map(df_danger_variables.set_index('Administration 3 Code')['MEDICOX10000HABITANTENORMAL'])
    df_health_infrastructure['Per Ess 100hab'] =  df_health_infrastructure['Administration 3 Code'].map(df_danger_variables.set_index('Administration 3 Code')['Per Ess 100hab'])
    df_health_infrastructure['PHC Dist Km'] =  df_health_infrastructure['Administration 3 Code'].map(df_danger_variables.set_index('Administration 3 Code')['PHC Dist Km'])
    df_health_infrastructure['Hospital Dist Km'] =  df_health_infrastructure['Administration 3 Code'].map(df_danger_variables.set_index('Administration 3 Code')['Hospital Dist Km'])

    return df_health_infrastructure
   



######################################################################
#
# This method plots PHC and Hospitals  natioal prioritization
#
######################################################################
def plot_natioal_prioritization(df, 
                                colors, #disctionaty with color codes
                               country, #String of the counry analysis  
                               level,   #String of the analysis level PHC or Hospital
                               shock, #string of the shock name (floods, snow, etc)
                               directory_path,
                               region_shape = pd.DataFrame()):
    fig, ax = plt.subplots(figsize=(14, 14))
    
    if country == 'colombia':
        ax.set_xlim([-82.0, -66.0])  # type: ignore # Límites en la coordenada X
        ax.set_ylim([-4.9, 14.0])  # type: ignore # Límites en la coordenada Y
    else:  
        ax.set_xlim([-82.0, -68.0])  # type: ignore # Límites en la coordenada X
        ax.set_ylim([-19.0, 0.1])  # type: ignore # Límites en la coordenada Y
    ax.axis("off")


    markersizes = {1: 25, 2: 15, 3: 20}
    markersizes = {1: 40, 2: 30, 3: 20}

    plot = df.plot(
        ax=ax,
        c=df["Nivel"].map(colors),
        markersize=df["Nivel"].map(markersizes)
    )

    A = [
        -70.1 * np.pi / 180.0,
        -2.5 * np.pi / 180.0,
    ]  # set latitude & longitude of interest here
    B = [
        -70.8 * np.pi / 180.0,
        -2.5 * np.pi / 180.0,
    ]  # set latitude & longitude (longitude in A +1) of interest here
    dx = (6371000) * haversine_distances([A, B])[0, 1]
    ax.add_artist(ScaleBar(dx=dx, units="m"))
    
    if not region_shape.empty:
        region_shape.plot(ax=plot, facecolor="none", edgecolor="gray")

    # add a legend
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=v,
            label=f"Category {fsupport.category_dict[k]}",
            markersize=12,
        )
        for k, v in colors.items()
    ]
    ax.legend(
        title="Health\ninfrastructure",
        handles=handles,
        bbox_to_anchor=(0.83, 0.96),
        loc="upper left",
    )
    ctx.add_basemap(
        plot, crs="epsg:4326", source=ctx.providers.Stamen.TonerBackground, attribution=""
    )
    plt.savefig(f"{directory_path}/{country}_{level}_priority_{shock}.png", format="png", dpi=150, bbox_inches="tight")




######################################################################
#
# This method split PHC and Hospitals for natioal prioritization
#
######################################################################


def prioritization_national_level(df_health_infrastructure, # This is the dataframe outputed from healthcare_prioritization
                                  variables_for_prioritization, # list of variables between 0 to 1 characterizing poverty for prioritization
                                 top, # Number of top hospitals to prioritize (By default top 10 most in danger hospitals to prioritize)
                                 country, #String of the counry analysis  
                                 shock, #string of the shock name (floods, snow, etc)
                                 directory_path = ""):
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #
    #  SPLITING DATA FOR SPARATE TREATEMENT OF PHC & HOSPITALS
    #
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Filtering exposed healthcare infrstructure
    df_exposed_health_infra = df_health_infrastructure[df_health_infrastructure['Shock Name'] == 1]
    df_exposed_health_infra = df_exposed_health_infra.reset_index(drop=True)
    # Getting PHC
    df_exposed_health_infra_phc = df_exposed_health_infra[df_exposed_health_infra['Nivel'] == 1 ]
    df_exposed_health_infra_phc = df_exposed_health_infra_phc.reset_index(drop=True)
    #Getting Hospitals
    df_exposed_health_infra_hospital = df_exposed_health_infra[df_exposed_health_infra['Nivel'] != 1 ]
    df_exposed_health_infra_hospital = df_exposed_health_infra_hospital.reset_index(drop=True)
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #
    #  ## Prioritization at national level ##
    #
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    #Computing the average of prioritization variables
    df_exposed_health_infra_phc['mean'] = df_exposed_health_infra_phc.loc[:,variables_for_prioritization].mean(axis=1)
    df_exposed_health_infra_hospital['mean'] = df_exposed_health_infra_hospital.loc[:,variables_for_prioritization].mean(axis=1)

    variables_to_keep = ['Hospital','Name','Nivel','administration_1','administration_2','administration_3','PHC Dist Km','mean', 'geometry']
    
    # PHC
    print("PHC processing...")
    df_phc_plot = df_exposed_health_infra_phc[variables_to_keep].sort_values(by=['mean','PHC Dist Km'], ascending = [False , False]).head(top)
    df_phc_plot = gpd.GeoDataFrame(df_phc_plot)
    # df_phc_plot['geometry'] = df_phc_plot['geometry'].apply(wkt.loads)
    colors = {1: "blue"}
    plot_natioal_prioritization(df_phc_plot,colors,country,"PHC",shock, directory_path)
    df_phc_plot.to_csv(f"{directory_path}/{country}_PHC_priority_{shock}.csv")


    print("Hospital processing...")
    df_hospital_plot = df_exposed_health_infra_hospital[variables_to_keep].sort_values(by=['mean','PHC Dist Km'], ascending = [False , False]).head(top)
    df_hospital_plot = gpd.GeoDataFrame(df_hospital_plot)
    # df_hospital_plot['geometry'] = df_hospital_plot['geometry'].apply(wkt.loads)
    colors = {2: "orange",3: "red",}
    plot_natioal_prioritization(df_hospital_plot,colors,country,"Hospitals",shock, directory_path)
    df_hospital_plot.to_csv(f"{directory_path}/{country}_Hospitals_priority_{shock}.csv")



######################################################################
#
# This method plots PHC and Hospitals  natioal prioritization
#
######################################################################
def plot_by_department_prioritization(df, 
                               country, #String of the counry analysis  
                               level,   #String of the analysis level PHC or Hospital
                               shock, #string of the shock name (floods, snow, etc)
                               directory_path = "",
                               region_shape = pd.DataFrame()):
    fig, ax = plt.subplots(figsize=(14, 14))
    
    if country == 'colombia':
        ax.set_xlim([-82.0, -66.0])  # type: ignore # Límites en la coordenada X
        ax.set_ylim([-4.9, 14.0])  # type: ignore # Límites en la coordenada Y
    else:  
        ax.set_xlim([-82.0, -68.0])  # type: ignore # Límites en la coordenada X
        ax.set_ylim([-19.0, 0.1])  # type: ignore # Límites en la coordenada Y

    ax.axis("off")
    colors = {
        1: "red",
        2: "orange",
        3: "green"
    }

    markersizes = {1: 40, 2: 30, 3: 20}

    plot = df.plot(
        ax=ax,
        c=df["priority"].map(colors),
        markersize=df["priority"].map(markersizes)
    )

    A = [
        -70.1 * np.pi / 180.0,
        -2.5 * np.pi / 180.0,
    ]  # set latitude & longitude of interest here
    B = [
        -70.8 * np.pi / 180.0,
        -2.5 * np.pi / 180.0,
    ]  # set latitude & longitude (longitude in A +1) of interest here
    dx = (6371000) * haversine_distances([A, B])[0, 1]
    ax.add_artist(ScaleBar(dx=dx, units="m"))
    
    if not region_shape.empty:
        region_shape.plot(ax=plot, facecolor="none", edgecolor="gray")

    # add a legend
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=v,
            label=f"Priority {k}",
            markersize=12,
        )
        for k, v in colors.items()
    ]
    
    infra_title = ""
    print(infra_title)
    if (level == 'PHC'):
        infra_title="PHC prioritization"
    else :
        infra_title="Hospital prioritization"
    
    ax.legend(
        title=infra_title, 
        # title="PHC prioritization",
        handles=handles,
        bbox_to_anchor=(0.83, 0.96),
        loc="upper left",
    )
    ctx.add_basemap(
        plot, crs="epsg:4326", source=ctx.providers.Stamen.TonerBackground, attribution=""
    )
    plt.savefig(f"{directory_path}/{country}_{level}_by_departament_priority_{shock}.png", format="png", dpi=150, bbox_inches="tight")




######################################################################
#
# This method split PHC and Hospitals for by department prioritization
#
######################################################################


def prioritization_by_department_level(df_health_infrastructure, # This is the dataframe outputed from healthcare_prioritization
                                  variables_for_prioritization, # list of variables between 0 to 1 characterizing poverty for prioritization
                                 country, #String of the counry analysis  
                                 shock, #string of the shock name (floods, snow, etc)
                                 directory_path = ""):
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #
    #  SPLITING DATA FOR SPARATE TREATEMENT OF PHC & HOSPITALS
    #
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Filtering exposed healthcare infrstructure
    df_exposed_health_infra = df_health_infrastructure[df_health_infrastructure["Shock Name"] == 1]
    df_exposed_health_infra = df_exposed_health_infra.reset_index(drop=True)
    # Getting PHC
    df_exposed_health_infra_phc = df_exposed_health_infra[df_exposed_health_infra['Nivel'] == 1 ]
    df_exposed_health_infra_phc = df_exposed_health_infra_phc.reset_index(drop=True)
    #Getting Hospitals
    df_exposed_health_infra_hospital = df_exposed_health_infra[df_exposed_health_infra['Nivel'] != 1 ]
    df_exposed_health_infra_hospital = df_exposed_health_infra_hospital.reset_index(drop=True)
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #
    #  ## Prioritization by level ##
    #
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    #Computing the average of prioritization variables
    df_exposed_health_infra_phc['mean'] = df_exposed_health_infra_phc.loc[:,variables_for_prioritization].mean(axis=1)
    df_exposed_health_infra_hospital['mean'] = df_exposed_health_infra_hospital.loc[:,variables_for_prioritization].mean(axis=1)
    
    #variables_to_keep = ['Hospital','Name','Nivel','geometry','Administration 1 Code','Administration 3 Code','PHC Dist Km','mean']
    variables_to_keep = ['Hospital','Name','Nivel', 'administration_1','administration_2','administration_3','PHC Dist Km','mean', 'geometry','Administration 1 Code']

    # PHC
    print("PHC processing...")
    df_phc_plot = df_exposed_health_infra_phc[variables_to_keep].sort_values(by=['administration_1','mean','PHC Dist Km'], ascending = [True, False , False])
    df_phc_plot = gpd.GeoDataFrame(df_phc_plot)
    # df_phc_plot['geometry'] = df_phc_plot['geometry'].apply(wkt.loads)
    df_phc_plot['priority'] = 0
    #Prioritizing
    df_phc_plot['priority'] = df_phc_plot.groupby("administration_1").cumcount() + 1
    df_phc_plot = df_phc_plot[['Hospital','Name','Nivel','priority', 'administration_1','administration_2','administration_3','PHC Dist Km','mean', 'geometry','Administration 1 Code']]
    df_phc_plot = df_phc_plot.sort_values(by=['administration_1','mean','PHC Dist Km'], ascending = [True, False , False]).groupby('administration_1').head(3)
    #plot
    plot_by_department_prioritization(df_phc_plot,country,"PHC",shock, directory_path)
    df_phc_plot.to_csv(f"{directory_path}/{country}_PHC_by_department_priority_{shock}.csv")
    

    print("Hospital processing...")
    df_hospital_plot = df_exposed_health_infra_hospital[variables_to_keep].sort_values(by=['administration_1','mean','PHC Dist Km'], ascending = [True, False , False])
    df_hospital_plot = gpd.GeoDataFrame(df_hospital_plot)
    # df_hospital_plot['geometry'] = df_hospital_plot['geometry'].apply(wkt.loads)
    df_hospital_plot['priority'] =0
    #Prioritizing
    df_hospital_plot['priority'] = df_hospital_plot.groupby("administration_1").cumcount() + 1
    df_hospital_plot = df_hospital_plot[['Hospital','Name','Nivel','priority', 'administration_1','administration_2','administration_3','PHC Dist Km','mean', 'geometry','Administration 1 Code']]
    df_hospital_plot = df_hospital_plot.sort_values(by=['administration_1','mean','PHC Dist Km'], ascending = [True, False , False]).groupby('administration_1').head(3)
    #plot
    plot_by_department_prioritization(df_hospital_plot,country,"Hospitals",shock, directory_path)
    df_hospital_plot.to_csv(f"{directory_path}/{country}_Hospitals_by_department_priority_{shock}.csv")
    
    return df_phc_plot, df_hospital_plot

    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#  COLOMBIA
#
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def col_healthcare_prioritization(df_exposed_population, #Read exposed population
                              df_health_infrastructure, #Read exposed healthcare infrastructure
                              df_danger_variables, #Read danger (flood, landslides, etc) variables
                              factor_variables #List of variables to take into account in @df_danger_variables
                             ):
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #
    #  DATA PREPARTION
    #
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    #  Compute proportion of total and exposed population
    df_exposed_population['total pop'] = df_exposed_population['Exposed Pop'] + df_exposed_population['Not Exposed Pop']
    df_exposed_population['Exposed Pop'] = df_exposed_population['Exposed Pop'] / df_exposed_population['total pop']
    df_exposed_population = df_exposed_population[['Administration 2 Code', 'Exposed Pop', 'Not Exposed Pop','total pop']]

    count_na = df_health_infrastructure.isna().sum().sum()
    if (count_na > 0):
        # warnings.warn(f"WARNING! the health infrstructure information hss NaN. Please verify for consistency of results. We filll all NA as zero.")
        df_health_infrastructure.fillna(0,inplace=True)
        # df_health_infrastructure.replace(math.nan, 0, inplace=True)
    
    #################################################
    #
    #         THIS MUST BE GENERALIZED !!!!!!!
    #
    #################################################
    
    #Adding variables to health information 
    # df_health_infrastructure['Exposed Pop'] =  df_health_infrastructure['Administration 3 Code'].map(df_exposed_population.set_index('Administration 3 Code')['Exposed Pop'])
    # # df_health_infrastructure['Per Pobrezanbi'] =  df_health_infrastructure['Administration 3 Code'].map(df_danger_variables.set_index('Administration 3 Code')['Per Pobrezanbi'])
    # df_health_infrastructure['PHC Dist Km'] =  df_health_infrastructure['Administration 3 Code'].map(df_danger_variables.set_index('Administration 3 Code')['PHC Dist Km'])

    return df_health_infrastructure