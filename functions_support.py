import os
import sys

import contextily as ctx
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import shapefile as shp
from matplotlib.lines import Line2D  # for legend handle
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import Point, Polygon
from sklearn.metrics.pairwise import haversine_distances

global category_dict
category_dict = {
    1: "I",
    2: "II",
    3: "III",
    "1": "I",
    "2": "II",
    "3": "III",
}

global departamento_name_dict
departamento_name_dict = {
    "Antioquia": "ANTIOQUIA",
    "Bogotá, D.C.": "BOGOTÁ, D.C.",
    "Bogotá D.C": "BOGOTÁ, D.C.",  # new
    "Boyacá": "BOYACÁ",
    "Cundinamarca": "CUNDINAMARCA",
    "Meta": "META",
    "Atlántico": "ATLÁNTICO",
    "Magdalena": "MAGDALENA",
    "Caldas": "CALDAS",
    "Risaralda": "RISARALDA",
    "Valle del Cauca": "VALLE DEL CAUCA",
    "Santander": "SANTANDER",
    "Bolívar": "BOLÍVAR",
    "Nariño": "NARIÑO",
    "Cesar": "CESAR",
    "Norte de Santander": "NORTE DE SANTANDER",
    "Sucre": "SUCRE",
    "Cauca": "CAUCA",
    "Valle del cauca": "VALLE DEL CAUCA",  # new
    "Huila": "HUILA",
    "Caquetá": "CAQUETÁ",
    "La Guajira": "LA GUAJIRA",
    "Quindío": "QUINDIO",
    "Tolima": "TOLIMA",
    "Arauca": "ARAUCA",
    "Córdoba": "CÓRDOBA",
    "Casanare": "CASANARE",
    "Putumayo": "PUTUMAYO",
    "Chocó": "CHOCÓ",
    "Guaviare": "GUAVIARE",
    "Vichada": "VICHADA",
    "Amazonas": "AMAZONAS",
    "Vaupés": "VAUPÉS",
    "San Andrés y Providencia": "ARCHIPIÉLAGO DE SAN ANDRÉS, PROVIDENCIA Y SANTA CATALINA",  # new
    "Barranquilla": "ATLÁNTICO",  # new
    "Buenaventura": "VALLE DEL CAUCA",  # new
    "Cali": "VALLE DEL CAUCA",  # new
    "Cartagena": "BOLÍVAR",  # new
    "Guainía": "GUAINÍA",  # new
    "Santa Marta": "MAGDALENA",  # new
}


def departamento_name_func(x):
    try:
        return departamento_name_dict[x]
    except KeyError:
        return ""


def get_color_scale(df, col_name):
    norm = mcolors.Normalize(
        vmin=min(df[col_name].dropna()), vmax=max(df[col_name].dropna())
    )
    return norm


def get_country_maps(name="peru"):
    """
    Getting shape of some country from osmnx library

    Parameters:
     - name: name of the country
    """
    shape = ox.geocode_to_gdf(name)
    shape = shape.to_crs("epsg:4326")

    return shape


def extract_centroid_if_exist(geo_df):
    new_geo_df = geo_df.copy()
    new_geo_df["geometry"] = new_geo_df["geometry"].apply(
        lambda x: x.centroid if isinstance(x, Polygon) else x
    )
    new_geo_df = new_geo_df.to_crs("EPSG:4326")
    return new_geo_df


def hosp_join_points(df, df_polygons):
    """
    Takes in two dataframes: df and df_polygons. It generates a new GeoDataFrame called geo_df with a new geometry.
    """
    geometry = [Point(xy) for xy in zip(df["lat"], df["lon"])]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry)
    geo_df.set_crs(epsg=4326, inplace=True)
    geo_df = geo_df.to_crs(epsg=3857)
    df_polygons = df_polygons.to_crs(epsg=3857)
    data_temp = gpd.sjoin(geo_df, df_polygons, how="left", predicate="within")
    data_temp = data_temp.to_crs("EPSG:4326")
    del data_temp["index_right"]
    del data_temp["lat"]
    del data_temp["lon"]

    return data_temp


def pop_join_nearest_points(
    df_point, df_polygons, how="left", save_result=False, path=None
):
    """
    This function assigns a polygon (df_polygons) to every point (df_point),
    if the point does not in some polygon the nearest polygons will be assigned
    to that point

    """
    df_polygons["region"] = df_polygons.geometry
    df_point = df_point.to_crs(epsg=3857)
    df_polygons = df_polygons.to_crs(epsg=3857)
    data_temp = gpd.sjoin(df_point, df_polygons, how=how, predicate="within")

    data_resto = data_temp[data_temp["region"].isna()][
        ["fid", "geometry", "population"]
    ].reset_index(drop=True)

    data_resto = gpd.sjoin_nearest(
        data_resto, df_polygons, how=how, distance_col="distances"
    )
    # data_resto = gpd.sjoin_nearest(data_resto, df_polygons, how=how)

    data_temp = data_temp[~(data_temp["region"].isna())].reset_index(drop=True)
    data_temp["distances"] = 0

    data_temp = pd.concat([data_temp, data_resto]).sort_values(by="fid", ascending=True)
    data_temp = data_temp.to_crs(epsg=4326)  # type: ignore
    del data_temp["index_right"]
    del data_temp["distances"]

    if save_result:
        data_temp_sub = data_temp.copy()
        data_temp_sub = rename_columns(data_temp_sub)
        data_temp_sub.to_csv(path, index=False)

    return data_temp


def shock_shape(path, shock_name="shock_name"):
    """
    Read shape of shock phenomenon and conver in a unique global polygon.

    Parameters:
        - path: path from shape file (.shp)
    """
    shape = gpd.read_file(path, crs=4326)
    shape = shape[shape["geometry"].notna()]
    shape["geometry"] = shape["geometry"].buffer(0)
    union = shape["geometry"].unary_union
    shape = gpd.GeoDataFrame({"geometry": [union]})
    # shape[shock_name+'_polygon'] = shape.geometry
    shape.crs = 4326
    # shape[shock_name] = shock_name
    shape["shock_name"] = 1
    
    return shape


def plot_shock_maps(
    country,
    shock,
    country_name="COUNTRY_NAME",
    shock_name="SHOCK_NAME",
    scale=True,
    save_result=False,
    path="PATH_FIG",
    show=False,
):
    """
    Parameters:

    """

    fig, ax = plt.subplots(figsize=(14, 10))
    if country_name == "colombia":
        ax.set_xlim([-82.0, -66.0])  # type: ignore # Límites en la coordenada X
        ax.set_ylim([-4.9, 14.0])  # type: ignore # Límites en la coordenada Y
    if country_name == "peru":
        ax.set_xlim([-82.0, -68.0])  # type: ignore # Límites en la coordenada X
        ax.set_ylim([-19.0, 0.1])  # type: ignore # Límites en la coordenada Y
    ax.axis("off")

    plot = country.plot(ax=ax, marker=".", color="gray", markersize=1, alpha=0.2)
    shock_areas = shock.plot(ax=ax, marker=".", color="green", markersize=1)

    ctx.add_basemap(
        plot,
        crs=country.crs.to_string(),
        source=ctx.providers.Stamen.TonerBackground,
        attribution="",
    )

    # Legend
    color_dots = Line2D(
        [0],
        [0],
        marker="o",
        color="green",
        markerfacecolor="green",
        label=shock_name,
        markersize=8,
        linewidth=0,
    )
    ax.legend(
        title="", handles=[color_dots], bbox_to_anchor=(0.05, 0.025), loc="lower left"
    )
    # ctx.add_basemap(plot, crs = 'epsg:4326', source = ctx.providers.Stamen.TonerBackground, attribution = '')

    if scale:
        # set latitude & longitude of interest here
        A = [-70.1 * np.pi / 180.0, -2.5 * np.pi / 180.0]
        # set latitude & longitude (longitude in A +1) of interest here
        B = [-70.8 * np.pi / 180.0, -2.5 * np.pi / 180.0]
        dx = (6371000) * haversine_distances([A, B])[0, 1]
        ax.add_artist(ScaleBar(dx=dx, units="m"))

    if save_result:
        path = path
        print(" " + path[2:])
        plt.savefig(path, format="png", dpi=150, bbox_inches="tight")

    if not show:
        plt.close(fig)
    return True


def merge_population_shock_shape(population_gdf, shock_shape, shock_name = "shock_name"):
    """
    Each populted point is assigned to scho shape
    """
    data_temp = gpd.sjoin(population_gdf, shock_shape, how="left", predicate="within")
    del data_temp["index_right"]
    data_temp[shock_name] = data_temp[shock_name].replace(np.nan, 0)
    data_temp[shock_name] = data_temp[shock_name].astype(int)
    # print(len(data_temp))

    return data_temp


def shock_population_affected_per_region(
    data,
    shock_name=None,
    adm_region="Administration 1",
    shape_region=None,
    save_result=False,
    path="",
    country_name="COUNTRY_NAME",
):
    """
    group population by region, keeping shock-affected population

    Parameters:

    """
    adm_name = ""
    columns = []
    if adm_region == "Administration 1":
        columns = ["administration_1_code"]
        adm_name = "1"
    elif adm_region == "Administration 2":
        columns = ["administration_1_code", "administration_2_code"]
        adm_name = "2"
    elif adm_region == "Administration 3":
        columns = [
            "administration_1_code",
            "administration_2_code",
            "administration_3_code",
        ]
        adm_name = "3"
    else:
        adm_name = ""

    data_temp = (
        data.groupby(columns + ["shock_name"])
        .agg(population=("population", "sum"))
        .reset_index()
    )
    data_temp = data_temp.pivot(index=columns, columns="shock_name")
    data_temp = data_temp.reset_index()

    data_temp.columns = data_temp.columns.to_flat_index()
    data_temp = data_temp.rename(
        columns={
            ("administration_1_code", ""): "administration_1_code",
            ("administration_2_code", ""): "administration_2_code",
            ("administration_3_code", ""): "administration_3_code",
            ("population", 0): "not_exposed_pop",
            ("population", 1): "exposed_pop",
        }
    )
    data_temp = data_temp[columns + ["exposed_pop", "not_exposed_pop"]]
    data_temp = data_temp.replace(np.nan, 0)

    data_temp["percentage"] = round(
        data_temp["exposed_pop"]
        / (data_temp["exposed_pop"] + data_temp["not_exposed_pop"])
        * 100,
        2,
    )
    data_temp = data_temp.sort_values(by="percentage", ascending=False)

    if save_result:
        data_temp_sub = data_temp.copy()
        data_temp_sub_columns = data_temp_sub.columns # jkn eliminate this
        data_temp_sub = pd.merge(# jkn eliminate this
            data_temp_sub, shape_region, how="left", left_on=columns, right_on=columns# jkn eliminate this
        )# jkn eliminate this
        del data_temp_sub['geometry']
        # data_temp_sub = data_temp_sub[["Administration_1"] + data_temp_sub_columns]# jkn eliminate this
        # path = path+'/'+country_name+'_'+shock_name+'_population_adm'+adm_name+'_table.csv'
        path = f"{path}/{country_name}_exposed_population_{shock_name}_adm{adm_name}_table.csv"
        data_temp_sub = rename_columns(data_temp_sub)
        data_temp_sub.to_csv(path, index=False)
        del data_temp_sub

    # Area afectada
    data_temp = pd.merge(
        data_temp, shape_region, how="left", left_on=columns, right_on=columns
    )

    data_temp = gpd.GeoDataFrame(data_temp, geometry="geometry")
    data_temp.crs = 4326

    return data_temp


def plot_regions_intensity_population(
    gdf,
    column_name,
    transform_column=0,
    shock_name="SHOCK_NAME",
    adm_region="administration_i",
    save_result=False,
    path="",
    country_name="COUNTRY_NAME",
    show=False,
):
    """
    Plot map intensity
    """

    cmap_color = "Blues"
    leg_color = "Blue"

    adm_name = ""
    if adm_region == "Administration 1":
        adm_name = "1"
    elif adm_region == "Administration 2":
        adm_name = "2"
    elif adm_region == "Administration 3":
        adm_name = "3"
    else:
        adm_name = ""

    trasformed_column = "temporal_column"

    # # gdf[trasformed_column] = gdf[column_name]
    # gdf[trasformed_column] = np.log(
    #     gdf[column_name]) / np.log(transform_column)

    # # gdf[trasformed_column] = np.log10(gdf[column_name])

    if transform_column < 0:
        gdf[trasformed_column] = gdf[column_name].apply(
            lambda x: pow(x, 1 / -1 * transform_column)
        )
    elif transform_column > 0:
        gdf[trasformed_column] = gdf[column_name].apply(
            lambda x: pow(x, transform_column)
        )
    else:
        gdf[trasformed_column] = gdf[column_name]

    fig, ax = plt.subplots(figsize=(14, 10))
    if country_name == "colombia":
        ax.set_xlim([-82.0, -66.0])  # type: ignore # Límites en la coordenada X
        ax.set_ylim([-4.9, 14.0])  # type: ignore # Límites en la coordenada Y
    if country_name == "peru":
        ax.set_xlim([-82.0, -68.0])  # type: ignore # Límites en la coordenada X
        ax.set_ylim([-19.0, 0.1])  # type: ignore # Límites en la coordenada Y
    ax.axis("off")

    norm = get_color_scale(gdf, trasformed_column)

    plot = gdf.plot(ax=ax, column=trasformed_column, cmap=cmap_color)
    ctx.add_basemap(
        plot,
        crs="epsg:4326",
        source=ctx.providers.Stamen.TonerBackground,
        attribution="",
    )

    # set latitude & longitude of interest here
    A = [-70.1 * np.pi / 180.0, -2.5 * np.pi / 180.0]
    # set latitude & longitude (longitude in A +1) of interest here
    B = [-70.8 * np.pi / 180.0, -2.5 * np.pi / 180.0]
    dx = (6371000) * haversine_distances([A, B])[0, 1]
    ax.add_artist(ScaleBar(dx=dx, units="m"))

    ticks = np.linspace(
        gdf[trasformed_column].min(), gdf[trasformed_column].max(), num=5, dtype=float
    )

    labels = np.linspace(
        gdf[column_name].min(), gdf[column_name].max(), num=5, dtype=int
    )
    labels = [f"{x}%" for x in labels]

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.05,
            ax.get_position().y0,
            0.01,
            ax.get_position().height,
        ]
    )
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap_color),
        cax=cax,
        orientation="vertical",
        label="",
        ticks=ticks,
    )

    cbar.ax.set_yticklabels(labels)

    # Legend
    red_dots = Line2D(
        [0],
        [0],
        marker="o",
        color=leg_color,
        markerfacecolor=leg_color,
        label="Percent of exposed population",
        markersize=5,
        linewidth=0,
    )
    ax.legend(
        title="", handles=[red_dots], bbox_to_anchor=(0.05, 0.02), loc="lower left"
    )

    if save_result:
        # path = path+'/'+country_name+'_'+shock_name + \
        #     '_population_adm'+adm_name+'_map.png'
        path = f"{path}/{country_name}_exposed_population_{shock_name}_adm{adm_name}_figure.png"
        plt.savefig(path, format="png", dpi=150, bbox_inches="tight")

    if not show:
        plt.close(fig)

    return True


def merge_hospital_shock_shape(hospital_df, shock_shape, shock_name):
    """
    Each hospital point is assigned to scho shape
    """
    data_temp = gpd.sjoin(hospital_df, shock_shape, how="left", predicate="within")
    del data_temp["index_right"]
    data_temp["shock_name"] = data_temp["shock_name"].replace(np.nan, 0)
    data_temp["shock_name"] = data_temp["shock_name"].astype(int)
    # print(len(data_temp))
    return data_temp


def shock_hospital_affected_per_region(
    data,
    shock_name=None,
    adm_region="administration_1",
    category_column_name="cat",
    shape_region=None,
    save_result=False,
    country_name="COUNTRY_NAME",
    path="",
):
    """
    group population by region, keeping shock-affected population

    Parameters:

    """
    adm_name = ""
    columns = []
    if adm_region == "Administration 1":
        columns = ["administration_1_code"]
        adm_name = "1"
    elif adm_region == "Administration 2":
        columns = ["administration_1_code", "administration_2_code"]
        adm_name = "2"
    elif adm_region == "Administration 3":
        columns = [
            "administration_1_code",
            "administration_2_code",
            "administration_3_code",
        ]
        adm_name = "3"
    else:
        adm_name = ""

    list_categories = set(data[category_column_name])
    data_temp = (
        data.groupby(columns + [category_column_name, "shock_name"])
        .agg(
            #   Population=('Population','sum')
            qty=("hospital", "count"),
        )
        .reset_index()
    )

    data_temp = data_temp.pivot(
        index=columns, columns=[category_column_name, "shock_name"]
    )
    data_temp = data_temp.reset_index()
    data_temp.columns = data_temp.columns.to_flat_index()

    data_temp = data_temp.rename(
        columns={
            ("administration_1_code", "", ""): "administration_1_code",
            ("administration_2_code", "", ""): "administration_2_code",
            ("administration_3_code", "", ""): "administration_3_code",
        }
    )

    column_shock_positive = 1
    column_shock_negative = 0
    data_temp = data_temp.replace(np.nan, 0)

    total_per_names = [[], []]
    dict_categories = dict()

    for element in list_categories:
        # positive_name = 'qty_'+element+'_'+column_shock_positive
        # negative_name = 'qty_'+element+'_'+column_shock_negative
        positive_name = f"qty_category_{element}_shock_{column_shock_positive}"
        negative_name = f"qty_category_{element}_shock_{column_shock_negative}"
        data_temp = data_temp.rename(
            columns={
                ("qty", element, column_shock_positive): positive_name,
                ("qty", element, column_shock_negative): negative_name,
            }
        )
        total_per_names[0].append(positive_name)
        total_per_names[1].append(positive_name)
        total_per_names[1].append(negative_name)
        try:
            data_temp[positive_name]
        except:
            data_temp[positive_name] = 0
        try:
            data_temp[negative_name]
        except:
            data_temp[negative_name] = 0

        data_temp[
            # f"Percent of exposed healthcare infrastructure (category {category_dict[element]})"
            f"Percentage_category_{category_dict[element]}"
        ] = round(
            data_temp[positive_name]
            / (data_temp[positive_name] + data_temp[negative_name])
            * 100,
            2,
        )
        # f"Percent of exposed healthcare infrastructure (category {category_dict[element]})"
        # ercentage_category_
        dict_categories[element] = f"Percentage_category_{category_dict[element]}"

    data_temp = data_temp.replace(np.nan, 0)
    data_temp["percentage_total"] = data_temp[total_per_names[0]].sum(
        axis=1
    ) / data_temp[total_per_names[1]].sum(axis=1)
    data_temp["percentage_total"] = data_temp["percentage_total"].apply(
        lambda x: round(x * 100, 2)
    )
    data_temp = data_temp.sort_values(by=["percentage_total"], ascending=False)

    # Area afected
    neo_data = pd.merge(
        data_temp, shape_region, how="left", left_on=columns, right_on=columns
    )

    neo_data = gpd.GeoDataFrame(neo_data, geometry="geometry")
    neo_data.crs = "epsg:4326"

    if save_result:
        data_temp = neo_data.copy()
        del data_temp["geometry"]
        # path = f'{path}/{country_name}_{shock_name}_population_facilities_adm{adm_name}_table.csv'
        path = f"{path}/{country_name}_exposed_health_infrastructure_{shock_name}_adm{adm_name}_table.csv"
        data_temp = rename_columns(data_temp)
        data_temp.to_csv(path, index=False)

    return neo_data, dict_categories


def plot_regions_intensity_hospitals(
    gdf,
    column_name,
    category_hospital,
    transform_column=0,
    shock_name="SHOCK_NAME",
    adm_region="administration_1",
    save_result=False,
    path="",
    country_name="COUNTRY_NAME",
    show=False,
):
    """
    Plot map intensity

    """
    cmap_color = "Reds"
    leg_color = "Red"

    adm_name = ""
    if adm_region == "Administration 1":
        adm_name = "1"
    elif adm_region == "Administration 2":
        adm_name = "2"
    elif adm_region == "Administration 3":
        adm_name = "3"
    else:
        adm_name = ""

    trasformed_column = "temporal_column"
    if transform_column < 0:
        gdf[trasformed_column] = gdf[column_name].apply(
            lambda x: pow(x, 1 / -1 * transform_column)
        )
    elif transform_column > 0:
        gdf[trasformed_column] = gdf[column_name].apply(
            lambda x: pow(x, transform_column)
        )
    else:
        gdf[trasformed_column] = gdf[column_name]

    norm = get_color_scale(gdf, trasformed_column)
    fig, ax = plt.subplots(figsize=(14, 10))
    if country_name == "colombia":
        ax.set_xlim([-82.0, -66.0])  # type: ignore # Límites en la coordenada X
        ax.set_ylim([-4.9, 14.0])  # type: ignore # Límites en la coordenada Y
    if country_name == "peru":
        ax.set_xlim([-82.0, -68.0])  # type: ignore # Límites en la coordenada X
        ax.set_ylim([-19.0, 0.1])  # type: ignore # Límites en la coordenada Y
    ax.axis("off")

    plot = gdf.plot(
        ax=ax,
        column=trasformed_column,
        cmap=cmap_color,
    )
    ctx.add_basemap(
        plot,
        crs="epsg:4326",
        source=ctx.providers.Stamen.TonerBackground,
        attribution="",
    )

    # set latitude & longitude of interest here
    A = [-70.1 * np.pi / 180.0, -2.5 * np.pi / 180.0]
    B = [-70.8 * np.pi / 180.0, -2.5 * np.pi / 180.0]
    dx = (6371000) * haversine_distances([A, B])[0, 1]
    ax.add_artist(ScaleBar(dx=dx, units="m"))

    ticks = np.linspace(
        gdf[trasformed_column].min(), gdf[trasformed_column].max(), num=5, dtype=float
    )

    labels = np.linspace(
        gdf[column_name].min(), gdf[column_name].max(), num=5, dtype=int
    )
    labels = [f"{x}%" for x in labels]

    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.05,
            ax.get_position().y0,
            0.01,
            ax.get_position().height,
        ]
    )
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap_color),
        cax=cax,
        orientation="vertical",
        label="",
        ticks=ticks,
    )

    cbar.ax.set_yticklabels(labels)
    # Legend
    red_dots = Line2D(
        [0],
        [0],
        marker="o",
        color=leg_color,
        markerfacecolor=leg_color,
        label=f"Percent of exposed healthcare infrastructure (category {category_dict[category_hospital]})",
        markersize=5,
        linewidth=0,
    )
    ax.legend(
        title="", handles=[red_dots], bbox_to_anchor=(0.05, 0.02), loc="lower left"
    )

    if save_result:
        # path = f'{path}/{country_name}_{shock_name}_population_facilities_cat{category_hospital}_adm{adm_name}_map.png'
        path = f"{path}/{country_name}_exposed_health_infrastructure_{shock_name}_adm{adm_name}_Category{category_hospital}_figure.png"
        plt.savefig(path, format="png", dpi=150, bbox_inches="tight")

    if not show:
        plt.close(fig)
    return True


def plot_points_maps(
    gdf,
    columns="Population",
    transfor_column=0,
    scale=True,
    save_result=False,
    save_path="",
):
    """
    Parameters:
        - gdf: Dataframe
        - transfor_column: rescales columns values, is used if maps is to disperse
            -1: transformation sqrt
            0: not transformation
            1: transformation power
        - scale:
    """
    gdf = gdf.copy()
    colors = "Reds"
    colors_leg = "Red"
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")

    col = columns
    cmap_color = colors
    leg_color = colors_leg

    if transfor_column == -1:
        gdf["new_col"] = gdf[col].apply(lambda x: x ** (1 / 5))
    else:
        gdf["new_col"] = gdf[col]

    norm = get_color_scale(gdf, col)

    plot = gdf.plot(ax=ax, column="new_col", cmap=cmap_color, markersize=1)

    # Legend
    red_dots = Line2D(
        [0],
        [0],
        marker="o",
        color=leg_color,
        markerfacecolor=leg_color,
        label=col,
        markersize=5,
        linewidth=0,
    )
    ax.legend(
        title="", handles=[red_dots], bbox_to_anchor=(0.9, 0.94), loc="upper left"
    )

    ctx.add_basemap(
        plot,
        crs="epsg:4326",
        source=ctx.providers.Stamen.TonerBackground,
        attribution="",
    )

    if scale:
        # set latitude & longitude of interest here
        A = [-70.1 * np.pi / 180.0, -2.5 * np.pi / 180.0]
        # set latitude & longitude (longitude in A +1) of interest here
        B = [-70.8 * np.pi / 180.0, -2.5 * np.pi / 180.0]
        dx = (6371000) * haversine_distances([A, B])[0, 1]
        ax.add_artist(ScaleBar(dx=dx, units="m"))

    if save_result:
        plt.savefig(save_path, format="png", dpi=150, bbox_inches="tight")

    return plt


def rename_columns(df):
    """
    Renombra todas las columnas de un DataFrame eliminando guiones y guiones bajos
    y poniendo la primera letra de cada palabra en mayúscula.
    Excepto para la columna 'geometry'.
    """
    # Crea un diccionario para almacenar los nuevos nombres de columna
    new_names = {}

    # Itera sobre las columnas del DataFrame
    for column in df.columns:
        # Si la columna es 'geometry', la agrega al diccionario sin cambios
        if column == "geometry":
            new_names[column] = column
        else:
            # Reemplaza los guiones y guiones bajos con espacios
            new_name = column.replace("-", " ").replace("_", " ")

            # Capitaliza la primera letra de cada palabra
            new_name = " ".join(
                word.capitalize() if word.islower() else word
                for word in new_name.split()
            )

            # Agrega el nuevo nombre de columna al diccionario
            new_names[column] = new_name

    # Renombra las columnas del DataFrame con los nuevos nombres
    df = df.rename(columns=new_names)

    # Devuelve el DataFrame con las columnas renombradas
    return df


def clean_list(lst):
    new_lst = []
    for s in lst:
        # Remover _ y - y poner la primera letra en mayúscula
        s = capitalize_string(s)
        new_lst.append(s)
    return new_lst


def capitalize_string(s):
    s = s.replace("_", " ").replace("-", " ").capitalize()
    return s.capitalize()

def capitalize_string2(s):
    s = s.replace("_", " ").replace("-", " ")
    s = ' '.join(word.capitalize() for word in s.split())
    return s


def capitalize_string_reverse(s):
    s = s.replace(" ", "_")
    return s.capitalize()


def qty_departments_complete(COUNTRY_NAME, data_length=None, administrative_level=1):
    if COUNTRY_NAME == 'colombia' and administrative_level==1:
        if data_length == 33:
            return True
        else:
            return False
    if COUNTRY_NAME == 'peru' and administrative_level==1:
        if data_length == 24:
            return True
        else:
            return False
    if COUNTRY_NAME == 'peru' and administrative_level==3:
        if data_length == 2100:
            return True
        else:
            return False





# def rename_columns(df):
#     """
#     Renombra todas las columnas de un DataFrame eliminando guiones y guiones bajos
#     y poniendo la primera letra de cada palabra en mayúscula.
#     """
#     # Crea un diccionario para almacenar los nuevos nombres de columna
#     new_names = {}

#     # Itera sobre las columnas del DataFrame
#     for column in df.columns:
#         # Reemplaza los guiones y guiones bajos con espacios
#         new_name = column.replace('-', ' ').replace('_', ' ')

#         # Capitaliza la primera letra de cada palabra
#         new_name = ' '.join(word.capitalize() if word.islower() else word for word in new_name.split())

#         # Agrega el nuevo nombre de columna al diccionario
#         new_names[column] = new_name

#     # Renombra las columnas del DataFrame con los nuevos nombres
#     df = df.rename(columns=new_names)

#     # Devuelve el DataFrame con las columnas renombradas
#     return df
