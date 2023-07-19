import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as GeomPath
from shapely.geometry import shape, Polygon, MultiPolygon
import geopandas as gpd


def draw_geojson_on_image(image, geojson, save_as=None):
    """
    Draw the given GeoJSON on top of the given image using matplotlib.

    :param image: The image as a numpy array.
    :param geojson: The GeoJSON as a dictionary.
    :return: The image with the GeoJSON drawn on top as a numpy array.
    """
    fig, ax = plt.subplots()

    # Plot the image.
    ax.imshow(image)
    geo = gpd.GeoDataFrame.from_features(geojson["features"])
    geo.plot(ax=ax, edgecolor='red', linewidth=1, facecolor='none', marker='+', markersize=1)
    # gpd.GeoDataFrame(geojson).plot(ax=ax, color='red')
    # geojson.plot(ax=ax, color='red')
    # Convert the GeoJSON to a Shapely geometry.
    for feat in geojson["features"]:
        break
        geometry = shape(feat['geometry'])

        # Plot the geometry on top of the image.
        ax.add_patch(geometry)

    # Set the axis limits to the image dimensions.
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([image.shape[0], 0])
    if save_as is not None:
        plt.savefig(save_as)
        plt.close()


def draw_polygons_from_geopandas(arr, geo_df, rescale=1., reverse=False):
    arr = arr.copy()
    nrows, ncols = arr.shape[:2]

    polygons = geo_df['geometry']
    for shape in polygons:
        if isinstance(shape, Polygon):
            seq = [shape]
        elif isinstance(shape, MultiPolygon):
            seq = shape.geoms
        else:
            continue

        for poly in seq:
            x, y = poly.exterior.coords.xy
            x, y = np.array(x) * rescale, np.array(y) * rescale
            if reverse:
                x, y = y, x
            path = GeomPath(list(zip(x, y)))
            row_coords, col_coords = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing='ij')
            points = np.vstack((row_coords.flatten(), col_coords.flatten())).T
            mask = path.contains_points(points).reshape(nrows, ncols)
            arr[mask] = 1

    return arr


def draw_polygon(arr, poly, rescale=1., reverse=False):
    arr = arr.copy()
    nrows, ncols = arr.shape[:2]

    if isinstance(poly, Polygon):
        seq = [poly]
    elif isinstance(poly, MultiPolygon):
        seq = poly.geoms
    else:
        raise TypeError("poly should be of type `Polygon` or `MultiPolygon`")

    for poly in seq:
        x, y = poly.exterior.coords.xy
        x, y = np.array(x) * rescale, np.array(y) * rescale
        if reverse:
            x, y = y, x
        path = GeomPath(list(zip(x, y)))
        row_coords, col_coords = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing='ij')
        points = np.vstack((row_coords.flatten(), col_coords.flatten())).T
        mask = path.contains_points(points).reshape(nrows, ncols)
        arr[mask] = 1

    return arr


def draw_polygons(arr, polygons, rescale=1., reverse=False):
    for poly in polygons:
        arr = draw_polygon(arr, poly, rescale, reverse)
    return arr
