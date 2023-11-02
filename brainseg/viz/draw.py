import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as GeomPath
from shapely.geometry import shape, Polygon, MultiPolygon
import geopandas as gpd
from rasterio import features

from brainseg.polygon import rescale_polygon, get_holes


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


def draw_polygons_from_geopandas_2(arr, geo_df, rescale=1., reverse=False):
    from rasterio import features, transform
    # import pdb; pdb.set_trace()

    holes = [x for geometry in geo_df['geometry'] for x in get_holes(geometry)]

    shapes_out = [(rescale_polygon(geometry, rescale), 1) for geometry in geo_df['geometry']]
    shapes_in = [(rescale_polygon(geometry, rescale), 0) for geometry in holes]
    features.rasterize(shapes=shapes_out + shapes_in, out=arr,
                       # transform=transform.from_bounds(0, 0, arr.shape[0], arr.shape[1],
                       #                                 arr.shape[0], arr.shape[1]),
                       all_touched=True)

    return arr


def draw_polygons_from_geopandas_3(arr, geo_df, rescale=1., reverse=False):
    shapes = []
    for geometry in geo_df['geometry']:
        shapes.append((rescale_polygon(geometry, rescale), 1))
        for holes in get_holes(geometry):
            shapes.append((rescale_polygon(holes, rescale), 0))

    features.rasterize(shapes=shapes, out=arr,
                       all_touched=True)
    return arr


def draw_polygons_from_geopandas_corrected(arr, geo_df, rescale=1., reverse=False):
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
            # Process the exterior ring
            exterior = poly.exterior
            x_ext, y_ext = exterior.coords.xy
            x_ext, y_ext = np.array(x_ext) * rescale, np.array(y_ext) * rescale
            if reverse:
                x_ext, y_ext = y_ext, x_ext
            path_ext = GeomPath(list(zip(x_ext, y_ext)))

            # Process the interior rings (holes)
            interiors = poly.interiors
            paths_int = []
            for interior in interiors:
                x_int, y_int = interior.coords.xy
                x_int, y_int = np.array(x_int) * rescale, np.array(y_int) * rescale
                if reverse:
                    x_int, y_int = y_int, x_int
                path_int = GeomPath(list(zip(x_int, y_int)))
                paths_int.append(path_int)

            # Create the combined path
            path = GeomPath.make_compound_path(path_ext, *paths_int)
            import pdb; pdb.set_trace()

            # Generate the mask
            row_coords, col_coords = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing='ij')
            points = np.vstack((row_coords.flatten(), col_coords.flatten())).T
            mask = path.contains_points(points).reshape(nrows, ncols)
            arr[mask] = 1

    return arr


def draw_polygon(arr, poly, rescale=1., reverse=False, value=1):
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
        arr[mask] = value

    return arr


def draw_polygons(arr, polygons, rescale=1., reverse=False, color_iteration=False):
    for i, poly in enumerate(polygons):
        value = i + 1 if color_iteration else 1
        arr = draw_polygon(arr, poly.simplify(tolerance=1 / rescale), rescale, reverse, value)
    return arr