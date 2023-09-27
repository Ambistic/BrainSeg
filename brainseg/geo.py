import math
from typing import Sequence

import geojson
from geojson import loads, dumps
from shapely import geometry
from shapely.geometry import shape
from shapely.ops import transform
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import copy

from shapely.validation import make_valid


def quickfix_multipolygon(geo):
    geo = geo.copy()
    for feat in geo["features"]:
        if feat["geometry"]["type"] == "MultiPolygon" \
                and len(feat["geometry"]["coordinates"]) == 1:
            feat["geometry"]["coordinates"].append([])
    return geo


def quickfix_multipolygon_shapely(geo):
    """Fix the qupath format such that shapely does not raise any error"""
    geo = geo.copy()
    features = []
    for feat in geo["features"]:
        if feat["geometry"]["type"] == "MultiPolygon":
            coords = []
            for coord in feat["geometry"]["coordinates"]:
                if len(coord) > 0 and len(coord[0]) >= 3:
                    coords.append(coord)
            feature = geojson.Feature(geometry=geojson.MultiPolygon(coords), properties=feat["properties"])
        elif feat["geometry"]["type"] == "Polygon":
            try:
                geometry.shape(feat["geometry"])
            except:
                continue
            else:
                feature = feat
        elif feat["geometry"]["type"] == "PolygonX":
            coords = []
            if len(feat["geometry"]["coordinates"]) == 0 or len(feat["geometry"]["coordinates"][0]) <= 2:
                continue
            for coord in feat["geometry"]["coordinates"]:
                if len(coord) > 2:
                    coords.append(coord)
            feature = geojson.Feature(geometry=geojson.Polygon(coords), properties=feat["properties"])
        else:
            feature = feat
        features.append(feature)

    feature_collection = geojson.FeatureCollection(features)
    return feature_collection


def fix_geojson_file(filename):
    with open(filename, 'r') as f:
        data = geojson.load(f)

    # Iterate through features and add empty coordinates to Multipolygons
    for feature in data['features']:
        if feature['geometry']['type'] == 'MultiPolygon':
            for coord in feature['geometry']['coordinates']:
                if len(coord) == 1 and len(coord[0]) == 0:
                    coord.pop()

    # Save the modified data back to the same location
    with open(filename, 'w') as f:
        geojson.dump(data, f, indent=2)


def get_gm_polygon(geo):
    for feature in geo['features']:
        if feature['properties'].get('classification', dict()).get("name", "").startswith('auto_wm'):
            return geometry.shape(feature['geometry'])
        raise AttributeError("could not find 'auto_wm'")


def get_outline_polygon(geo):
    for feature in geo['features']:
        if feature['properties'].get('classification', dict()).get("name", "").startswith('auto_outline'):
            return geometry.shape(feature['geometry'])
        raise AttributeError("could not find 'auto_outline'")


def get_polygon_by_classification(geo, classification):
    for feature in geo['features']:
        if feature['properties'].get('classification', dict()).get("name", "") == classification:
            return geometry.shape(feature['geometry'])
    raise AttributeError(f"could not find '{classification}'")


def transform_backward_histo(args, geo):
    print(args.histo_downscale)
    geo = geo.copy()
    feats = geo["features"].copy()

    def rescale(x, y):
        if isinstance(x, Sequence):
            raise TypeError("")  # To fail the attempt
        return x * args.histo_downscale, y * args.histo_downscale
        # return x * 10.0, y * 10.0

    new_feats = list(map(
        lambda f: geojson_mapping(f, rescale),
        feats
    ))

    geo["features"] = new_feats
    return geo


def transform_rescale(scale, geo):
    geo = geo.copy()
    feats = geo["features"].copy()

    def rescale(x, y):
        if isinstance(x, Sequence):
            raise TypeError("")  # To fail the attempt
        return x * scale, y * scale
        # return x * 10.0, y * 10.0

    new_feats = list(map(
        lambda f: geojson_mapping(f, rescale),
        feats
    ))

    geo["features"] = new_feats
    return geo


def geojson_mapping(feature, mapping):
    feature = feature.copy()

    geom = geometry.mapping(
        transform(mapping, geometry.shape(feature["geometry"]))
    )

    feature["geometry"] = loads(dumps(geom))

    return feature


def record_point_coordinates(geo):
    geo = geo.copy()
    point_list = []

    def record(x, y):
        if isinstance(x, Sequence):
            raise TypeError("")  # To fail the attempt
        point_list.append((x, y))
        return x, y

    feats = geo["features"].copy()
    list(map(
        lambda f: geojson_mapping(f, record),
        feats
    ))

    return point_list


def transform_from_dict(geo, dict_mapping):
    geo = geo.copy()
    feats = geo["features"].copy()

    def func_mapping(x, y):
        if isinstance(x, Sequence):
            raise TypeError("")  # To fail the attempt
        dx, dy = dict_mapping[(x, y)]
        point = (x + dx, y + dy)
        return point

    new_feats = list(map(
        lambda f: geojson_mapping(f, func_mapping),
        feats
    ))

    geo["features"] = new_feats
    return geo


def create_affine_mapping(center_x=0, center_y=0, rot=0, flip=False, shift_x=0, shift_y=0, scale=1):
    def mapping_function(x, y):
        if isinstance(x, Sequence):
            raise TypeError("")  # To fail the attempt
        x = x - center_x
        y = y - center_y

        if flip:
            x *= -1

        rotation_angle_rad = math.radians(rot)
        new_x = x * math.cos(rotation_angle_rad) - y * math.sin(rotation_angle_rad)
        new_y = x * math.sin(rotation_angle_rad) + y * math.cos(rotation_angle_rad)

        x = new_x + center_x + shift_x
        y = new_y + center_y + shift_y

        x *= scale
        y *= scale

        return x, y

    return mapping_function


def create_inverse_affine_mapping(center_x=0, center_y=0, rot=0, flip=False, shift_x=0, shift_y=0, scale=1,):
    def mapping_function(x, y):
        if isinstance(x, Sequence):
            raise TypeError("")  # To fail the attempt

        # Compute the inverse transformation
        x = x / scale
        y = y / scale

        x = x - center_x - shift_x
        y = y - center_y - shift_y

        rotation_angle_rad = math.radians(-rot)
        new_x = x * math.cos(rotation_angle_rad) - y * math.sin(rotation_angle_rad)
        new_y = x * math.sin(rotation_angle_rad) + y * math.cos(rotation_angle_rad)

        if flip:
            new_x *= -1

        x = new_x + center_x
        y = new_y + center_y

        return x, y

    return mapping_function


def transform_forward_histo(geo, args):
    geo = geo.copy()
    feats = geo["features"].copy()

    def rescale(x, y):
        return x / args.histo_downscale, y / args.histo_downscale

    new_feats = list(map(
        lambda f: geojson_mapping(f, rescale),
        feats
    ))

    geo["features"] = new_feats
    return geo


def extract_shape(histo_geojson, name):
    return histo_geojson[histo_geojson["name"] == name].iloc[0]


def polygons_to_geopandas(polygons_list):
    shapes_series = gpd.GeoSeries(polygons_list)
    gdf = gpd.GeoDataFrame(geometry=shapes_series)
    return gdf


def get_feature_center(feature):
    poly = shape(feature["geometry"])
    x1, y1, x2, y2 = poly.bounds

    return (x1 + x2) / 2, (y1 + y2) / 2


def split_multipolygons_to_polygons(feature_collection):
    new_features = []

    for feature in feature_collection['features']:
        geometry = feature['geometry']
        properties = feature['properties']

        # If the geometry is a MultiPolygon, split it into individual polygons
        if geometry['type'] == 'MultiPolygon':
            for polygon_coords in geometry['coordinates']:
                new_feature = copy.deepcopy(feature)
                new_feature['geometry'] = {
                    'type': 'Polygon',
                    'coordinates': polygon_coords
                }
                new_features.append(new_feature)
        else:
            # If it's not a MultiPolygon, keep the original feature
            new_features.append(feature)

    # Create a new GeoJSON Feature Collection with the split polygons
    split_feature_collection = {
        'type': 'FeatureCollection',
        'features': new_features
    }

    return split_feature_collection


def transform_from_manual_correction(geo, full_params, direction="forward"):
    geo = geo.copy()
    feats = geo["features"].copy()
    new_feats = []

    for feat, full_param in zip(feats, full_params):
        if full_param is None:
            new_feats.append(feat)
            continue

        center_x, center_y, flip, rotation_angle, shift_x, shift_y = full_param
        # error is here, it's not the center of the feat but of the pial poly
        # center_x, center_y = get_feature_center(feat)

        if direction == "forward":
            map_func = create_affine_mapping(flip=flip, rot=rotation_angle,
                                             shift_x=shift_x, shift_y=shift_y,
                                             center_x=center_x, center_y=center_y)
        elif direction == "backward":
            map_func = create_inverse_affine_mapping(flip=flip, rot=rotation_angle,
                                                     shift_x=shift_x, shift_y=shift_y,
                                                     center_x=center_x, center_y=center_y)
        else:
            raise ValueError(f"Not recognized direction : {direction}")

        new_feat = geojson_mapping(feat, map_func)
        new_feats.append(new_feat)

    geo["features"] = new_feats
    return geo


def process_feature(feature):
    geom = shape(feature['geometry'])

    if not geom.is_valid:
        print("validity issue")
        geom = make_valid(geom)

    if feature["geometry"]["type"] == "MultiPolygon":
        geom = list(geom.geoms)

    return geom


def flatten_geojson(geo):
    flattened_shapes = []

    for feature in geo['features']:
        shape = process_feature(feature)

        if isinstance(shape, list):
            flattened_shapes.extend(shape)
        else:
            flattened_shapes.append(shape)

    return flattened_shapes


def convert_multipolygons(feature):
    geometry = feature["geometry"]
    if geometry["type"] == "MultiPolygon":
        polygons = list(shape(geometry).geoms)
        return [{"type": "Feature", "properties": feature["properties"], "geometry": p} for p in polygons]
    else:
        return [feature]


def explode_multipolygon_geojson(geo):
    # Convert MultiPolygons to separate Polygons
    converted_features = [convert_multipolygons(feature) for feature in geo["features"]]
    flattened_features = [feature for sublist in converted_features for feature in sublist]

    # Create a new GeoJSON object with the converted features
    converted_geojson = {"type": "FeatureCollection", "features": flattened_features}
    return converted_geojson
