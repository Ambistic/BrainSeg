import math
from typing import Sequence

import geojson
from geojson import loads, dumps
from shapely import geometry
from shapely.ops import transform


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
