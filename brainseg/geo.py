import math
import re
from typing import Sequence

import geojson
import numpy as np
from geojson import loads, dumps, Point as GeoPoint, Feature, Polygon as GeoPolygon, FeatureCollection, \
    LineString as GeoLineString, utils, dump
from shapely import geometry
from shapely.geometry import shape, Point, LineString
from shapely.ops import transform
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import copy

from shapely.validation import make_valid

from brainseg.svg.utils import is_point, is_polygon, points_to_numpy, css_to_dict, is_comment
from prepro_svg import is_line
from brainseg.math import distance


def quickfix_multipolygon(geo):
    geo = geo.copy()
    for feat in geo["features"]:
        if feat is None:
            continue

        if feat["geometry"]["type"] == "MultiPolygon" \
                and len(feat["geometry"]["coordinates"]) == 1:
            feat["geometry"]["coordinates"].append([])
    return geo


def quickfix_multipolygon_shapely(geo):
    """Fix the qupath format such that shapely does not raise any error"""
    geo = geo.copy()
    features = []
    for feat in geo["features"]:
        if feat is None:
            continue

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

    print("debug", len(data["features"]))
    # remove not ok Polygons
    filt_features = []
    for feature in data['features']:
        discard = False
        if feature['geometry']['type'] == 'Polygon':
            for coord in feature['geometry']['coordinates']:
                if len(coord) < 3:
                    discard = True
                    print("Discarding a Polygon")
        if not discard:
            filt_features.append(feature)

    data['features'] = filt_features
    print("debug", len(data["features"]))
    # remove not ok shapely
    filt_features = []
    for feature in data['features']:
        discard = False
        try:
            shape(feature["geometry"])
        except Exception as e:
            print(f"A mistake occurs due to shapely error {e}")
            # print(feature)
            print("=" * 40)
        else:
            pass
        if not discard:
            filt_features.append(feature)

    data['features'] = filt_features
    print("debug", len(data["features"]))
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
    if type(name) is list:
        res = histo_geojson[histo_geojson["name"].isin(name)]
    else:
        res =  histo_geojson[histo_geojson["name"] == name]

    if len(res) > 0:
        return res.iloc[0]
    else:
        raise ValueError(f"No {name} found in the geojson file")


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


def geojson_to_svg(svg, mapping_stroke_classification):
    """
    For plotfast
    :param svg:
    :param mapping_stroke_classification:
    :return:
    """
    features = []
    last_comment = None
    for x in svg.iterchildren():
        if is_point(x):
            point_name = get_point_name_from_comment(str(last_comment))
            if "x" in x.attrib:
                kx, ky = "x", "y"
            else:
                kx, ky = "cx", "cy"
            point = GeoPoint((float(x.attrib[kx]), float(x.attrib[ky])))
            properties = {"classification": {"name": point_name}}
            feat = Feature(geometry=point, properties=properties)
            features.append(feat)

        if is_polygon(x):
            # process color
            # add a polygon
            coords = points_to_numpy(x.attrib["points"])
            polygon = GeoPolygon([coords.tolist()])
            style = css_to_dict(x.attrib["style"])
            category = mapping_stroke_classification.get(style["stroke"], "none")
            if category == "none":
                print(style["stroke"])
            properties = {"classification": {"name": category}}
            feat = Feature(geometry=polygon, properties=properties)
            features.append(feat)

        if is_comment(x):
            last_comment = x
        else:
            last_comment = None

    return FeatureCollection(features)


def svg_to_geojson(svg, mapping_stroke_classification):
    features = []
    last_comment = None
    for x in svg.iterchildren():
        if is_point(x):
            point_name = get_point_name_from_comment(str(last_comment))
            if "x" in x.attrib:
                kx, ky = "x", "y"
            else:
                kx, ky = "cx", "cy"
            point = GeoPoint((float(x.attrib[kx]), float(x.attrib[ky])))
            properties = {"classification": {"name": point_name}}
            feat = Feature(geometry=point, properties=properties)
            features.append(feat)

        if is_polygon(x):
            # process color
            # add a polygon
            coords = points_to_numpy(x.attrib["points"])
            polygon = GeoPolygon([coords.tolist()])
            style = css_to_dict(x.attrib["style"])
            category = mapping_stroke_classification.get(style["stroke"], "none")
            if category == "none":
                print(style["stroke"])
            properties = {"classification": {"name": category}}
            feat = Feature(geometry=polygon, properties=properties)
            features.append(feat)

        if is_comment(x):
            last_comment = x
        else:
            last_comment = None

    return FeatureCollection(features)


def svg_to_geojson_with_lines(svg, mapping_stroke_classification):
    features = []
    last_comment = None
    for x in svg.iterchildren():
        if is_point(x):
            point_name = get_point_name_from_comment(str(last_comment))
            if "x" in x.attrib:
                kx, ky = "x", "y"
            else:
                kx, ky = "cx", "cy"
            point = GeoPoint((float(x.attrib[kx]), float(x.attrib[ky])))
            properties = {"classification": {"name": point_name}}
            feat = Feature(geometry=point, properties=properties)
            features.append(feat)

        if is_polygon(x):
            # process color
            # add a polygon
            coords = points_to_numpy(x.attrib["points"])
            polygon = GeoPolygon([coords.tolist()])
            style = css_to_dict(x.attrib["style"])
            category = mapping_stroke_classification.get(style["stroke"], "none")
            if category == "none":
                print(style["stroke"])
            properties = {"classification": {"name": category}}
            feat = Feature(geometry=polygon, properties=properties)
            features.append(feat)

        if is_line(x):
            # process color
            # add a polygon
            coords = points_to_numpy(x.attrib["points"])
            polygon = GeoLineString([coords.tolist()])
            style = css_to_dict(x.attrib["style"])
            category = mapping_stroke_classification.get(style["stroke"], "none")
            if category == "none":
                print(style["stroke"], style)
            properties = {"classification": {"name": category}}
            feat = Feature(geometry=polygon, properties=properties)
            features.append(feat)

        if is_comment(x):
            last_comment = x
        else:
            last_comment = None

    return FeatureCollection(features)


def simplify_line(line, tol=20):
    if len(line) < 3:
        return np.array(line)

    line = geometry.LineString(np.array(line))
    line = line.simplify(tol)  # arbitrary

    return np.array(line.coords)


def simplify_all(geo: FeatureCollection):
    geo = geo.copy()

    for feat in geo["features"]:
        coords = feat["geometry"]["coordinates"]
        if feat["geometry"]["type"] == "Polygon":
            new_coords = [simplify_line(coords[0]).tolist()]
        elif feat["geometry"]["type"] == "MultiPolygon":
            new_coords = list(map(lambda x: [simplify_line(x[0]).tolist()], coords))
        else:
            new_coords = coords

        feat["geometry"]["coordinates"] = new_coords

    return geo


def get_point_name_from_comment(comment):
    match = re.search(r'<!-- Point (.*?) -->', comment)
    if match:
        s = match.group(1)
        return s
    return None


def create_qupath_point(x, y, point_name="none"):
    point = GeoPoint((float(x), float(y)))
    properties = {"classification": {"name": point_name}}
    feat = Feature(geometry=point, properties=properties)

    return feat


def create_qupath_single_point(x, y, point_name="none"):
    point = GeoPoint((float(x), float(y)))
    properties = {"objectType": "annotation", "name": point_name}
    feat = Feature(geometry=point, properties=properties)

    return feat


# QuPath v0.5 proofed
def create_qupath_line(coords, line_name="none") -> Feature:
    coords = np.array(coords)
    polygon = GeoLineString(coords.tolist())
    properties = {"objectType": "annotation", "name": line_name}
    # properties = {"classification": {"name": line_name}}
    feat = Feature(geometry=polygon, properties=properties)

    return feat


def create_qupath_polygon(coords, line_name="none"):
    """Provide a 2-D array, a list of lists or a list of arrays"""
    coords = np.array(coords)
    polygon = GeoPolygon([coords.tolist()])
    properties = {"objectType": "annotation", "name": line_name}
    # properties = {"classification": {"name": line_name}}
    feat = Feature(geometry=polygon, properties=properties)

    return feat


def find_feature_by_name(feats, name):
    for feat in feats:
        if feat["properties"]["name"] == name:
            return feat
    return None


def get_closest_point_id(coords, polygon_coords):
    coords = np.array(coords)
    polygon_coords = np.array(polygon_coords)
    distances = distance(coords, polygon_coords)

    wheremin = np.where(distances == np.min(distances))[0][0]
    return wheremin


def get_closest_point_id_from_list(coords, polygon_coords_list):
    listmin, wheremin, value_min = None, None, 1e50
    for i, poly in enumerate(polygon_coords_list):
        current_wheremin = get_closest_point_id(coords, poly)
        current_value = distance(coords, poly[current_wheremin])
        if current_value < value_min:
            value_min = current_value
            wheremin = current_wheremin
            listmin = i

    return listmin, wheremin


def get_furthest_point_id(coords, polygon_coords):
    """Note that `polygon_coords` can be just a list of points"""
    coords = np.array(coords)
    polygon_coords = np.array(polygon_coords)
    distances = distance(coords, polygon_coords)

    wheremax = np.where(distances == np.max(distances))[0][0]
    return wheremax


def find_segment_intersected(polygon, point: Point, threshold=1e-3):
    # Find the segments that form the intersecting segment
    segments_line = [(polygon.coords[i], polygon.coords[i + 1]) for i in range(len(polygon.coords) - 1)]

    # Check which segment contains the intersection point
    intersecting_segment_line = None

    for segment in segments_line:
        segment_line = LineString(segment)

        if segment_line.distance(point) < threshold:
            intersecting_segment_line = segment
            break

    return intersecting_segment_line


def find_close_head(tail, heads, threshold=200):
    for i, h in enumerate(heads):
        if distance(tail, h) < threshold:
            return i
    return None


def valid_tail_point(head, tail, point):
    coords = point.coords[0]
    if distance(head, coords) > distance(tail, coords) * 1.2:
        return True
    return False


def valid_tail_points(head, tail, points):
    valid_points = []
    for p in points:
        if valid_tail_point(head, tail, p):
            valid_points.append(p)

    return valid_points


def interpolate_line(line, max_length=100):
    # This function interpolates points on a LineString
    points = list(line.coords)
    new_points = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        segment_length = math.dist(start, end)
        if segment_length > max_length:
            num_segments = math.ceil(segment_length / max_length)
            for i in range(1, num_segments):
                new_points.append((
                    start[0] + (end[0] - start[0]) * i / num_segments,
                    start[1] + (end[1] - start[1]) * i / num_segments
                ))
        new_points.append(end)
    return new_points


def interpolate_polygon(polygon, max_length=100):
    exterior_coords = interpolate_line(LineString(polygon.exterior.coords), max_length)
    new_interiors = []
    for interior in polygon.interiors:
        interior_coords = interpolate_line(LineString(interior.coords), max_length)
        new_interiors.append(interior_coords)

    new_polygon = Polygon(exterior_coords, new_interiors)
    return new_polygon


def transform_geojson(json_obj, transform_matrix):
    json_obj = copy.deepcopy(json_obj)

    def wrapped_transform(x):
        # why the "-" ? I don't know, but it works like this
        x = np.array(x + [1])  # because it must be 1 at the moment of the transform
        y_ = x @ transform_matrix.T
        return list(y_)

    ls = []

    for feat in json_obj["features"]:
        obj = utils.map_tuples(wrapped_transform, feat)
        ls.append(obj)

    json_obj["features"] = ls

    return json_obj


def save_geojson(args, geo, output):
    with open(output, "w") as f:
        dump(geo, f)
