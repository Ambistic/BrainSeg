import warnings

from shapely.geometry import Polygon, MultiPolygon

from brainseg.geo import extract_shape
from brainseg.polygon import make_polygon_ordering, match_polygon, transform_polygons


def as_polygons(poly):
    if isinstance(poly, Polygon):
        return [poly]
    elif isinstance(poly, MultiPolygon):
        return list(poly.geoms)
    else:
        raise ValueError(f"Not recognized shape {poly}")


def process_pial_gm_manual_correction(dict_affine_params, histo_geojson, section_id):
    # pial = extract_shape(histo_geojson, "auto_outline").explode()
    # gm = extract_shape(histo_geojson, "auto_wm").explode()
    # pial = list(pial["geometry"])
    # gm = list(gm["geometry"])
    pial = as_polygons(extract_shape(histo_geojson, "auto_outline")["geometry"])
    gm = as_polygons(extract_shape(histo_geojson, "auto_wm")["geometry"])
    order = make_polygon_ordering(pial)
    ordered_pial = [pial[index] for index in order]
    params = build_params_from_dict(dict_affine_params, ordered_pial, section_id)
    transformed_pial = transform_polygons(ordered_pial, params)

    assert_polygons_not_too_small(pial, threshold=100000)

    params_gm = match_params(gm, ordered_pial, params)
    transformed_gm = transform_polygons(gm, params_gm)
    return ordered_pial, gm, params, params_gm, transformed_pial, transformed_gm


def build_params_from_dict(dict_affine_params, pial, section_id):
    params = []
    for i, poly in enumerate(pial):
        value = dict_affine_params.get((section_id, i))
        if value is None:
            params.append(None)
            continue

        p = (*get_center(poly), *value)
        params.append(p)
    return params


def get_center(poly):
    x1, y1, x2, y2 = poly.bounds
    return (x1 + x2) / 2, (y1 + y2) / 2


def match_params(source_polygons, target_polygons, params, contained=None, warn=False):
    id_match = [match_polygon(poly_gm, target_polygons, contained=contained, warn=warn)
                for poly_gm in source_polygons]
    params_gm = []
    for idx in id_match:
        if idx is None:
            params_gm.append(None)
            continue
        if params[idx] is None:
            params_gm.append(None)
            continue
        # p = (*params[idx], *get_center(target_polygons[idx]))
        p = params[idx]
        params_gm.append(p)

    return params_gm


def assert_polygons_not_too_small(polygons_outline, threshold):
    for poly in polygons_outline:
        if poly.area < threshold:
            warnings.warn("Small polygon detected, it might not be visible !")
