import argparse
from pathlib import Path
from shapely.affinity import scale
import geopandas as gpd


def rescale_polygon(polygon, scale_factor):
    rescaled_polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    return rescaled_polygon


def main(args):
    gdf = gpd.read_file(args.geojson)
    gdf["geometry"] = gdf["geometry"].apply(lambda x: rescale_polygon(x, args.scale))
    gdf.to_file(args.geojson, driver="GeoJSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--geojson", type=Path)
    parser.add_argument("-s", "--scale", type=float)
    args_ = parser.parse_args()
    main(args_)
