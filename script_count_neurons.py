import argparse
from pathlib import Path

from brainseg.config import fill_with_config
from brainseg.path import build_path_histo
from brainseg.utils import read_histo


def main(args):
    total_dict = dict()
    for i in range(args.start, args.end, args.step):
        file = build_path_histo(args.mri_atlas_dir, i, args.merged_annotations_mask)
        histo = read_histo(file)

        for feat in histo["features"]:
            if not feat['properties'].get('classification', dict()).get("name", "").startswith('area'):
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default=Path("/media/tower/LaCie/REGISTRATION_PROJECT/"
                                                                  "TMP_PIPELINE_M148/config.ini"))
    parser.add_argument("--mri_atlas_dir", type=str, default=None)
    parser.add_argument("--merged_annotations_mask", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    args_ = fill_with_config(parser)

    main(args_)
