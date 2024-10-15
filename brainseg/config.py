import configparser
import argparse
from pathlib import Path


def fill_with_config(parser):
    args_ = parser.parse_args()
    reparsing = {action.dest: action.type for action in parser._actions}

    if args_.config is not None:
        config = configparser.ConfigParser()
        if not Path(args_.config).exists():
            raise ValueError(
                f"""
The configuration file does not exists:
{args_.config}\n
Make sure to use double quotes (") if the path contains a space.
"""
            )
        config.read(args_.config)
        cfg = dict(config.items('DEFAULT'))
    else:
        cfg = dict()

    arg_dict = dict()
    for k, v in vars(args_).items():
        if v is not None:
            arg_dict[k] = v
        elif k == "config":
            arg_dict[k] = v
        else:
            try:
                arg_dict[k] = reparsing[k](cfg[k])
            except KeyError:
                raise KeyError(f"Missing `{k}` in both arguments and config file")

    return argparse.Namespace(**arg_dict)
