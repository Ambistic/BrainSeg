import configparser
import argparse


def fill_with_config(parser):
    args_ = parser.parse_args()
    reparsing = {action.dest: action.type for action in parser._actions}

    if args_.config is not None:
        config = configparser.ConfigParser()
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
