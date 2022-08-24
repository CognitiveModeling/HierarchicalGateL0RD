import argparse
import sys
import smart_settings
from train_forward_inverse_model import main_FPP_events


parser = argparse.ArgumentParser()
parser.add_argument("-config", help="Path to configuration file")
parser.add_argument("-seed", type=int, help="Seed for RNG")

if __name__ == '__main__':

    args = parser.parse_args(sys.argv[1:])
    params = smart_settings.load(args.config)
    rs = -1
    if args.seed is not None:
        rs = args.seed
    main_FPP_events(params=params)
