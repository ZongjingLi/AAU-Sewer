import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name",default = "AAU")
parser.add_argument("--data_dir",default = "/Users/melkor/Desktop/datasets/AAU")

config = parser.parse_args(args = [])