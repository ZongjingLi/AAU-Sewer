import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name",default = "AAU")
parser.add_argument("--data_dir",default = "/Users/melkor/Desktop/datasets/AAU")

parser.add_argument("--epoch",default = 100)

opt = parser.parse_args(args = [])


