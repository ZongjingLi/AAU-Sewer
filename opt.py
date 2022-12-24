import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name",default = "AAU")
parser.add_argument("--data_dir",default = "/Users/melkor/Desktop/datasets/AAU")

parser.add_argument("--epoch",      default = 100)
parser.add_argument("--lr",         default = 2e-4)
parser.add_argument("--batch_size", default = 5)

opt = parser.parse_args(args = [])


