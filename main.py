import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--bob", action="store_true")
parser.add_argument("--joe", action="store_true")
parser.add_argument("dir1")
parser.add_argument("dir2")
args = parser.parse_args()