import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--inputfile', type=str, help='Input PGN file', required=True)
parser.add_argument('--outputfile', type=str, help='Output PGN file', required=True)
args = parser.parse_args()

prefixes = ('[Round', '[White', '[Black', '[Result')
with open(Path(args.inputfile).resolve(), 'r') as infile:
    with open(Path(args.outputfile).resolve(), 'w') as outfile:
        print("Shrinking PGN...")
        for line in infile:
            if line.startswith(prefixes):
                outfile.write(line)
        print("Process complete.")