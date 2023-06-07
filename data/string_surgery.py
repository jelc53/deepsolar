import json
import os
from glob import glob
from collections import defaultdict
from argparse import ArgumentParser
import pandas as pd

TIME_LIMIT = 600


def process_arguments():
    parser = ArgumentParser()

    parser.add_argument("in_file", type=str, default='file_list1.txt')
    parser.add_argument("--change_str", type=str)
    parser.add_argument("--change_idx", type=int)

    args = parser.parse_args()
    return args.in_file, args.change_str, args.change_idx


def main():
    in_file, change_str, change_idx = process_arguments()
    data_path = '/home/ubuntu/deepsolar/data'

    # load file names txt
    with open(os.path.join(data_path, in_file), 'r') as f:
        lines = f.readlines()

    # path surgery
    new_paths = []
    for path in lines:
        # print(path)
        path_dirs = path.split('/')
        path_dirs[change_idx] = change_str
        new_paths.append('/'.join(path_dirs))

    # write to file
    out_file = in_file.split('.')[0] + '_update.txt'
    with open(os.path.join(data_path, out_file), 'w') as f:
        for new_path in new_paths:
            f.write(new_path)

if __name__ == "__main__":
    main()
