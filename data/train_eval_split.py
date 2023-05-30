#!/usr/bin/python3
import os
import random
from argparse import ArgumentParser

def process_arguments():
    parser = ArgumentParser()
    parser.add_argument("in_file", type=str, default='file.txt')
    parser.add_argument("--num_eval", type=int, default=5000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = process_arguments()
    filename = args.in_file
    data_path = '/home/ubuntu/deepsolar/data'

    with open(os.path.join(data_path, filename), 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)
    eval_lines = lines[:args.num_eval]
    train_lines = lines[args.num_eval:]

    with open(os.path.join(data_path, 'train.txt'), 'w') as f:
        for line in train_lines:
            f.write(line)

    with open(os.path.join(data_path, 'eval.txt'), 'w') as f:
        for line in eval_lines:
            f.write(line)
    