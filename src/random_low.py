import sys
import random
import linecache

def random_lines(filename):
    idxs = random.sample(range(25010), 10)
    return [linecache.getline(filename, i) for i in idxs]

file_path="./../data_sets/poker-hand-training-true.data"

for line in random_lines(file_path):
    print(line)
