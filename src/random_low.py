import sys
import random
import linecache

def random_lines(filename,num):
    idxs = random.sample(range(1000000), num)
    return [linecache.getline(filename, i) for i in idxs]

file_path="./../data_sets/poker-hand-training-true.data"
file_test="./../data_sets/poker-hand-testing.data"
path_w="./../data_sets/poker-test.data"

num=25000
with open(path_w,mode='w') as f:
    for line in random_lines(file_test,num):
        f.write(line)

f.close()


