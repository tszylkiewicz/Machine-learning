import argparse
import sys


def main(args):
    description_file = args['description']

    inputs = []
    components = []
    k = 0
    n = 0

    with open(description_file) as f:
        first_line = f.readline().split()
        n = int(first_line[0])
        k = int(first_line[1])
        for line in f:
            components.append([float(x) for x in line.split()])


    for line in sys.stdin:
        inputs.append([float(x) for x in line.split()])  

    for variables in inputs:
        result = 0
        for component in components:
            value = 1
            for i in range(k):
                if component[i] != 0:
                    value *= variables[int(component[i]) - 1]
            result += value * component[-1]
        print(result)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--description", type=str,
                    default="description.txt", help="Description data file")
    args = vars(ap.parse_args())
    main(args)
