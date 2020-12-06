import argparse
import sys


def main(args):
    train_set = args['train_set']
    data_in = args['data_in']
    data_out = args['data_out']

    initial_polynomial = []
    training_points = []
    weights = []

    k = 0
    learning_rate = 0.001
    max_iterations = 0
    n = 0
    precision = 0.00001

    with open(data_in) as f:
        line = f.readline().split('=')
        if line[0] == 'iterations':
            max_iterations = int(line[1])

    for line in open(train_set):
        training_points.append([float(x) for x in line.split()])

    first_line = sys.stdin.readline().split()
    n = int(first_line[0])
    k = int(first_line[1])

    for line in sys.stdin:
        initial_polynomial.append([float(x) for x in line.split()])
        weights.append(float(line.split()[-1]))

    current_iteration = 0
    prev_result = None
    for iteration in range(max_iterations):
        summation = 0
        for row in training_points:
            prediction = weights[0]
            for i in range(n):
                prediction += weights[i+1]*row[i]
            diff = prediction - row[-1]
            summation += diff*diff
            weights[0] -= (learning_rate*diff)
            for i in range(n):
                weights[i+1] -= (learning_rate*diff*row[i])

        result = summation/len(training_points)
        current_iteration += 1
        if prev_result != None and prev_result - result < precision:
            break
        prev_result = result

    f = open(data_out, 'w')
    f.write("iterations=%s" % current_iteration)
    f.close()

    print(str(n) + " " + str(k))
    for i in range(len(weights)):
        print(str(i) + " " + str(weights[i]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train_set", type=str,
                    default="train_set.txt", help="Train set file")
    ap.add_argument("-i", "--data_in", type=str,
                    default="data_in.txt", help="Input data file")
    ap.add_argument("-o", "--data_out", type=str,
                    default="data_out.txt", help="Output data file")
    args = vars(ap.parse_args())
    main(args)
