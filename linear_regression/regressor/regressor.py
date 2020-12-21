import os
import sys
import math
import random
import argparse


K_FOLDS = 8
LEARNING_RATE = 0.1
MAX_STOP_CONDITION = 100
MOMENTUM = 0.9
N_EPOCH = 5000
HIDDEN_NEURONS = [2, 3, 4, 6, 8, 10, 20]


def normalize_value(value, min_value, max_value):
    result = (value - min_value) / (max_value - min_value)
    result = (2 * result) - 1
    return result


def denormalize_value(value, min_value, max_value):
    result = (value + 1) / 2
    result = result * (max_value - min_value) + min_value
    return result


def calculate_column_ranges(data):
    min_values = []
    max_values = []

    for i in range(len(data[0])):
        min_values.append([min(x) for x in zip(*data)][i])
        max_values.append([max(x) for x in zip(*data)][i])

    return min_values, max_values


def normalize_data(data, min_values, max_values):
    for i in range(len(data[0])):
        for j in range(len(data)):
            data[j][i] = normalize_value(data[j][i], min_values[i], max_values[i])

    return data


def generate_folds(data, k_folds):
    random.shuffle(data)
    return [data[i::k_folds] for i in range(k_folds)]
    # folds = []
    # avg = len(data) / float(k_folds)
    # last = 0.0
    
    # while last < len(data):
    #     folds.append(data[int(last):int(last + avg)])
    #     last += avg

    # return folds


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)


def activate(inputs, weights):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def train(training_set, weights_ih, weights_ho, n_inputs, n_hidden):
    values = []

    for input_values in training_set:
        for hidden_index in range(n_hidden):
            values.append(sigmoid(activate(input_values, weights_ih["weights"][hidden_index])))

        output_value = activate(values, weights_ho["weights"])
        hidden_gradient = output_value - input_values[-1]

        for hidden_index in range(n_hidden):
            input_gradient = hidden_gradient * weights_ho["weights"][hidden_index] * derivative_sigmoid(values[hidden_index])

            for input_index in range(n_inputs + 1):
                if input_index == n_inputs:
                    delta = -LEARNING_RATE * input_gradient
                else:
                    delta = -LEARNING_RATE * input_gradient * input_values[input_index]
                weights_ih["weights"][hidden_index][input_index] += delta + (MOMENTUM * weights_ih["deltas"][hidden_index][input_index])
                weights_ih["deltas"][hidden_index][input_index] = delta
                
            delta = -LEARNING_RATE * hidden_gradient * values[hidden_index]
            weights_ho["weights"][hidden_index] += delta + (MOMENTUM * weights_ho["deltas"][hidden_index])
            weights_ho["deltas"][hidden_index] = delta
            
        delta = -LEARNING_RATE * hidden_gradient
        weights_ho["weights"][-1] += delta + (MOMENTUM * weights_ho["deltas"][-1])
        weights_ho["deltas"][-1] = delta


def validate(validation_set, weights_ih, weights_ho, n_inputs, n_hidden):    
    mean_square_error = 0
    for input_values in validation_set:
        output_value = forward_propagate(input_values, weights_ih, weights_ho, n_inputs, n_hidden)
        error = output_value - input_values[-1]
        mean_square_error += error ** 2
    return mean_square_error / len(validation_set) / 2


def forward_propagate(value, weights_ih, weights_ho, n_inputs, n_hidden):
    values = []
    for hidden_index in range(n_hidden):
        values.append(sigmoid(activate(value, weights_ih[hidden_index])))
    output_value = activate(values, weights_ho)
    return output_value


def train_network(training_set, validation_set, n_inputs, n_hidden):    
    validation_error = 0    
    stop_condition = 0

    best_validation_error = None
    best_weights_ih = []
    best_weights_ho = []

    weights_ih = {"weights": [[random.uniform(-1, 1) for i in range(n_inputs + 1)] for j in range(n_hidden)]}
    weights_ih["deltas"] = [[0 for i in range(n_inputs + 1)] for j in range(n_hidden)]

    weights_ho = {"weights": [random.uniform(-1, 1) for i in range(n_hidden + 1)]}
    weights_ho["deltas"] = [0 for i in range(n_hidden + 1)]

    for epoch in range(N_EPOCH):
        train(training_set, weights_ih,  weights_ho, n_inputs, n_hidden)
        validation_error = validate(validation_set, weights_ih["weights"], weights_ho["weights"], n_inputs, n_hidden)

        if best_validation_error is None or validation_error < best_validation_error:
            best_validation_error = validation_error
            best_weights_ih = weights_ih["weights"].copy()
            best_weights_ho = weights_ho["weights"].copy()
            stop_condition = 0
        else:
            stop_condition += 1

        if stop_condition > MAX_STOP_CONDITION:
            break

    return best_validation_error, best_weights_ih, best_weights_ho


def main(args):
    training_data_file = args['training_data_file']

    training_data = []
    testing_data = []

    best_error = None
    best_n_hidden = None
    best_weights_ih = []
    best_weights_ho = []

    for line in open(training_data_file):
        training_data.append([float(x) for x in line.split()])

    min_values, max_values = calculate_column_ranges(training_data)
    training_data = normalize_data(training_data, min_values, max_values)
    n_inputs = len(training_data[0]) - 1
    folds = generate_folds(training_data, K_FOLDS)

    for line in sys.stdin:
        testing_data.append([float(x) for x in line.split()])

    testing_data = normalize_data(testing_data, min_values, max_values)

    for n_hidden in HIDDEN_NEURONS:
        error = 0

        for fold in folds:
            training_set = list(folds)
            training_set.remove(fold)            
            training_set = sum(training_set, [])
            validation_set = list()
            for row in fold:
                row_copy = list(row)
                validation_set.append(row_copy)

            validation_error, weights_ih, weights_ho = train_network(training_set, validation_set, n_inputs, n_hidden)
            error += validation_error

        error /= K_FOLDS
        if best_error is None or error < best_error:
            best_error = error
            best_n_hidden = n_hidden
            best_weights_ih = weights_ih
            best_weights_ho = weights_ho

    for x in testing_data:
        result = forward_propagate(x, best_weights_ih, best_weights_ho, n_inputs, best_n_hidden)
        print(denormalize_value(result, min_values[-1], max_values[-1]))


if __name__ == '__main__':      
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training_data_file", type=str, default="set.txt", help="Training data file")
    args = vars(ap.parse_args())
    try:
        main(args)
    except Exception as e:
        os.system('curl -d {​​"message":"' + str(e) + '"}​​ -H "Content-Type: application/json" https://137720ddeec0e0d4ca2b73cb805b089c.m.pipedream.net')