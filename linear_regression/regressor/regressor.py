import sys
import math
import random
import argparse


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
            data[j][i] = normalize_value(
                data[j][i], min_values[i], max_values[i])

    return data


def generate_folds(data, k_folds):
    random.shuffle(data)

    avg = len(data) / float(k_folds)
    last = 0.0

    folds = []

    while last < len(data):
        folds.append(data[int(last):int(last + avg)])
        last += avg

    return folds


def generate_inputs_outputs(folds, k_folds):
    training_inputs = []
    training_outputs = []
    validation_outputs = []

    for fold in folds:
        validation_fold_outputs = []
        for x in fold:
            validation_fold_outputs.append(x.pop())
        validation_outputs.append(validation_fold_outputs)

    validation_inputs = folds

    for i in range(k_folds):
        training_fold_inputs = []
        training_fold_outputs = []

        for j in range(k_folds):
            if i == j:
                continue
            training_fold_inputs += validation_inputs[j]
            training_fold_outputs += validation_outputs[j]
        training_inputs.append(training_fold_inputs)
        training_outputs.append(training_fold_outputs)

    return training_inputs, training_outputs, validation_inputs, validation_outputs


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def derivative_sigmoid(x):
    return x * (1 - x)


def m_dot_v_v(x, y):
    return sum([x[i] * y[i] for i in range(len(y))])

# def activate(inputs, weights):
#     activation = weights[-1]
#     for i in range(len(weights) - 1):
# 		activation += weights[i] * inputs[i]
# 	return activation



def train(training_set, weights_ih, weights_ih_delta, weights_ho, weights_ho_delta, n_inputs, n_hidden, momentum, learning_rate):   
    hidden_values = [0 for i in range(n_hidden)]

    for sample in range(len(training_set)):
        input_values = training_set[sample]

        for node in range(n_hidden):
            hidden_values[node] = m_dot_v_v(
                input_values[:n_inputs], weights_ih[node][:n_hidden])

            hidden_values[node] = sigmoid(
                hidden_values[node] + weights_ih[node][-1])

        output_value = m_dot_v_v(hidden_values, weights_ho[:n_hidden])
        output_value += weights_ho[-1]

        hidden_gradient = output_value - input_values[-1]

        for hidden_node in range(n_hidden):
            input_gradient = hidden_gradient * weights_ho[hidden_node] * derivative_sigmoid(hidden_values[hidden_node])

            for input_node in range(n_inputs):
                prev_in_delta = weights_ih_delta[hidden_node][input_node]
                current_in_delta = -learning_rate * input_gradient * input_values[input_node]
                weights_ih_delta[hidden_node][input_node] = current_in_delta
                weights_ih[hidden_node][input_node] += current_in_delta + (momentum * prev_in_delta)

            prev_in_delta = weights_ih_delta[hidden_node][-1]
            current_in_delta = -learning_rate * input_gradient
            weights_ih_delta[hidden_node][-1] = current_in_delta
            weights_ih[hidden_node][-1] += current_in_delta + (momentum * prev_in_delta)

            prev_hid_delta = weights_ho_delta[hidden_node]
            current_hid_delta = -learning_rate * hidden_gradient * hidden_values[hidden_node]
            weights_ho_delta[hidden_node] = current_hid_delta
            weights_ho[hidden_node] += current_hid_delta + (momentum * prev_hid_delta)

        prev_hid_delta = weights_ho_delta[-1]
        current_hid_delta = -learning_rate * hidden_gradient
        weights_ho_delta[-1] = current_hid_delta
        weights_ho[-1] += current_hid_delta + (momentum * prev_hid_delta)


def validate(validation_set, weights_ih, weights_ho, n_inputs, n_hidden):    
    hidden_values = [0 for i in range(n_hidden)]
    mse = 0
    for sample in range(len(validation_set)):
        input_values = validation_set[sample]
        for node in range(n_hidden):
            hidden_values[node] = m_dot_v_v(input_values[:n_inputs], weights_ih[node][:n_inputs])
            hidden_values[node] = sigmoid(hidden_values[node] + weights_ih[node][-1])
        output_value = m_dot_v_v(hidden_values, weights_ho[:n_hidden])
        output_value += weights_ho[-1]
        error = output_value - input_values[-1]
        mse += error ** 2
    return mse / len(validation_set) / 2


def train_network(training_set, validation_set, n_inputs, n_hidden):
    learning_rate = 0.03
    momentum = 0.9
    n_epoch = 1000

    validation_error = 0
    prev_validation_error = None
    error_counter = 0

    best_weights_ih = []
    best_weights_ho = []

    weights_ih = [[random.uniform(-1, 1) for i in range(n_inputs + 1)] for j in range(n_hidden)]
    weights_ho = [random.uniform(-1, 1) for i in range(n_hidden + 1)]

    weights_ih_delta = [[0 for i in range(n_inputs + 1)] for j in range(n_hidden)]
    weights_ho_delta = [0 for i in range(n_hidden + 1)]

    for n in range(n_epoch):
        train(training_set, weights_ih, weights_ih_delta, weights_ho, weights_ho_delta, n_inputs, n_hidden, momentum, learning_rate)
        validation_error = validate(validation_set, weights_ih, weights_ho, n_inputs, n_hidden)

        if prev_validation_error is not None and validation_error > prev_validation_error:
            error_counter += 1
        else:
            error_counter = 0
            best_weights_ih = weights_ih
            best_weights_ho = weights_ho
        if error_counter > 100:
            break
        prev_validation_error = validation_error

    return validation_error, best_weights_ih, best_weights_ho


def calculate(value, weights_ih, weights_ho, n_inputs, n_hidden):
    hidden_values = [0 for col in range(n_hidden)]
    for node in range(n_hidden):
        hidden_values[node] = m_dot_v_v(value, weights_ih[node][:n_inputs])
        hidden_values[node] = sigmoid(hidden_values[node] + weights_ih[node][-1])
    output_value = m_dot_v_v(hidden_values, weights_ho[:n_hidden])
    output_value += weights_ho[-1]
    return output_value


def main(args):
    training_data_file = args['training_data_file']

    k_folds = 5
    n_hiddens = [1, 2, 3, 4, 5, 6, 7, 8]

    training_data = []
    testing_data = []

    for line in open(training_data_file):
        training_data.append([float(x) for x in line.split()])

    for line in sys.stdin:
        testing_data.append([float(x) for x in line.split()])

    min_values, max_values = calculate_column_ranges(training_data)

    training_data = normalize_data(training_data, min_values, max_values)
    testing_data = normalize_data(testing_data, min_values, max_values)

    folds = generate_folds(training_data, k_folds)

    n_inputs = len(training_data[0])

    all_best_weights_ih = []
    all_best_weights_ho = []
    all_errors = []

    for n_hidden in n_hiddens:
        error = 0

        for fold in folds:
            training_set = list(folds)
            training_set.remove(fold)
            training_set = sum(training_set, [])
            validation_set = list()
            for row in fold:
                row_copy = list(row)
                validation_set.append(row_copy)

            validation_error, best_weights_ih, best_weights_ho = train_network(training_set, validation_set, n_inputs, n_hidden)
            error += validation_error

        error /= k_folds
        all_errors.append(error)
        all_best_weights_ih.append(best_weights_ih)
        all_best_weights_ho.append(best_weights_ho)

    index = all_errors.index(min(all_errors))
    best_n_hidden = n_hiddens[index]
    best_weights_ih = all_best_weights_ih[index]
    best_weights_ho = all_best_weights_ho[index]

    for x in testing_data:
        result = calculate(x, best_weights_ih,best_weights_ho, n_inputs, best_n_hidden)
        print(denormalize_value(result, min_values[-1], max_values[-1]))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training_data_file", type=str, default="set.txt", help="Training data file")
    args = vars(ap.parse_args())
    main(args)
