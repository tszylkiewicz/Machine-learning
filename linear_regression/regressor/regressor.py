import os
import sys
import math
import random
import argparse
import traceback


FOLDS_NUM = 5
LEARNING_RATE = 0.1
EPOCHS_NUM = 9000
POLYNOMIAL_DEGREES = [2, 3, 4, 5, 6, 7, 8]


def calculate_min_max(data):
    min_values, max_values = data[0], data[0] 
    
    for row in data[1:]:
        max_values = [max(x, y) for x, y in zip(max_values, row)]
        min_values = [min(x, y) for x, y in zip(min_values, row)]

    return min_values, max_values


def normalize_value(value, min_value, max_value):
    result = (value - min_value) / (max_value - min_value)
    result = (2 * result) - 1
    return result


def denormalize_value(value, min_value, max_value):
    result = (value + 1) / 2
    result = result * (max_value - min_value) + min_value
    return result


def normalize_data(data, min_values, max_values):
    for i in range(len(data[0])):
        for j in range(len(data)):
            data[j][i] = normalize_value(data[j][i], min_values[i], max_values[i])

    return data


def generate_folds(data):
    dataset_split = list()
    dataset_copy = list(data)
    fold_size = int(len(data) / FOLDS_NUM)
    for i in range(FOLDS_NUM):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def vector_power(x, value):
    return [x[i] ** value for i in range(len(x))]


def vector_substract(x, y):
    return [x[i] - y[i] for i in range(len(x))]


def vector_multiply(x, value):
    return [x[i] * value for i in range(len(x))]


def matrix_transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


def matrix_dot(x, y):
    return [sum([x[i][j] * y[j] for j in range(len(y))]) for i in range(len(x))]


def vector_dot(x, y):
    return sum([x[i] * y[i] for i in range(len(y))])


def sums(length, total_sum):
    if length == 1:
        yield (total_sum,)
    else:
        for value in range(total_sum + 1):
            for permutation in sums(length - 1, total_sum - value):
                yield (value,) + permutation


def create_permutations(n, k):
    perms = []
    for i in range(1, k):
        perms += list(sums(n, i))
    return perms


def create_input(perms, inputs):
    new_input = []
    for perm in perms:
        val = 1
        for i in range(len(perm)):
            if(perm[i] != 0):
                val *= inputs[i] ** perm[i]
        new_input.append(val)
    new_input.append(1)
    return new_input


def calculate(inputs, weights, k):
    perms = create_permutations(len(inputs), k)
    new_input = create_input(perms, inputs)    
    return vector_dot(new_input, weights)


def generate_inputs(inputs, k):  
    perms = create_permutations(len(inputs[0]), k)
    
    new_inputs = []
    for input in inputs:
        new_input = create_input(perms, input)          
        new_inputs.append(new_input)
    return new_inputs


def divide_dataset(dataset):
    target = []
    target_output = []    

    for row in dataset.copy():
        target_output.append(row[-1])
        target.append(row[:-1])

    return target, target_output


def cross_validation(training_set, validation_set, polynomial_degree):
    train_target, train_target_output = divide_dataset(training_set) 
    validate_target, validate_target_output = divide_dataset(validation_set)   

    train_data = generate_inputs(train_target, polynomial_degree)
    validate_data = generate_inputs(validate_target, polynomial_degree)

    weights = [random.uniform(-1, 1) for col in range(len(train_data[0]))]

    validation_cost = None
    prev_cost = None
    best_cost = None

    for epoch in range(EPOCHS_NUM):        
        train_prediction = matrix_dot(train_data, weights)
        train_loss = vector_substract(train_prediction, train_target_output)
        gradient = matrix_dot(matrix_transpose(train_data), train_loss)
        weights = vector_substract(weights, vector_multiply(gradient, LEARNING_RATE / len(train_target_output)))

        validation_prediction = matrix_dot(validate_data, weights)
        validation_loss = vector_substract(validation_prediction, validate_target_output)
        validation_cost = sum(vector_power(validation_loss, 2)) / len(validate_target_output)

        if prev_cost is None or validation_cost < prev_cost:          
            best_cost = validation_cost 
       
        prev_cost = validation_cost

    return best_cost


def full_training(training_set, polynomial_degree):
    train_target, train_target_output = divide_dataset(training_set)
    train_data = generate_inputs(train_target, polynomial_degree)
    weights = [random.uniform(-1, 1) for col in range(len(train_data[0]))]

    cost = None
    prev_cost = None
    best_weights = None

    m = len(train_target_output)

    for epoch in range(EPOCHS_NUM):        
        train_prediction = matrix_dot(train_data, weights)
        train_loss = vector_substract(train_prediction, train_target_output)
        gradient = matrix_dot(matrix_transpose(train_data), train_loss)
        weights = vector_substract(weights, vector_multiply(gradient, LEARNING_RATE / m))
        cost = sum(vector_power(train_loss, 2)) / m

        if prev_cost is None or cost < prev_cost:                   
            best_weights = weights
       
        prev_cost = cost

    return best_weights


def main(args):
    training_data_file = args['training_data_file']

    training_data = []
    testing_data = []

    for line in open(training_data_file):
        training_data.append([float(x) for x in line.split()])

    for line in sys.stdin:
        testing_data.append([float(x) for x in line.split()])

    min_values, max_values = calculate_min_max(training_data)

    training_data = normalize_data(training_data, min_values, max_values)
    testing_data = normalize_data(testing_data, min_values, max_values)

    folds = generate_folds(training_data)

    all_errors = []

    for polynomial_degree in POLYNOMIAL_DEGREES:
        error = 0

        for fold in folds:
            training_set = list(folds.copy())
            training_set.remove(fold)
            training_set = sum(training_set, [])
            validation_set = list()
            for row in fold:
                row_copy = list(row.copy())
                validation_set.append(row_copy)

            error += cross_validation(training_set, validation_set, polynomial_degree)        

        error /= FOLDS_NUM
        all_errors.append(error)

    index = all_errors.index(min(all_errors))
    k = POLYNOMIAL_DEGREES[int(index)]
    best_weights = full_training(training_data, k)

    for x in testing_data:
        result = calculate(x, best_weights, k)
        print(denormalize_value(result, min_values[-1], max_values[-1]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--training_data_file", type=str,
                    default="set.txt", help="Training data file")
    args = vars(ap.parse_args())    
    main(args)    