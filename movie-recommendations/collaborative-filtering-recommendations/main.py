import argparse
from math import sqrt
import numpy as np
import pandas as pd
from scipy import spatial
import operator
from ast import literal_eval
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

K = 5
N_EPOCH = 20
LEARNING_RATE = 0.4

# def calculate_gradient(values, loss, mask):
#     result = []

#     for i in range(len(values)):
#         arr = []
#         for j in range(len(loss[0])):
#             value = 0
#             iter = 0
#             for k in range(len(loss)):
#                 if mask[k, j]:   # liczenie gradientu tylko dla ocenionych filmów
#                     value += values[i, k] * loss[k, j]
#                     iter += 1
#             arr.append(value / iter)
#         result.append(np.array(arr))

#     return np.array(result)


# def calculate_cost(loss):
#     m = 0
#     result = 0
#     for i in range(len(loss)):
#         for j in range(len(loss[i])):
#             # sprawdzenie czy nie jest NaN: NaN == NaN = false (w przypadku braku oceny)
#             if loss[i, j] == loss[i, j]:
#                 m += 1
#                 result += loss[i, j] ** 2
#     return result / m / 2


# def gradient_descent(inputs, expected, weights, mask):
#     costs = []
#     for i in range(N_EPOCH):
#         outputs = inputs.dot(weights)
#         loss = outputs - expected

#         gradient_i = calculate_gradient(weights, loss.T, mask.T)
#         gradient_w = calculate_gradient(inputs.T, loss, mask)

#         weights = weights - gradient_w * LEARNING_RATE
#         inputs = inputs - gradient_i.T * LEARNING_RATE

#         cost = calculate_cost(loss)

#         for j in range(len(inputs)):
#             inputs[j][-1] = 1
#         costs.append(cost)
#         print(f'\rEpoch {i+1}/{N_EPOCH}', end='\r')

#     return weights, inputs, costs


def predictions(P, Q):
    return np.dot(P.T, Q)

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


def predict(X_train, user_index, P, Q):
    y_hat = predictions(P, Q)
    predictions_index = np.where(X_train[user_index, :] == 0)[0]
    return y_hat[user_index, predictions_index].flatten()


def main(args):
    train_file = args['train']
    task_file = args['task']
    submission_file = args['submission']

    train = pd.read_csv(train_file, sep=';', names=["id", "user_id", "movie_id", "rating"])
    task = pd.read_csv(task_file, sep=';', names=["id", "user_id", "movie_id", "rating"])

    train['rating'] = train['rating'] / 5

    ratings_matrix = train.pivot(index='user_id',columns='movie_id',values='rating')
    
    n_users, n_movies = ratings_matrix.shape  
    lmbda=0.1
    # inputs = np.random.uniform(0, 1, [n_movies, K])
    # weights = np.random.uniform(0, 1, [K, n_users])

    P = np.random.uniform(0, 1, [K, n_users])
    Q = np.random.uniform(0, 1, [K, n_movies])

    scores = ratings_matrix.fillna(0).to_numpy()
    
    # scores_mask = (~ratings_matrix.isnull()).to_numpy()

    # for value in inputs:
    #     value[-1] = 1

    train_error = []
    users, movies = scores.nonzero()

    for epoch in range(N_EPOCH):
        for u, i in zip(users, movies):
            error = scores[u, i] - predictions(P[:,u], Q[:,i])
            P[:, u] += LEARNING_RATE * (error * Q[:, i] - lmbda * P[:, u])
            Q[:, i] += LEARNING_RATE * (error * P[:, u] - lmbda * Q[:, i])

        train_rmse = rmse(predictions(P, Q), scores)
        train_error.append(train_rmse)
        print(f'\rEpoch {epoch+1}/{N_EPOCH}', end='\r')
    # weights, inputs, costs = gradient_descent(inputs, scores, weights, scores_mask)
        
    # plt.plot(costs[10:])
    # plt.show()

    outputs = predictions(P, Q)
    ratings_matrix = pd.DataFrame.from_records(outputs, columns=ratings_matrix.columns)

    for index, row in task.iterrows():
        print(str(index) + " / " + str(len(task.index)), end='\r')     
        user = int(row['user_id'])
        movie = int(row['movie_id'])
        print(user)
        print(movie)
        score = ratings_matrix.loc[user, movie] * 5
        if score < 0:
            score = 0
        if score > 5:
            score = 5   
         
        task.loc[index, 'rating'] = str(int(round(score)))        

    task.to_csv(submission_file, sep=';', index=False, header=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", type=str,
                    default="train.csv", help="Train data file")
    ap.add_argument("-e", "--task", type=str,
                    default="task.csv", help="Task data file")
    ap.add_argument("-s", "--submission", type=str,
                    default="submission.csv", help="Submission data file")
    args = vars(ap.parse_args())
    main(args)