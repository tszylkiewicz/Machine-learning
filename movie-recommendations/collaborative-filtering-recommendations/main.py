import argparse
from math import sqrt
import numpy as np
import pandas as pd
from scipy import spatial
import operator
from ast import literal_eval
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

K = 6
N_EPOCH = 600
LEARNING_RATE = 0.01

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

    P = np.random.uniform(0, 1, [K, n_users])
    Q = np.random.uniform(0, 1, [K, n_movies])

    scores = ratings_matrix.fillna(0).to_numpy()

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

    outputs = predictions(P, Q) 
    test_output = pd.DataFrame.from_records(outputs, columns=ratings_matrix.columns)
    print(ratings_matrix)
    print(test_output)

    for index, row in task.iterrows():
        print(str(index) + " / " + str(len(task.index)), end='\r')          
        user = ratings_matrix.index.get_loc(row['user_id'])
        movie = row['movie_id']
        score = test_output.loc[user, movie] * 5
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