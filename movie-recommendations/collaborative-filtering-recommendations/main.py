import argparse
from math import sqrt
import numpy as np
import pandas as pd
from scipy import spatial
import operator
from ast import literal_eval
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

K = 9
N_EPOCH = 600
LMBDA = 0.01
LEARNING_RATE = 0.001

def predictions(P, Q):
    return np.dot(P.T, Q)


def rmse(prediction, ground_truth):
    prediction = prediction[np.where(ground_truth != -1)].flatten() 
    ground_truth = ground_truth[np.where(ground_truth != -1)].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


def main(args):
    train_file = args['train']
    task_file = args['task']
    submission_file = args['submission']

    train = pd.read_csv(train_file, sep=';', names=["id", "user_id", "movie_id", "rating"])
    task = pd.read_csv(task_file, sep=';', names=["id", "user_id", "movie_id", "rating"])

    train['rating'] = train['rating'] / 5

    ratings_matrix = train.pivot(index='user_id',columns='movie_id',values='rating')
    
    n_users, n_movies = ratings_matrix.shape      

    P = np.random.uniform(0, 1, [K, n_users])
    Q = np.random.uniform(0, 1, [K, n_movies])

    scores = ratings_matrix.fillna(-1).to_numpy()

    train_error = []
    users, movies = np.where(scores != -1)

    for epoch in range(N_EPOCH):
        for user, movie in zip(users, movies):
            error = scores[user, movie] - predictions(P[:,user], Q[:,movie])
            P[:, user] += LEARNING_RATE * (error * Q[:, movie] - LMBDA * P[:, user])
            Q[:, movie] += LEARNING_RATE * (error * P[:, user] - LMBDA * Q[:, movie])

        train_rmse = rmse(predictions(P, Q), scores)
        train_error.append(train_rmse)
        print(f'\rEpoch {epoch+1}/{N_EPOCH}', end='\r')

    outputs = predictions(P, Q) 
    test_output = pd.DataFrame.from_records(outputs, columns=ratings_matrix.columns)    

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