import argparse
from math import sqrt
import pandas as pd
from scipy import spatial
import operator
from ast import literal_eval

K = 4


def calculate_rating(row, similar_users):
    ratings = [] 
    users_movie_ratings = train.loc[train['movie_id'] == row['movie_id'], ['user_id', 'rating']]     
    for user_id, similarity in similar_users.items():
        value = users_movie_ratings.loc[(train['user_id'] == user_id), 'rating']            
        if len(value) > 0:
            ratings.append(value.iloc[0])
        if(len(ratings) > K):
            break

    result = ratings[:K]
    if(len(result) == 0):
        return 3
    return sum(result) / len(result)


def main(args):
    train_file = args['train']
    task_file = args['task']
    submission_file = args['submission']

    global train
    train = pd.read_csv(train_file, sep=';', names=["id", "user_id", "movie_id", "rating"])
    task = pd.read_csv(task_file, sep=';', names=["id", "user_id", "movie_id", "rating"])

    new_df = train.pivot(index='movie_id',columns='user_id',values='rating')
    correlated_users = new_df.corr(method ='pearson')

    for index, row in task.iterrows():
        print(str(index) + " / " + str(len(task.index)), end='\r')        
        similar_users = correlated_users[row['user_id']].copy()
        similar_users = similar_users.drop(labels=row['user_id']).dropna()
        similar_users.sort_values(ascending=False, inplace=True)
        score = calculate_rating(row, similar_users)    
         
        task.loc[index, 'evaluation'] = str(int(round(score)))        

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