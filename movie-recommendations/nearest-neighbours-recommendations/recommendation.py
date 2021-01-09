import argparse
import pandas as pd
from scipy import spatial
import operator
from ast import literal_eval

K = 9

def normalize_value(value, min_value, max_value):
    result = (value - min_value) / (max_value - min_value)    
    return result


def binary(values_list, unique_list):
    binaryList = []
    
    for unique in unique_list:
        if unique in values_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList


def prepare_dataset(movies): 
    all_genres = []
    all_directors = []
    all_keywords = []
    all_actors = []
    all_countries = []
    
    budget_min, budget_max = movies['budget'].min(), movies['budget'].max()
    popularity_min, popularity_max = movies['popularity'].min(), movies['popularity'].max()
    runtime_min, runtime_max = movies['runtime'].min(), movies['runtime'].max()

    for i,j in zip(movies['cast'], movies.index):
        list2 = []
        list2 = i[:5]
        movies.loc[j,'cast'] = str(list2)
    movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')
    movies['cast'] = movies['cast'].str.split(',')

    for index, row in movies.iterrows():        
        for genre in row['genres']:
            if genre not in all_genres:
                all_genres.append(genre)
        
        for keyword in row['keywords']:
            if keyword not in all_keywords:
                all_keywords.append(keyword)

        for actor in row['cast']:
            if actor not in all_actors:
                all_actors.append(i)

        for country in row['production_countries']:
            if country not in all_countries:
                all_countries.append(country)

        if row['director'] not in all_directors:
            all_directors.append(row['director'])

    movies['genres_bin'] = movies['genres'].apply(lambda x: binary(x, all_genres))
    movies['director_bin'] = movies['director'].apply(lambda x: binary(x, all_directors))
    movies['keywords_bin'] = movies['keywords'].apply(lambda x: binary(x, all_keywords))
    movies['actors_bin'] = movies['cast'].apply(lambda x: binary(x, all_actors))
    movies['country_bin'] = movies['production_countries'].apply(lambda x: binary(x, all_countries))

    movies['budget_norm'] = movies['budget'].apply(lambda x: normalize_value(x, budget_min, budget_max))
    movies['popularity_norm'] = movies['popularity'].apply(lambda x: normalize_value(x, popularity_min, popularity_max))
    movies['runtime_norm'] = movies['runtime'].apply(lambda x: normalize_value(x, runtime_min, runtime_max))
    
    return movies


def similarity(movies, movie_id_1, movie_id_2):
    a = movies.iloc[movie_id_1 - 1]
    b = movies.iloc[movie_id_2 - 1]
    
    genres_dist = spatial.distance.cosine(a['genres_bin'], b['genres_bin'])
    actors_dist = spatial.distance.cosine(a['actors_bin'], b['actors_bin'])
    director_dist = spatial.distance.cosine(a['director_bin'], b['director_bin'])
    keywords_dist = spatial.distance.cosine(a['keywords_bin'], b['keywords_bin'])
    country_dist = spatial.distance.cosine(a['country_bin'], b['country_bin'])

    budget_dist = abs(a['budget_norm'] - b['budget_norm'])
    popularity_dist = abs(a['popularity_norm'] - b['popularity_norm'])
    runtime_dist = abs(a['runtime_norm'] - b['runtime_norm'])

    return genres_dist + director_dist + actors_dist + keywords_dist + country_dist + budget_dist + popularity_dist + runtime_dist


def calculate_neighbours(movies, row, rated_movies):
        distances = []
        for index, movie in rated_movies.iterrows():
            dist = similarity(movies, int(row['movie_id']), int(movie['movie_id']))
            distances.append((int(movie['movie_id']), dist, movie['evaluation']))
    
        distances.sort(key=operator.itemgetter(1))
        neighbours = []
    
        for x in range(K):
            neighbours.append(distances[x])
        return neighbours


def main(args): 
    movies_file = args['movies_data']
    train_file = args['train']
    task_file = args['task']
    submission_file = args['submission']

    movies = pd.read_csv(movies_file, converters={'genres':literal_eval, 'cast':literal_eval, 'keywords':literal_eval, 'production_countries': literal_eval})    
    train = pd.read_csv(train_file, sep=';', names=["id", "user_id", "movie_id", "evaluation"])
    task = pd.read_csv(task_file, sep=';', names=["id", "user_id", "movie_id", "evaluation"])

    movies = prepare_dataset(movies)

    for index, row in task.iterrows():
        print(str(index) + " / " + str(len(task.index)), end ='\r')
        neighbours = calculate_neighbours(movies, row, train.loc[train['user_id'] == row["user_id"]])
        neighbours_df = pd.DataFrame(neighbours, columns=['movie_id', 'dist', 'evaluation'])
        task.loc[index, 'evaluation'] = str(int(round(neighbours_df['evaluation'].mean())))

    task.to_csv(submission_file, sep=';', index=False, header=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--movies_data", type=str,
                    default="movies.csv", help="Movies data file")
    ap.add_argument("-t", "--train", type=str,
                    default="train.csv", help="Train data file")
    ap.add_argument("-e", "--task", type=str,
                    default="task.csv", help="Task data file")
    ap.add_argument("-s", "--submission", type=str,
                    default="submission.csv", help="Submission data file")
    args = vars(ap.parse_args())
    main(args)