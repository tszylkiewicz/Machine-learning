import tmdbsimple as tmdb
import argparse
import pandas as pd

tmdb.API_KEY = '16c758c436a84bafd8d42e9b8afc27f7'


def prepare_movie_data(movie_file):
    movie_list = pd.read_csv(movie_file, sep=';', names=[
                             "pk", "movie_id", "title"])
    movies = pd.DataFrame(
        columns=['id', 'genres', 'cast', 'vote_average', 'director', 'keywords', 'budget', 'popularity', 'production_countries', 'runtime'])

    def map_name(a): return a['name']
    def map_original_name(a): return a['original_name']

    for index, row in movie_list.iterrows():
        movie = tmdb.Movies(row['movie_id'])
        movie_info = movie.info()
        movie_credits = movie.credits()
        movie_keywords = movie.keywords()
        
        id = row['pk']
        vote_average = movie_info['vote_average']
        budget = movie_info['budget']
        popularity = movie_info['popularity']
        runtime = movie_info['runtime']

        genres = list(map(map_name, movie_info['genres']))
        cast = list(map(map_original_name, movie_credits['cast']))
        production_countries = list(map(map_name, movie_info['production_countries']))
        keywords = list(map(map_name, movie_keywords['keywords']))
        
        director = None
        for crewmate in movie_credits['crew']:
            if(crewmate['job'] == "Director"):
                director = crewmate['original_name']
        
        movies.loc[id - 1] = [id, genres, cast, vote_average, director, keywords, budget, popularity, production_countries, runtime]
    movies.to_csv('movies.csv', index=False)


def main(args):
    movie_file = args['movie_list']
    prepare_movie_data(movie_file)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--movie_list", type=str,
                    default="movie.csv", help="Movie list file")
    args = vars(ap.parse_args())
    main(args)
