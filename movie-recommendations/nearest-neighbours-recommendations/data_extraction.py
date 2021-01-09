import tmdbsimple as tmdb
from numpy import asarray
from numpy import savetxt
import argparse
import sys

tmdb.API_KEY = '16c758c436a84bafd8d42e9b8afc27f7'


def create_movie_data(movie):
    info = movie.info()
    credits = movie.credits()
    keywords_list = movie.keywords()
    movie_data = []

    movie_data.append(info['id'])
    movie_data.append(info['budget'])

    genres = []
    for i in info['genres']:
        genres.append((i['name']))
    movie_data.append(genres)

    movie_data.append(info['original_language'])
    movie_data.append(info['title'])
    movie_data.append(info['overview'])
    movie_data.append(info['popularity'])

    production_companies = []
    for i in info['production_companies']:
        production_companies.append((i['name']))
    movie_data.append(production_companies)

    production_countries = []
    for i in info['production_countries']:
        production_countries.append((i['name']))
    movie_data.append(production_countries)

    movie_data.append(info['release_date'])
    movie_data.append(info['revenue'])
    movie_data.append(info['runtime'])
    movie_data.append(info['vote_average'])

    for i in credits['crew']:
        if i['job'].lower() == 'director':
            movie_data.append((i['name']))
            break

    cast = []
    for i in credits['cast']:
        cast.append((i['name']))
    movie_data.append(cast)

    keywords = []
    for i in keywords_list['keywords']:
        keywords.append((i['name']))
    movie_data.append(keywords)

    return movie_data


def main(args):
    movie_file = args['movie_set']
    movie_ids = []

    for line in open(movie_file):
        movie_ids.append(int(line.split(';')[1]))

    # movie = tmdb.Movies(389)
    # movie = create_movie_data(movie)
    # print(movie.info())
    # print(movie.credits())
    # print(movie)
    movies_data = []
    movies_data.append(['id', 'budget', 'genres', 'original_language', 'title', 'overview', 'popularity', 'production_companies',
                        'production_countries', 'release_date', 'revenue', 'runtime', 'vote_average', 'director', 'cast', 'keywords'])

    for movie_id in movie_ids:
        movie = tmdb.Movies(movie_id)
        movie_data = create_movie_data(movie)
        movies_data.append(movie_data)
    
    data = asarray(movies_data)
    # save to csv file
    savetxt('data.csv', data, delimiter=';')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--movie_set", type=str,
                    default="movie.csv", help="Movie set file")
    args = vars(ap.parse_args())
    main(args)
