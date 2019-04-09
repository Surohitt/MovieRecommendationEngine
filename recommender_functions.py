import numpy as np
import pandas as pd

def get_movie_names(movie_ids, movies_df):
    '''
    INPUT
    movie_ids - a list of movie_ids
    movies_df - original movies dataframe
    OUTPUT
    movies - a list of movie names associated with the movie_ids

    '''
    # Read in the datasets
    movie_lst = list(movies_df[movies_df['movie_id'].isin(movie_ids)]['movie'])

    return movie_lst


def create_ranked_df(movies, reviews):
        '''
        INPUT
        movies - the movies dataframe
        reviews - the reviews dataframe

        OUTPUT
        ranked_movies - a dataframe with movies that are sorted by highest avg rating, more reviews, then time, and must have more than 4 ratings
        '''

        # Pull the average ratings and number of ratings for each movie
        movie_ratings = reviews.groupby('movie_id')['rating']
        avg_ratings = movie_ratings.mean()
        num_ratings = movie_ratings.count()
        last_rating = pd.DataFrame(reviews.groupby('movie_id').max()['date'])
        last_rating.columns = ['last_rating']

        # Add Dates
        rating_count_df = pd.DataFrame({'avg_rating': avg_ratings, 'num_ratings': num_ratings})
        rating_count_df = rating_count_df.join(last_rating)

        # merge with the movies dataset
        movie_recs = movies.set_index('movie_id').join(rating_count_df)

        # sort by top avg rating and number of ratings
        ranked_movies = movie_recs.sort_values(['avg_rating', 'num_ratings', 'last_rating'], ascending=False)

        # for edge cases - subset the movie list to those with only 5 or more reviews
        ranked_movies = ranked_movies[ranked_movies['num_ratings'] > 4]

        return ranked_movies


def find_similar_movies(movie_id, movies_df):
    '''
    INPUT
    movie_id - a movie_id
    movies_df - original movies dataframe
    OUTPUT
    similar_movies - an array of the most similar movies by title
    '''
    # dot product to get similar movies
    movie_content = np.array(movies_df.iloc[:,4:])
    dot_prod_movies = movie_content.dot(np.transpose(movie_content))

    # find the row of each movie id
    movie_idx = np.where(movies_df['movie_id'] == movie_id)[0][0]

    # find the most similar movie indices - to start I said they need to be the same for all content
    similar_idxs = np.where(dot_prod_movies[movie_idx] == np.max(dot_prod_movies[movie_idx]))[0]

    # pull the movie titles based on the indices
    similar_movies = np.array(movies_df.iloc[similar_idxs, ]['movie'])

    return similar_movies


def popular_recommendations(user_id, n_top, ranked_movies):
    '''
    INPUT:
    user_id - the user_id (str) of the individual you are making recommendations for
    n_top - an integer of the number recommendations you want back
    ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time

    OUTPUT:
    top_movies - a list of the n_top recommended movies by movie title in order best to worst
    '''

    top_movies = list(ranked_movies['movie'][:n_top])

    return top_movies

def FunkSVD(ratings_mat, latent_features=4, learning_rate=0.0001, iters=50):
    '''
    This function performs matrix factorization using a basic form of FunkSVD with no regularization

    INPUT:
    ratings_mat - (numpy array) a matrix with users as rows, movies as columns, and ratings as values
    latent_features - (int) the number of latent features used
    learning_rate - (float) the learning rate
    iters - (int) the number of iterations

    OUTPUT:
    user_mat - (numpy array) a user by latent feature matrix
    movie_mat - (numpy array) a latent feature by movie matrix
    '''

    # Set up useful values to be used through the rest of the function
    n_users = ratings_mat.shape[0]
    n_movies = ratings_mat.shape[1]
    num_ratings = np.count_nonzero(~np.isnan(ratings_mat))

    # initialize the user and movie matrices with random values
    user_mat = np.random.rand(n_users, latent_features)
    movie_mat = np.random.rand(latent_features, n_movies)

    # initialize sse at 0 for first iteration
    sse_accum = 0

    # for each iteration
    for iteration in range(iters):

        # update our sse
        old_sse = sse_accum
        sse_accum = 0

        # For each user-movie pair
        for i in range(n_users):
            for j in range(n_movies):

                # if the rating exists
                if ratings_mat[i, j] > 0:

                    # compute the error as the actual minus the dot product of the user and movie latent features
                    diff = ratings_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                    # Keep track of the sum of squared errors for the matrix
                    sse_accum += diff**2

                    # update the values in each matrix in the direction of the gradient
                    for k in range(latent_features):
                        user_mat[i, k] += learning_rate * (2*diff*movie_mat[k, j])
                        movie_mat[k, j] += learning_rate * (2*diff*user_mat[i, k])

            mean_sse = sse_accum/num_ratings
    return mean_sse
