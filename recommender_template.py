import numpy as np
import pandas as pd
import recommender_functions as rf # Importing additional functions from another file
import time
import sys # can use sys to take command line arguments

class Recommender():
    '''
    What is this class all about - write a really good doc string here
    '''
    def __init__(self, reviews_pth, movies_pth ):
        '''
        The init function takes in the data path to be loaded in for building our recommender

        INPUTS:
        reviews_pth - path to csv with at least the four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies_pth - path to csv with each movie and movie information in each row
        Pandas read_csv function is called here. review docs incase of a read error.

        OUTPUTS:
        None - stores the following as attributes
        reviews - dataframe with four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies - dataframe of movies
        '''
        self.reviews = pd.read_csv(reviews_pth)
        self.movies = pd.read_csv(movies_pth)


    def fit(self,latent_features=4, learning_rate=0.01, iters=100):
        '''
        This function performs matrix factorization using a basic form of FunkSVD with no regularization

        INPUTS:
        latent_features - (int) number of features we want to extract from our reviews data set, default = 4
        learning_rate - (float) the rate at which we want our model to learn at, defualt = 0.01
        iters - (int) the number of iterations, default = 100

        OUTPUT:
        None - stores the following as attributes:
        n_users - the number of users (int)
        n_movies - the number of movies (int)
        num_ratings - the number of ratings made (int)
        reviews - dataframe with four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies - dataframe of
        user_item_mat - (np array) a user by item numpy array with ratings and nans for values
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations
        '''
        # Create user-by-item matrix
        user_items = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        self.user_item_mat = np.array(self.user_by_movie)

        # Storing the number of users and movies
        self.n_users = self.user_item_mat.shape[0]
        self.n_movies = self.user_item_mat.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.user_item_mat))

        # transfering our attributes into variables we can call on
        self.latent_features = latent_features
        self.learning_rate = learning_rate
        self.iters = iters

        # Setting up useful variables for future functions
        self.user_ids_series = np.array(self.user_by_movie.index)
        self.movie_ids_series = np.array(self.user_by_movie.columns)

        # Creating our random matrices of correct sizes
        user_mat = np.random.rand(self.n_users, latent_features)
        movie_mat = np.random.rand(latent_features, self.n_movies)

        # initializing our first iteration
        sse_accum = 0
        self.mean_sse = 0

        # Results headers
        print("Optimization Statistics")
        print("Iteration | Mean Squared Error")

        # starting our iterations
        for iteration in range(self.iters):
            # Update the sse_accum
            old_sse = sse_accum
            sse_accum = 0
            # Looping through each user
            for i in range(self.n_users):
                # Looping through each movie
                for j in range(self.n_movies):
                    # Checking if the value exists in the user movie rating matrix
                    if self.user_item_mat[i,j] > 0:
                        # calculating the difference between our matrix and
                        diff = self.user_item_mat[i,j] - np.dot(user_mat[i, :], movie_mat[:, j])

                        # Keeping track of our errors
                        sse_accum += diff**2

                        # updating our user and movie matrices
                        for k in range(latent_features):
                            user_mat[i,k] += learning_rate*(2*diff*movie_mat[k,j])
                            movie_mat[k,j] += learning_rate*(2*diff*user_mat[i,k])

            # print results of each iteration
            print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))
            self.mean_sse = sse_accum/self.num_ratings

        # Storing our matrices
        self.user_mat = user_mat
        self.movie_mat = user_mat

        # Knowledge based fit
        self.ranked_movies = rf.create_ranked_df(self.movies, self.reviews)

    def best_params(self, latent_features_list = [8,9,10], learning_rate_list = [0.1,0.01,0.001]):
        '''
        This function evaluates a list of parameters and returns the optimal model on 50 iterations

        INPUTS:
        latent_features - list of latent features to evaluate
        learning_rate_list - list of learning rates

        OUTPUT:

        Optimal_latent_features - returns the optimal number of latent latent_features
        Optimal_learning_rate - returns the optimal learning rate
        '''

        print("\nThis is going to take a while.")
        print("\nGrab a coffee in the meanwhile or call your mother, she misses you.\n")

        mean_sse = 20
        self.Optimal_latent_features = 0
        self.Optimal_learning_rate = 0

        user_items = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        self.user_item_mat = np.array(self.user_by_movie)

        print("\t\t\t\tOptimization Statistics")
        print("Calculation Duration | Mean Squared Error | Learning Rate | Latent Features")

        for i in latent_features_list:
            for j in learning_rate_list:

                start = time.time()

                mean_rf_sse = rf.FunkSVD(ratings_mat = self.user_item_mat, latent_features = i, learning_rate = j)

                end = time.time()

                print("%f \t\t %f \t\t %f \t\t %d" %(end-start, mean_rf_sse, j, i ))

                if mean_rf_sse < mean_sse:
                    mean_sse = mean_rf_sse
                    self.Optimal_learning_rate = j
                    self.Optimal_latent_features = i

        print("Done!\n Our best model had {} latent_features and a learning rate of {}".format(self.Optimal_latent_features, self.Optimal_learning_rate))


    def predict_rating(self, movie_id, user_id ):
        '''
        This function predicts ratings what rating a user will give a particular movie


        INPUTS:
        movie_id - the movie_id according the movies df
        user_id - the user_id from the reviews df

        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD
        '''
        try:# User row and Movie Column
            user_row = np.where(self.user_ids_series == user_id)[0][0]
            movie_col = np.where(self.movie_ids_series == movie_id)[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_col])

            movie_name = str(self.movies[self.movies['movie_id'] == movie_id]['movie']) [5:]
            movie_name = movie_name.replace('\nName: movie, dtype: object', '')
            print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(pred, 2), str(movie_name)))

            return pred

        except:
            print("I'm sorry, but a prediction cannot be made for this user-movie pair.  It looks like one of these items does not exist in our current database.")

            return None
    def make_recs(self, _id, _id_type='movie', rec_num=5):
        '''
        given a user id or a movie that an individual likes
        make recommendations

        INPUTS:
        _id - the user or movie id we want to make recommendations for
        _id_type -  a declaration of which type of id we are analyzing (default = "movie")
        num_recs - number of recommendations we want to make

        OUTPUT:
        recs - (array) a list or numpy array of recommended movies like the
                       given movie, or recs for a user_id given
        '''
        # if the user is available from the matrix factorization data,
        # I will use this and rank movies based on the predicted values
        # For use with user indexing
        rec_ids, rec_names = None, None
        if _id_type == 'user':
            if _id in self.user_ids_series:
                # Get the index of which row the user is in for use in U matrix
                idx = np.where(self.user_ids_series == _id)[0][0]

                # take the dot product of that row and the V matrix
                preds = np.dot(self.user_mat[idx,:],self.movie_mat)

                # pull the top movies according to the prediction
                indices = preds.argsort()[-rec_num:][::-1] #indices
                rec_ids = self.movie_ids_series[indices]
                rec_names = rf.get_movie_names(rec_ids, self.movies)

            else:
                # if we don't have this user, give just top ratings back
                rec_names = rf.popular_recommendations(_id, rec_num, self.ranked_movies)
                print("Because this user wasn't in our database, we are giving back the top movie recommendations for all users.")

        # Find similar movies if it is a movie that is passed
        else:
            if _id in self.movie_ids_series:
                rec_names = list(rf.find_similar_movies(_id, self.movies))[:rec_num]
            else:
                print("That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.")

        return rec_ids, rec_names

if __name__ == '__main__':
    import recommender as r

    #instantiate recommender
    rec = r.Recommender()

    # fit recommender
    rec.fit(reviews_pth='data/train_data.csv', movies_pth= 'data/movies_clean.csv', learning_rate=.01, iters=1)

    # predict
    rec.predict_rating(user_id=8, movie_id=2844)

    # make recommendations
    print(rec.make_recommendations(8,'user')) # user in the dataset
    print(rec.make_recommendations(1,'user')) # user not in dataset
    print(rec.make_recommendations(1853728)) # movie in the dataset
    print(rec.make_recommendations(1)) # movie not in dataset
    print(rec.n_users)
    print(rec.n_movies)
    print(rec.num_ratings)
