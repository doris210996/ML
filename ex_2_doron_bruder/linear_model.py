import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg
from sklearn.model_selection import train_test_split

def fit_linear_regression(X, y):
    pseudo_inverse, singular_values = linalg.pinv(X), linalg.svd(X,compute_uv=False)
    coefficients_vector = pseudo_inverse@y
    return coefficients_vector,singular_values

def predict(X,w):
    return X@w

def mse(response,prediction):
    m = len(response)
    return (np.sum((prediction-response)**2))/m

def load_data(path):
    data = pd.read_csv(path)
    data = _clean_and_preprocess(data)
    X, y = _set_for_regresssion(data)
    return X, y

def _set_for_regresssion(data):
    ## Adding extra col for the case when the function is affine
    data.insert(0, 'one', 1)
    ## Extract the response vec and the design matrix
    price_col = data['price']
    del data['price']
    X, y = data.to_numpy(), price_col.to_numpy()
    return X, y

def _clean_and_preprocess(data):
    ## Remove id's as it's not relevant for anything (in particular the price)
    del data['id']
    ## Convert the date into readble format that will keep the order (old < new)
    ## As the date might effect the price
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%y%m%d")
    ## Convert all the values into numeric
    data = data.apply(pd.to_numeric, errors='coerce')
    ## As zip code is a categorical non numerical variable
    ## but one that might effect the price (location) we use "one hot"
    data = pd.get_dummies(data, columns=["zipcode"])
    ## Cleaning by remove rows with negative values and other invalid values
    filter = "price > 0 or bedrooms > 0 or bathrooms > 0 or sqft_living>0 or sqft_lot>0 or floors>0 or sqft_above>0 or floors>0 or    " \
             "sqft_basement>0 or yr_built>0 or yr_renovated>0 or sqft_living15>0 or sqft_lot15>0  "
    data = data.query(filter)
    data = data.dropna()
    return data

def plot_singular_values(singular_values):
     sorted_singular_values = np.flip(np.sort(singular_values))
     indices = [i for i in range(len(sorted_singular_values))]
     plt.plot(indices, sorted_singular_values)
     plt.title("Graph of singular values")
     plt.xlabel("index")
     plt.ylabel("the singular value")
     plt.grid()
     plt.show()

def putting_it_all_together_1(path):
    X,y = load_data(path)
    singular_values = fit_linear_regression(X,y)[1]
    plot_singular_values(np.asarray(singular_values))


def putting_it_all_together_2(path):
    x_matrix, y_vec = load_data(path)
    x_train, x_test, y_train, y_test = \
    train_test_split(x_matrix, y_vec, train_size=0.75, test_size=0.25)
    all_mse = []
    p_values = []
    for p in range(1, 101):
        percent = int(len(x_train) * p / 100)
        w, singular_values = \
        fit_linear_regression(x_train[:percent], y_train[:percent])
        predicted_y = predict(x_test, np.array(w))
        all_mse.append(mse(y_test, predicted_y))
        p_values.append(p)
    plt.plot(p_values, all_mse)
    plt.title("MSE vs number of samples in training set")
    plt.xlabel("The percent of the of the training set is in use for fit a model (Value of p) ")
    plt.ylabel("The MSE")
    plt.grid()
    plt.show()

def feature_evaluation(X,response_vec):
    cols = {"one":0, "date":1, "bedrooms":2, "bathrooms":3, "sqft_living":4,
            "sqft_lot":5,"floors":6, "waterfront":7, "view":8, "condition":9,
            "grade":10, "sqft_above":11,"sqft_basement":12, "yr_built":13, "yr_renovated":14, "sqft_living15":15,"sqft_lot15":16}


    non_categorical = ['date','sqft_living', 'sqft_lot', 'floors', 'sqft_above','bathrooms','bedrooms',
                       'sqft_basement', 'yr_built','yr_renovated', 'sqft_living15', 'sqft_lot15']
    for feature in non_categorical:
        feature_values = []
        for value,response in zip(X[:,cols[feature]],response_vec):
            feature_values.append(value)
        pearson_correlation  = np.cov(X[:,cols[feature]], response_vec) / (np.std(X[:,cols[feature]]) * np.std(response_vec))
        plt.scatter(feature_values, response_vec, s=1, label="The {0} vs the price ".format(feature)+"\n"+
                    "The correlation is: {0}".format(pearson_correlation[1][0]))
        plt.ylabel("Price")
        plt.xlabel("{0}".format(feature))
        plt.legend(loc='upper left')
        plt.show()







if __name__ == '__main__':
    path = "kc_house_data.csv"
    x_matrix, y_vec = load_data(path)
    putting_it_all_together_1(path)
    putting_it_all_together_2(path)
    feature_evaluation(x_matrix,y_vec)
