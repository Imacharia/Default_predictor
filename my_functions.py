import calendar
import numpy as np
import pandas as pd 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def clean_data(df):
    # We drop Id column, sqft_living15, sqft_lot15 columns because Id column is not a relevant column for house price predictions.
    # sqft_living15, sqft_lot15 these two columns are not relevant in the model that we intent to develop (multiple linear regression
    # model since they are better suited for KNN models)
    col_to_drop = ["id"]
    df.drop(col_to_drop, axis =1, inplace = True)
    
    # Fill missing values for waterfront
    #Fill missing values for view
    #Fill missing values for yr_renovated
    # We assumed that yr_renovated missing values indicates that the house has never been renovated. 
    
    df['waterfront'].fillna(df['waterfront'].mode()[0], inplace=True)
    df['view'].fillna(df['view'].mode()[0], inplace=True)
    df['yr_renovated'].fillna(0, inplace=True)

    # Replace '?' in sqft_basement with 0.0
    df['sqft_basement'] = df.sqft_basement.replace('?', 0.0)
    
    # Convert sqft_basement to float
    df['sqft_basement']= df.sqft_basement.astype('float64')
    
    # Convert date column to 2 separate columns for month and year
    date = df['date'].str.split('/', expand=True)
    df['month_sold'] = date[0].astype('int64')
    df['year_sold'] = date[2].astype('int64')
    
    
    # Replace appropriate values in 'view' , 'waterfront', 'grade' and 'condition' columns
    df['view'] = df['view'].replace({'NONE': 0, 'AVERAGE': 3, 'GOOD': 4, 'FAIR': 2, 'EXCELLENT': 5})
    df['waterfront'] = df['waterfront'].replace({'NO': 0, 'YES': 1})
    df['grade'] = df['grade'].replace({3: 'Poor', 4: 'Low', 5: 'Fair', 6: 'Low Average', 7: 'Average', 8: 'Good', 9: 'Better', 10: 'Very Good', 11: 'Excellent', 12: 'Luxury', 13: 'Mansion'})
    df['condition'] = df['view'].replace({'Poor': 1, 'Fair': 2, 'Average': 3, 'Good': 4, 'Very Good': 5})
    
    # Convert 'view', 'waterfront', 'grade' and 'condition' columns to apppropriate datatype
    df['view'] = df['view'].fillna(0).astype('category')
    df['waterfront'] = df['waterfront'].astype('category')
    df['grade'] = df['view'].fillna(0).astype('category')
    df['condition'] = df['condition'].astype('category')
    df['zipcode'] = df['zipcode'].astype('category')
    

    
    # Drop original date column
    df.drop(columns=['date'], axis=1, inplace=True)
    
    # Convert year_built to age
    df['age'] = 2015 - df.yr_built
    df = df.drop(columns=['yr_built'], axis=1)
    
    # Fill missing values
    df.yr_renovated.fillna(0.0, inplace=True)
    
    # Create renovated column
    df['renovated'] = df.year_sold - df.yr_renovated
    renovated = df.renovated.values
    age = df.age.values
    values = np.where(renovated <= 10, 1, 0)
    df['renovated'] = np.where(age <= 5, 1, values)
    
    # Drop yr_renovated column
    #df.drop(columns=['yr_renovated'], axis=1, inplace=True)
    
    return df.copy()


# Define function to remove outliers. i.e entries with z-score above 3 for specific columns
def remove_outliers(df):
    variables = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above']
    for variable in variables:
        df = df[np.abs(df[variable]-df[variable].mean()) <= (3*df[variable].std())]
        
    return df       

# Define one-hot encoding function
def one_hot_encode(df, catcols):
    '''Returns df with dummy vars and drops original column'''
    
    # Create DataFrame with above columns
    dfonehot = df[catcols].astype('category')
    
    # Get dummy variables and drop first one to not create dependency
    dfonehot = pd.get_dummies(dfonehot, drop_first = True)
    
    # Recreate DataFrame with one-hot encoded variables
    df = pd.concat([df,dfonehot], axis=1)
    
    # Drop columns where we have done one-hot encoding
    df = df.drop(catcols, axis = 1)
        
    return df

# Define function to switch from lat/long to mercator coordinates
def x_coord(x, y):
    
    lat = x
    lon = y
    
    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x/lon
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + 
        lat * (np.pi/180.0)/2.0)) * scale
    return (x, y)

## Baseline/Simple linear regression helper function

def fit_simple_linear_reg(df, target_col, predictor_col):
    '''
    Fits a simple linear regression model and prints a summary of the results using statsmodels.

    Parameters:
    df (pandas DataFrame): the input data
    target_col (str): the name of the target column
    predictor_col (str): the name of the predictor column

    Returns:
    results (OLSRegressionResults): the results of the fitted model
    '''
    y = df[target_col]
    X = df[predictor_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.summary())
    return results

## Helper Function # Need to look into

def fit_multiple_linear_reg(df, target_col, predictor_cols):
    '''
    Fits a multiple linear regression model using scikit-learn's LinearRegression model.

    Parameters:
    df (pandas DataFrame): the input data
    target_col (str): the name of the target column
    predictor_cols (list of str): the names of the predictor columns

    Returns:
    model (LinearRegression): the fitted model
    train_r2 (float): the R-squared value on the training set
    test_r2 (float): the R-squared value on the test set
    '''
    X = df[predictor_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    print('Training R^2:', train_r2)
    print('Test R^2:', test_r2)
    return model, train_r2, test_r2

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def fit_polynomial_reg(df, target_col, predictor_cols, degree):
    '''
    Fits a polynomial regression model using scikit-learn's PolynomialFeatures and LinearRegression models.

    Parameters:
    df (pandas DataFrame): the input data
    target_col (str): the name of the target column
    predictor_cols (list of str): the names of the predictor columns
    degree (int): the degree of the polynomial regression

    Returns:
    model (LinearRegression): the fitted model
    train_r2 (float): the R-squared value on the training set
    test_r2 (float): the R-squared value on the test set
    '''
    X = df[predictor_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    poly = PolynomialFeatures(degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    train_r2 = r2_score(y_train, model.predict(X_poly_train))
    test_r2 = r2_score(y_test, model.predict(X_poly_test))
    print('Training R^2:', train_r2)
    print('Test R^2:', test_r2)
    return model, train_r2, test_r2


# Helper Function for calculating rmse
def calculate_rmse(y_true, y_pred):
    """Calculate the root mean squared error (RMSE) for a regression model."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


def plot_partial_residuals_all(results, predictor_cols):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), sharey=True)
    axs = axs.flatten()

    for i, col in enumerate(predictor_cols):
        plot_partial_residuals(results, focus_exog=col, ax=axs[i])
        axs[i].set_xlabel('Predictor')
        axs[i].set_ylabel('Partial Residual')
        axs[i].set_title(col)

    plt.tight_layout()
    plt.show()