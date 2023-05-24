import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# function to read data
def read_data():
    return pd.read_excel('C:/Users/rianm/Documents/data science/Default_predictor/data/default of credit card clients.xls',index_col=0)



def clean_data(data):
    # Selecting correct column names
    column_names = data.iloc[0]
    # Replacing column names to the dataset
    data = data.iloc[1:]
    data.columns = column_names
    # replacing formatting to have only the first letter capitalized
    data = data.rename(columns=str.capitalize)
    data.rename(columns={'Default payment next month': 'Target'}, inplace=True)
    data.drop_duplicates(inplace=True)
    
    # This function converts the columns to appropriate data type based on their contents
    def convert_columns(data):
        data = data.infer_objects()
        return data
    
    # applying the function to our data
    data = convert_columns(data)
    
    # Converting columns to appropriate data type
    # Sex column. 
    data['Sex'] = data.Sex.astype('category')
    # Education column
    data['Education'] = data.Education.astype('category')
    # Marriage column
    data['Marriage'] = data.Marriage.astype('category')
    # Define function to map gender
    def map_gender(gender):
        return 'Male' if gender == 1 else 'Female'
    
    # Applying the function
    data['Sex'] = data['Sex'].apply(map_gender)
    
    # Define a function to map the education level to their corresponding value
    def education_level(level):
        if level == 1:
            return "Graduate School"
        elif level == 2:
            return "University"
        elif level == 3:
            return "High School"
        else:
            return "Other"
    # Applying the function to Education column
    data['Education'] = data['Education'].apply(education_level)
    
    # Define a function to map marital status to corresponding value
    def marital_status(status):
        if status == 1:
            return "Married"
        elif status == 2:
            return "Single"
        else:
            return "Other"
    # Applying the function
    data['Marriage'] = data['Marriage'].apply(marital_status)
    
    # Define a function to map the repayment status values to their corresponding groups
    def map_repayment_status(status):
        if status in [-1, 0]:
            return "Performing"
        elif status in [1, 2, 3]:
            return "Watch"
        elif status in [4, 5, 6]:
            return "Substandard"
        elif status in [7, 8, 9]:
            return "Debt Collection"
        else:
            return "Defaulter"
        
        # Applying the function to relevant columns
    relevant_columns = ['Pay_0', 'Pay_2', 'Pay_3', 'Pay_4', 'Pay_5', 'Pay_6']
    for column in relevant_columns:
        data[column] = data[column].map(map_repayment_status)
    
    # define function to rename columns
    def rename_columns(data, column_mapping):
        return data.rename(columns=column_mapping)
    # define columns to rename
    column_mapping = {
    'Pay_0': 'Pay_status_Apr',
    'Pay_2': 'Pay_status_May',
    'Pay_3': 'Pay_Status_Jun',
    'Pay_4': 'Pay_Status_Jul',
    'Pay_5': 'Pay_Status_Aug',
    'Pay_6': 'Pay_Status_Sept',
    'Bill_amt1': 'Bill_amt_Apr',
    'Bill_amt2': 'Bill_amt_May',
    'Bill_amt3': 'Bill_amt_Jun',
    'Bill_amt4': 'Bill_amt_Jul',
    'Bill_amt5': 'Bill_amt_Aug',
    'Bill_amt6': 'Bill_amt_Sept',
    'Pay_amt1' : 'Paid_amt_Apr',
    'Pay_amt2' : 'Paid_amt_May',
    'Pay_amt3' : 'Paid_amt_Jun',
    'Pay_amt4' : 'Paid_amt_Jul',
    'Pay_amt5' : 'Paid_amt_Aug',
    'Pay_amt6' : 'Paid_amt_Sept'
    }
    #Applying the function
    data = rename_columns(data, column_mapping)
    
    return data.copy()


class ClassificationEvaluator:
    def __init__(self,y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        
    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    def precision(self):
        return precision_score(self.y_true, self.y_pred)
    
    def recall(self):
        return recall_score(self.y_true, self.y_pred)
    
    def f1_score(self):
        return f1_score(self.y_true, self.y_pred)
    
    def roc_auc(self):
        return roc_auc_score(self.y_true, self.y_pred)
    
    def evaluate(self):
        metrics = {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(),
            'Recall': self.recall(),
            'F1-Score': self.f1_score(),
            'ROC AUC': self.roc_auc(),
        }
        return metrics
    

