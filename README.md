## Forecasting Churn of Telecom Customers with Machine Learning

This project uses machine learning to determine which customers of telecom operator, Interconnect are most likely to end their service with the company in the near future based on historical data of previous customer churn. The target feature for the supervised learning models is the churn of customer contracts. 

The code for the preprocessing, EDA and model training/evaluation can be found in the Jupyter Notebook file.

The data is taken from: 
- contract.csv
- internet.csv
- personal.csv
- phone.csv

The requirements.txt file displays the necessary library and package versions needed for pushing the code to github. 

The following libraries and packages are used in this project:
- pandas as pd
- numpy as np
- seaborn as sns
- matplotlib.pyplot as plt
- sklearn.model_selection import train_test_split, GridSearchCV
- sklearn.preprocessing import StandardScaler
- sklearn.utils import resample
- imblearn.over_sampling import SMOTE
- sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
- sklearn.dummy import DummyClassifier
- sklearn.linear_model import LogisticRegression
- lightgbm as lgb
- sklearn.tree import DecisionTreeClassifier
- sklearn.ensemble import RandomForestClassifier

This jupyter notebook works locally and on hosted notebook servers. Simply open the "Forecasting Churn of Customers - Telecom.ipynb" file. 

