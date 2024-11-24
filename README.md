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

Some EDA samples from time series analysis:
![image](https://github.com/user-attachments/assets/4f77ae0b-1274-4a55-aa54-cc9b150c4601)
![image](https://github.com/user-attachments/assets/3f50f5ba-97b7-40d4-9e2e-db771bddce30)

Exploring relationships between features and customer subscriptions:
![image](https://github.com/user-attachments/assets/96e9ac0f-b2e9-4106-be40-15680cc2c1e5)
![image](https://github.com/user-attachments/assets/992e4380-4b14-4a0b-bfc7-7f1af6bb9356)
![image](https://github.com/user-attachments/assets/dbc60ce7-51ac-4093-bb44-7db208dce231)

Final Conclusions:
The Random Forest model was able to achieve a ROC_AUC metric of 0.8195-0.8200 with a few different manual adjustments of the hyperparameters. The settings attempted based on the training set did not yield optimal results, suggesting overfitting of the model.

Overall, the model was able to perform well given the small test set size of only 1798 samples. With SMOTE adjusting for class weight imbalance of the target variable, we were able to achieve satisfactory results in determining the likelihood of churn rate behavior among the customer database. Perhaps the model could be improved upon with a larger dataset over time. This metric was appropriate in determining the model's ability to predict customer behavior in how well it can distinguish between the positive and negative classes for true positive and false positive predictions of the churn rate.
