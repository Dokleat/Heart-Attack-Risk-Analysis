#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('heart_attack_prediction_dataset.csv')

data.head()


# In[44]:


# Preparing the data
X = data.drop(['Heart Attack Risk', 'Patient ID'], axis=1)  # Assuming these columns are to be excluded
y = data['Heart Attack Risk']
X_encoded = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Training the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Getting feature importances
feature_importances = rf_model.feature_importances_

# Creating a DataFrame to store features and their importance
features_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sorting features by importance in descending order
features_df = features_df.sort_values(by='Importance', ascending=False)

# Visualizing the top 10 most important features
top_features_df = features_df.head(10)
plt.figure(figsize=(10, 6))
plt.barh(top_features_df['Feature'][::-1], top_features_df['Importance'][::-1], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Most Important Features for Predicting Heart Attack Risk')
plt.show()

# Displaying the top 10 features
print(top_features_df)


# In[45]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assuming 'data' is your DataFrame and you've already preprocessed it

# Selecting a subset of predictors related to lifestyle for the demonstration
predictors = ['Age', 'BMI', 'Physical Activity Days Per Week', 'Sedentary Hours Per Day',
              'Exercise Hours Per Week', 'Triglycerides']

# Checking and selecting the predictors present in the encoded dataset
X_lifestyle = X_encoded[[col for col in predictors if col in X_encoded.columns]]

# Scaling the features - important for logistic regression to ensure comparable impact
scaler = StandardScaler()
X_lifestyle_scaled = scaler.fit_transform(X_lifestyle)

# Splitting the scaled data into training and testing sets
X_train_ls, X_test_ls, y_train_ls, y_test_ls = train_test_split(X_lifestyle_scaled, y, test_size=0.2, random_state=42)

# Training the logistic regression model with class weights balanced to address class imbalance
log_reg_lifestyle_balanced = LogisticRegression(random_state=42, class_weight='balanced')
log_reg_lifestyle_balanced.fit(X_train_ls, y_train_ls)

# Making predictions on the test set
y_pred_ls_balanced = log_reg_lifestyle_balanced.predict(X_test_ls)

# Evaluating the model using precision, recall, and F1-score
print(classification_report(y_test_ls, y_pred_ls_balanced))

# Extracting and displaying the model's coefficients and odds ratios for interpretation
coefficients_balanced = pd.DataFrame(log_reg_lifestyle_balanced.coef_[0], index=predictors, columns=['Coefficient'])
coefficients_balanced['Odds Ratio'] = np.exp(coefficients_balanced['Coefficient'])
print(coefficients_balanced)


# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'data' is your DataFrame with a 'Country' and 'Heart Attack Risk' columns
# Grouping data by 'Country' and calculating the mean heart attack risk
country_risk = data.groupby('Country')['Heart Attack Risk'].mean().reset_index()

# Sorting countries by heart attack risk for visualization
sorted_risk = country_risk.sort_values(by='Heart Attack Risk', ascending=False)

# Visualizing the top 10 countries with the highest average heart attack risk
plt.figure(figsize=(10, 6))
sns.barplot(x='Heart Attack Risk', y='Country', data=sorted_risk.head(10), palette='Reds_r')
plt.title('Top 10 Countries with Highest Average Heart Attack Risk')
plt.xlabel('Average Heart Attack Risk')
plt.ylabel('Country')
plt.show()

# Optionally, analyze correlations with other factors
# For example, calculating the average BMI by country and comparing it with heart attack risk
country_bmi = data.groupby('Country')['BMI'].mean().reset_index()
# Merging the risk and BMI dataframes for correlation analysis
risk_bmi_correlation = pd.merge(country_risk, country_bmi, on='Country')
# Calculating correlation
correlation_matrix = risk_bmi_correlation[['Heart Attack Risk', 'BMI']].corr()
print(correlation_matrix)


# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Preparing the independent (X) and dependent (y) variables
X = data.drop(['Heart Attack Risk', 'Patient ID', 'Country', 'Continent', 'Hemisphere'], axis=1)
# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

y = data['Heart Attack Risk']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the data to improve model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Making predictions and evaluating the model
y_pred = log_reg.predict(X_test_scaled)

# Displaying the classification report
classification_rep = classification_report(y_test, y_pred)

print(classification_rep)



# In[48]:


bins = [0, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '>90']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)


age_group_risk = data.groupby('Age_Group')['Heart Attack Risk'].mean().reset_index()


continent_risk = data.groupby('Continent')['Heart Attack Risk'].mean().reset_index()

age_group_risk, continent_risk


# In[ ]:




