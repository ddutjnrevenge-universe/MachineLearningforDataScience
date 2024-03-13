#!/usr/bin/env python
# coding: utf-8

# # Assignment 1: Naive Bayes classification {-}
# 
# This assignment aims at familiarizing you with training and testing Naive Bayes model. You will have to:
# 
# - Load the dataset.
# - Analyze the dataset.
# - Split the dataset into training, validation and test set.
# - Train a Gaussian Naive Bayes (GaussianNB, https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) model and find the best set of hyperparameters (var_smoothing hyperparameter) using validation set and cross validation technique (see GridSearchCV )
# - Train a Mixed Naive Bayes (MixedNB, https://pypi.org/project/mixed-naive-bayes/) model.
# - Evaluate and compare the model performance of GaussianNB and MixedNB on the test set using the following metrics: precision, recall, f1-score.
# 
# The dataset you will be working on is 'travel-insurance.csv'. It is composed of attributes such as age, employment type, etc, to predict if a customer is going to buy a travel insurance.
# 
# ### Submission {-}
# The structure of submission folder should be organized as follows:
# 
# - ./\<StudentID>-assignment1-notebook.ipynb: Jupyter notebook containing source code.
# 
# The submission folder is named ML4DS-\<StudentID>-Assignment1 (e.g., ML4DS-2012345-Assigment1) and then compressed with the same name.
#     
# ### Evaluation {-}
# Assignment evaluation will be conducted on how properly you handle the data for training and testing purpose, build a Naive Bayes classifier and evaluate the model performance. In addition, your code should conform to a Python coding convention such as PEP-8.
# 
# ### Deadline {-}
# Please visit Canvas for details.

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# # Load & Analyze the Dataset

# In[42]:


data = pd.read_csv("travel-insurance.csv")
data.head()


# Here follows the list of columns in the dataset:
# 
# * Age - Age of the customer
# * Employment Type - The sector in which customer is employed
# * GraduateOrNot - Whether the customer is college graduate or not
# * AnnualIncome - The yearly income of the customer in indian rupees
# * FamilyMembers - Number of members in customer's family
# * ChronicDisease - Whether the customer suffers from any major disease or conditions like diabetes/high BP or asthama, etc.
# * FrequentFlyer - Derived data based on customer's history of booking air tickets on atleast 4 different instances in the last 2 Years (2017-2019).
# * EverTravelledAbroad - Has the customer ever travelled to a foreign country.
# * TravelInsurance: (label) Did the customer buy travel insurance package during introductory offering held in the year 2019.

# In[43]:


# Display basic information about the dataset
print("Dataset Information:")
print(data.info())


# In[44]:


# Summary statistics for numerical columns
print("\nSummary Statistics:")
print(data.describe())


# In[45]:


# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())


# **There is no missing value in this dataset!**

# In[46]:


# Drop unnamed column as it does not contain useful information for building the model
data = data.drop(columns=["Unnamed: 0"], axis=1)
data


# In[47]:


data.columns


# In[48]:


# Select categorical variables columns
cat_cols = [col for col in data.columns if data[col].dtypes == "O"]

# Encode categorical variables
data = pd.get_dummies(data, columns=cat_cols)

# Show the dataframe
data


# In[49]:


data.info()


# In[50]:


data.describe()


# In[51]:


#Variables' distribution
data.hist(figsize=(30,30))
plt.show()


# In[52]:


# Correlation heatmap between variables
plt.figure(figsize = (15,15))
sns.heatmap(data.corr(), cmap="viridis", annot=True)
plt.show()


# In[53]:


# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Boxplot for 'Age'
axes[0].boxplot(data['Age'], vert=False)
axes[0].set_title("Distribution of Age")

# Boxplot for 'AnnualIncome'
axes[1].boxplot(data['AnnualIncome'], vert=False)
axes[1].set_title("Distribution of Annual Income")

# Boxplot for 'FamilyMembers'
axes[2].boxplot(data['FamilyMembers'], vert=False)
axes[2].set_title("Distribution of Family Members")

# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.show()


# **There are no outliers in the distribution of those numerical features!**

# # Split the Dataset into Training, Validation, and Test Sets

# In[54]:


df = data.copy()


# In[55]:


from sklearn.model_selection import train_test_split

# Define features and target variable
X = df.drop("TravelInsurance", axis=1)
y = df["TravelInsurance"]


# In[56]:


X.shape


# In[57]:


y.shape


# In[63]:


# Split the data into train/test set using sklearn library
from sklearn.model_selection import train_test_split
# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# In[66]:


# Standardize the data using Standard scaler
from sklearn.preprocessing import StandardScaler
# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# # Train Gaussian Naive Bayes Model

# In[107]:


# Initialize and train Gaussian Naive Bayes model using X_normal_train (data features) and y_train (data label)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)


# In[108]:


# Impport libraries to calculate evaluation metrics: precision, recall, f1 score.
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# Make predictions on the validation set
predicted_gnb = gnb.predict(X_test_scaled)

# Calculate evaluation metrics by comparing the prediction with the data label y_test
print(precision_score(predicted_gnb, y_test))
print(recall_score(predicted_gnb, y_test))
print(f1_score(predicted_gnb, y_test))
print(classification_report(predicted_gnb, y_test))


# # Cross-validation

# In[109]:


# Define the values of hyperparameter var_smoothing we want to try
grid_search = {'var_smoothing': np.logspace(0, -9, num=10)}

# Set up GridSearchCV to find the best value of hyperparameter var_smoothing, with 10-fold cross validation
from sklearn.model_selection import GridSearchCV
naive_cv=GridSearchCV(gnb, grid_search, cv=10, scoring = 'accuracy')


# In[110]:


# Show the results of each hyperparameter var_smoothing with 10-fold cross validation
naive_scaled_plot = naive_cv.fit(X_train_scaled, y_train)
result_scaled = pd.DataFrame(naive_scaled_plot.cv_results_)
result_scaled = result_scaled.set_index('params')
result_scaled


# In[111]:


# Ranking the score of each hyperparameter var_smoothing to choose the best one
result_scaled[['mean_test_score', 'rank_test_score']].sort_values(by=['rank_test_score'])


# In[114]:


print('Best Hyperparameter:',naive_scaled_plot.best_params_)


# In[115]:


naive_scaled_plot.best_score_


# In[116]:


naive_scaled = GaussianNB(var_smoothing=naive_scaled_plot.best_params_['var_smoothing']) 

# Train the model
naive_scaled.fit(X_train_scaled, y_train)  

# Test accuracy of the 'best' hyperparameter var_smoothing = 0.002
naive_scaled.score(X_test_scaled, y_test)  


# # Mixed Naive Bayes

# In[114]:


# Install mixed-naive-bayes library
get_ipython().system('pip install git+https://github.com/remykarem/mixed-naive-bayes#egg=mixed_naive_bayes')


# In[117]:


from mixed_naive_bayes import MixedNB

# Initialize and train Mixed Naive Bayes model
mixed_nb = MixedNB(categorical_features=[3,4,5,6,7,8,9,10,11])
mixed_nb.fit(X_train,y_train)

# Make predictions on the validation set
pred_mixed = mixed_nb.predict(X_test)

# Evaluate performance on validation set
print("Precision (Mixed NB):", precision_score(pred_mixed,y_test))
print("Recall (Mixed NB):", recall_score(pred_mixed,y_test))
print("F1 Score (Mixed NB):", f1_score(pred_mixed,y_test))
print("Classification Report (Mixed NB):\n", classification_report(pred_mixed,y_test))


# # Evaluate and compare the model performance

# In[118]:


# GNB
print('\033[1m'+"Gaussian Naive Bayes Metrics:")
print('\033[0m'+"Precision:", precision_score(predicted_gnb, y_test))
print("Recall:", recall_score(predicted_gnb, y_test))
print("F1 Score:", f1_score(predicted_gnb, y_test))
print("Classification Report:\n", classification_report(predicted_gnb, y_test))

# Mxed Naive Bayes
print('\033[1m'+"\nMixed Naive Bayes Metrics:")
print('\033[0m'+"Precision (Mixed NB):", precision_score(pred_mixed,y_test))
print("Recall (Mixed NB):", recall_score(pred_mixed,y_test))
print("F1 Score (Mixed NB):", f1_score(pred_mixed,y_test))
print("Classification Report (Mixed NB):\n", classification_report(pred_mixed,y_test))


# **Precision:**
# 
# GaussianNB has a slightly higher precision (0.54) than Mixed Naive Bayes (0.53) for class 1 (positive class), indicating that it has a slightly better ability to correctly identify positive cases (those who purchased travel insurance) among all predicted positive cases. However, the difference is not significant.
# 
# **Recall:**
# 
# MixedNB has a higher recall (0.75) compared to GaussianNB (0.70) for class 1, implying that it can better capture the actual positive cases in the dataset. This means MNB is better at identifying individuals who purchased travel insurance among all actual positive cases.
# 
# **F1 Score:**
# 
# The F1 score of MixedNB (0.62) is slightly higher than that of GaussianNB (0.61), suggesting that MixedNB achieves a better balance between precision and recall compared to GaussianNB.
# 
# **Accuracy:**
# 
# MixedNB has a slightly higher accuracy (0.77) compared to GaussianNB (0.75), which indicates the overall correctness of the predictions of MixedNB is slightly better than that of GaussianNB.
# 
# **Conclusion:**
# 
# Both models perform reasonably well, with a balanced trade-off between precision and recall.
# MixedNB performs slightly better in terms of recall and accuracy, while GaussianNB has a higher precision.
