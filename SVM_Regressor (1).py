#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Support Vector Regression model from scikit-learn
from sklearn.svm import SVR
# Import pandas for data manipulation and analysis
import pandas as pd
# Import train_test_split for splitting dataset into training and testing sets
from sklearn.model_selection import train_test_split
# Import StandardScaler for feature scaling/normalization
from sklearn.preprocessing import StandardScaler
# Import R-squared score metric for model evaluation
from sklearn.metrics import r2_score
# Import datasets module for accessing built-in datasets
from sklearn import datasets


# In[2]:


# Load the diabetes dataset from scikit-learn as a pandas DataFrame
df = datasets.load_diabetes(as_frame = True).frame


# In[3]:


# Display the first 5 rows of the DataFrame to examine the data structure and content
df.head()


# In[4]:


# Get the dimensions of the DataFrame (rows, columns)
df.shape


# In[5]:


# Separate features from target variable
X = df.drop("target", axis = 1)  # Features: all columns except 'target'
y = df["target"]  # Target variable: the 'target' column


# In[6]:


X.head()


# In[7]:


y.head()


# In[8]:


# Split the dataset into training and testing sets
# 70% for training, 30% for testing with fixed random state for reproducibility
X_train, X_test, y_train,y_test = train_test_split(
    X, y,test_size = 0.3,random_state = 42 
)


# In[9]:


# Initialize StandardScaler for target variable normalization
y_scaler = StandardScaler()
# Fit scaler on training data and transform y_train to scaled values
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1,1)).ravel()
# Transform y_test using the fitted scaler (no fitting on test data)
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()


# In[10]:


# Initialize StandardScaler to normalize target variables
scaler = StandardScaler()
# Fit scaler on training data and transform y_train to scaled values
# reshape(-1,1) converts 1D array to 2D column vector, ravel() flattens back to 1D
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1,1)).ravel()
# Transform y_test using the same scaler fitted on training data
y_test_scaled = scaler.transform(y_test.values.reshape(-1,1)).ravel()


# In[11]:


# Initialize Support Vector Regression model with default parameters
model = SVR()
# Train the SVR model on the scaled training data
model.fit(X_train,y_train_scaled)


# In[12]:


# Generate predictions on the test set using the trained model
y_pred_scaled = model.predict(X_test)
y_train_pred_scaled = model.predict(X_train)


# In[13]:


# Calculate and print the R-squared score to evaluate model performance
# Compares actual scaled test values (y_test_scaled) with predicted scaled values (y_pred_scaled)
print("r2: ",r2_score(y_train_scaled,y_train_pred_scaled))
print("r2: ",r2_score(y_test_scaled,y_pred_scaled))


# In[14]:


# Create a Support Vector Regression model with linear kernel
model = SVR(kernel = "linear")
# Train the model on scaled training data
model.fit(X_train,y_train_scaled)
# Generate predictions on test data (note: missing parentheses for function call)
y_pred_scaled = model.predict(X_test)
# Calculate and display R-squared score between actual and predicted values
print("r2: ",r2_score(y_test_scaled,y_pred_scaled))
print("r2: ",r2_score(y_train_scaled,y_train_pred_scaled))


# # Hyperparmater Tuning using GridSearchCV

# In[16]:


from sklearn.model_selection import GridSearchCV


# In[17]:


# Define hyperparameter grid for SVM model tuning
param_grid = {
       # Regularization parameter - controls trade-off between smooth decision boundary and classifying training points correctly
       "C":[1,2,5,10,50,100],
       # Kernel type - specifies the kernel function to be used in the algorithm
       "kernel":["rbf","linear"],
       # Epsilon parameter - defines margin of tolerance where no penalty is given to errors
       "epsilon":[0.01,0.1,0.2,0.3,0.5] 
}


# In[18]:


# Initialize Support Vector Regression model
svr = SVR()
# Set up GridSearchCV to find optimal hyperparameters using 5-fold cross-validation and R² scoring
grid_search = GridSearchCV(svr,param_grid,scoring = "r2", cv = 5)
# Train the model with hyperparameter tuning on scaled training data
grid_search.fit(X_train,y_train_scaled)


# In[19]:


# Display the optimal hyperparameters found by grid search
print("best params - ",grid_search.best_params_)


# In[20]:


best_model = SVR(kernel = "linear",C = 10, epsilon = 0.1)
best_model.fit(X_train,y_train_scaled)
y_test_pred_scaled = best_model.predict(X_test)
y_train_pred_scaled = best_model.predict(X_train)
print("r2: ",r2_score(y_test_scaled,y_pred_scaled))
print("r2: ",r2_score(y_train_scaled,y_train_pred_scaled))


# In[ ]:




