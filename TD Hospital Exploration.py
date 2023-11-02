#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('display.max_columns', None)


# In[2]:


#reading the data
df=pd.read_csv("TDHospital/TD_HOSPITAL_TRAIN.csv")


# In[3]:


df.head()


# In[4]:


print(df.isnull().sum())


# In[5]:


#droping cols which contains large number of missing values 
drop_cols=['cost','bloodchem1','glucose','psych2','bloodchem3','bloodchem4','sleep','bloodchem5','pdeath','psych3','disability','urine','income','bloodchem6','education']

df.drop(columns=drop_cols, inplace=True)


# In[6]:


df.head()


# In[7]:


#replace sex values 
df['sex'] = df['sex'].replace({'male': 'Male','M': 'Male', '1': 'Male', 'female': 'Female'})
df['sex'] = df['sex'].replace({'Male': '1','Female': '0'})


# In[8]:


print(df.isnull().sum())


# In[9]:


##filling null values - blood
def mean_val(column_name):
    mean_value = df[column_name].mean()
    df[column_name].fillna(mean_value, inplace=True)
    return 
    
#     'blood','information','bloodchem2','confidence','totalcost','administratorcost','psych5'


# In[10]:


mean_cols = ['blood','information','bloodchem2','confidence','totalcost','administratorcost','psych5','psych4']

for i in mean_cols:
    mean_val(i)
    


# In[11]:


def mode_val(column_name):
    mode_race = df[column_name].mode()[0]
    df[column_name].fillna(mode_race, inplace=True)
    return 
    
# 'race','dnr'


# In[12]:


mode_cols=['race','dnr','psych4']
for i in mode_cols:
    mode_val(i)


# 

# In[13]:


df


# In[14]:


import pandas as pd

# Define a function to find outliers using IQR
def find_outliers_iqr(data_series):
    q1 = data_series.quantile(0.25)
    q3 = data_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data_series[(data_series < lower_bound) | (data_series > upper_bound)]
    return outliers

# Loop through columns and print the count of outliers
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):  # Check if the column contains numeric data
        outliers = find_outliers_iqr(df[column])
        if not outliers.empty:
            print(f"Column: {column}, Number of Outliers: {outliers.shape[0]}")


# In[15]:


def reduce_outliers(values):
    v=[]
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    for val in values:
        if val>upper_bound:
            v.append(upper_bound)
        elif val<lower_bound:
            v.append(lower_bound)
        else:
            v.append(val)
    return v


# In[16]:


columns=['timeknown','blood','bloodchem2','confidence','totalcost','administratorcost','information','age','psych4','reflex','breathing','meals','temperature']
for col in columns:
    v=reduce_outliers(df[col])
    df[col]=v


# In[17]:


import pandas as pd

# Define a function to find outliers using IQR
def find_outliers_iqr(data_series):
    q1 = data_series.quantile(0.25)
    q3 = data_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data_series[(data_series < lower_bound) | (data_series > upper_bound)]
    return outliers

# Loop through columns and print the count of outliers
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):  # Check if the column contains numeric data
        outliers = find_outliers_iqr(df[column])
        if not outliers.empty:
            print(f"Column: {column}, Number of Outliers: {outliers.shape[0]}")


# In[18]:


#Drop duplicates 
df.drop_duplicates(inplace=True)


# In[19]:


df.duplicated().sum()


# In[20]:


df.duplicated().sum()


# In[21]:


df.head()


# In[22]:


drop_cols=['primary','dnr','extraprimary']

df.drop(columns=drop_cols, inplace=True)


# In[23]:


import pandas as pd
from itertools import combinations
from scipy.stats import chi2_contingency

def chi_square_association_test(df):

    categorical_columns = df.select_dtypes(include=['object']).columns  # Select categorical columns

    for col1, col2 in combinations(categorical_columns, 2):
        contingency_table = pd.crosstab(df[col1], df[col2])
        chi2, p, _, _ = chi2_contingency(contingency_table)

        print(f"Chi-Square Test for {col1} and {col2}:")
        print(f"Chi-Square Value: {chi2}")
        print(f"P-Value: {p}")

        alpha = 0.05  
        if p < alpha:
            print("There is a significant association between the columns.")
        else:
            print("There is no significant association between the columns.")
        print("\n")


chi_square_association_test(df)


# In[24]:


df.duplicated().sum()


# In[25]:


df['race'] = df['race'].replace({'white': '1','black': '0','hispanic':'2','other':'4','asian':'3'})
df['cancer'] = df['cancer'].replace({'yes': '1','no': '0','metastatic':'2'})


# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[27]:


train_data=df.drop(['death'],axis=1).values
train_label=df['death']


# In[28]:


X = train_data
y = train_label

# Split the data into a training and testing set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)


# In[ ]:





# In[ ]:




