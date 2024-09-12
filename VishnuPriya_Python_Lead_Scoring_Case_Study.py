#!/usr/bin/env python
# coding: utf-8

# # <font color='Blue'>Lead Scoring - Case Study</font>
# ## <font color = 'cyan'>Problem Statement</font>
# An X Education need help to select the most promising leads, i.e. the leads that are most likely to convert into paying customers. 
# The company requires us to build a model wherein you need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance. 
# The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.
# 
# ## <font color = 'cyan'>Goals of Case Study</font>
# 
# Build a logistic regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. 
# A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted.

# ## <font color = 'sky blue'> Data Collection and EDA </font>

# ### <font color = 'Green'> Importing File and Inspection</font>

# In[4]:


#Importing required Libararies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#supress warnings
import warnings
warnings.filterwarnings("ignore")

# Sklearn libraries

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

#statmodel libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# In[5]:


#importing dataset to csv

lead_df=pd.read_csv("Leads.csv")


# In[6]:


lead_df.info()


# In[7]:


lead_df.head()


# In[8]:


lead_df.describe()


# In[9]:


lead_df.nunique()


# In[10]:


lead_df.shape


# In[11]:


lead_df.isnull().sum()/lead_df.shape[0]


# ### <font color = 'Green'> Data Cleaning </font>

# #### <font color = 'Megenta'>Identifying Missing Values</font>

# In[14]:


print(lead_df.Magazine.value_counts())
print(lead_df['Receive More Updates About Our Courses'].value_counts())
print(lead_df['Update me on Supply Chain Content'].value_counts())
print(lead_df['Get updates on DM Content'].value_counts())
print(lead_df['I agree to pay the amount through cheque'].value_counts())

# The columns have only one unique values ("All values are no') will not help us in the model building so better we drop them.
1.Magazine 2.Receive More Updates About Our Courses  3.Update me on Supply Chain Content  4.Get updates on DM Content  5.I agree to pay the amount through cheque
# In[15]:


lead_df.drop(['Magazine','Receive More Updates About Our Courses','Update me on Supply Chain Content',
              'Get updates on DM Content','I agree to pay the amount through cheque'],axis = 1, inplace = True)


# In[16]:


# The value Select in the columns are actaullay nulls since its the default data when data is not avaiable
#Replacing 'Select' values with Nan
lead_df=lead_df.replace("Select", np.nan)


# In[17]:


#Identify Dulicate Rows if any
lead_df.duplicated().sum()


# In[18]:


round((lead_df.isnull().sum()/lead_df.shape[0])*100,2)


# #### <font color = 'Megenta'>Dropping Columns with Missing Values >=35% and analysing other null values </font>

# In[20]:


#Drop all the columns with more than 35% missing values
cols=lead_df.columns

for i in cols:
    if((100*(lead_df[i].isnull().sum()/lead_df.shape[0])) >= 35):
        lead_df.drop(i, axis =1, inplace = True)


# In[21]:


round((lead_df.isnull().sum()/lead_df.shape[0])*100,4)


# In[22]:


# Columns Prospect_ id and Lead Number are unique Numbers for each applicant and thus not needed for Modelling.
print(lead_df['Prospect ID'].nunique())
print(lead_df['Lead Number'].nunique())


# In[23]:


lead_df.drop(['Prospect ID','Lead Number'],axis = 1, inplace = True)


# In[24]:


print(lead_df.shape)
print(lead_df.info())
round((lead_df.isnull().sum()/lead_df.shape[0])*100,4)


# In[25]:


## Replacing Null values with Mode for Catogariacal Columns
lead_df['Lead Source'].value_counts()


# In[26]:


lead_df['Lead Source'].fillna(lead_df['Lead Source'].mode()[0], inplace = True)


# In[27]:


lead_df['Lead Source'] = lead_df['Lead Source'].replace('google','Google')


# In[28]:


#combining low frequency values to Others

lead_df['Lead Source'] = lead_df['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM','Live Chat','Social Media'] ,'Others')


# In[29]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x= 'Lead Source', hue='Converted' , data =lead_df)
s1.set_xticklabels(s1.get_xticklabels(),rotation=45)
plt.show()


# In[30]:


lead_df.Country.value_counts()


# In[31]:


lead_df['Country'].fillna(lead_df['Country'].mode()[0], inplace = True)

lead_df['Country'] = lead_df['Country'].replace(['Indonesia','Sri Lanka','Malaysia',
                                                     'Vietnam','Russia',
                                                     'Philippines','Bangladesh','Asia/Pacific Region',
                                                    'China','Kuwait','Oman','Bahrain','Hong Kong','Qatar','Saudi Arabia','Singapore','United Arab Emirates'] ,'Asia')
lead_df['Country'] = lead_df['Country'].replace(['Denmark','Switzerland','Netherlands',
                                                     'Belgium','Italy',
                                                     'Sweden','Canada','Germany',
                                                    'France','Australia','United Kingdom','United States','Kenya','Liberia','Tanzania',
                                                     'Ghana','Uganda',
                                                     'Nigeria','South Africa'] ,'Others')


# In[32]:


plt.figure(figsize=(15,5))
s1=sns.countplot(x= 'Country', hue='Converted' , data =lead_df)
s1.set_xticklabels(s1.get_xticklabels(),rotation=45)
plt.show()

X Education sells online courses and appx 96% of the customers are from India. Does not make business sense right now to impute missing values with India. Hence `Country column can be dropped
# In[33]:


lead_df.drop('Country',axis = 1, inplace = True)


# In[34]:


lead_df['What is your current occupation'] = lead_df['What is your current occupation'].replace(np.nan, 'Unknown')


# In[35]:


#visualizing count of Variable based on Converted value

s1=sns.countplot(x='What is your current occupation', hue='Converted' , data = lead_df)
s1.set_xticklabels(s1.get_xticklabels(),rotation=45)
plt.show()

Maximum leads generated are unemployed and their conversion rate is more than 50%.
Conversion rate of working professionals is very high.
# In[36]:


lead_df['What matters most to you in choosing a course'].value_counts()


# In[37]:


# Most Values have only one Option - 'Better Career Prospects '. Not useful for modelling, so can be dropped.
lead_df.drop('What matters most to you in choosing a course', axis= True, inplace = True)


# In[38]:


# the number of columns missing in TotalVisits and Pages per Visit is same. I.e Pages per visit is dependent on Total Visits.
#These Remaining missing values percentage is less than 2%, we can drop those rows without affecting the data
lead_df = lead_df.dropna()


# In[39]:


# since TotalVisits have Outliners, replacing nulls with Median
lead_df.TotalVisits.fillna(lead_df.TotalVisits.median(),inplace = True)


# In[40]:


# plotting countplot for object dtype and histogram for number to get data distribution
categorical_col = lead_df.select_dtypes(include=['category', 'object']).columns.tolist()
plt.figure(figsize=(12,40))

plt.subplots_adjust(wspace=.2,hspace=2)
for i in enumerate(categorical_col):
    plt.subplot(8,2, i[0]+1)
    ax=sns.countplot(x=i[1],data=lead_df,color = '#AAFF32') 
    plt.xticks(rotation=90)
    
    for p in ax.patches:
        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                      ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')


plt.show()


# In[41]:


# Grouping low frequency value levels to Others in Last Activity Column
lead_df['Last Activity'] = lead_df['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                               'Had a Phone Conversation', 
                                                               'Approached upfront',
                                                               'View in browser link Clicked',       
                                                               'Email Marked Spam',                  
                                                               'Email Received','Visited Booth in Tradeshow',
                                                               'Resubscribed to emails'],'Others')


# In[42]:


# Dropping columns that are Skewed
lead_df.drop(['Search',
 'Newspaper Article',
 'X Education Forums',
 'Newspaper',
 'Digital Advertisement',
 'Through Recommendations',
 'Do Not Call'], axis= 1, inplace = True)
              
              


# In[43]:


lead_df.info()


# #### <font color = 'Megenta'>Handling Outliners </font>
# 

# In[45]:


# Visualizing TotalVisits w.r.t Target Variable 'Converted'
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = lead_df)
plt.show()


# In[46]:


# Visualizing TotalVisits w.r.t Target Variable 'Converted'
sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = lead_df)
plt.show()


# In[47]:


# Visualizing Page Views Per Visit w.r.t Target Variable 'Converted'
sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data = lead_df)
plt.show()


# In[48]:


# Defining UDF to treat outliers via IQR

def Outlier_treat(df,NumCols):
    for i in NumCols:
        q1 = df[i].describe()["25%"]
        q3 = df[i].describe()["75%"]
        IQR = q3 - q1

        upper_bound = q3 + 1.5*IQR
        lower_bound = q1 - 1.5*IQR

        # capping upper_bound
        df[i] = np.where(df[i] > upper_bound, upper_bound,df[i])

        # flooring lower_bound
        df[i] = np.where(df[i] < lower_bound, lower_bound,df[i])


# In[49]:


ColList = ['Page Views Per Visit','TotalVisits']
Outlier_treat(lead_df,ColList)


# In[50]:


# Visualizing Page Views Per Visit w.r.t Target Variable 'Converted'
sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data = lead_df)
plt.show()
# Visualizing Page Views Per Visit w.r.t Target Variable 'Converted'
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = lead_df)
plt.show()


# In[51]:


#clubbing lower frequency values

lead_df['Last Notable Activity'] = lead_df['Last Notable Activity'].replace(['Had a Phone Conversation',
                                                                       'Email Marked Spam',
                                                                         'Unreachable',
                                                                         'Unsubscribed',
                                                                         'Email Bounced',                                                                    
                                                                       'Resubscribed to emails',
                                                                       'View in browser link Clicked',
                                                                       'Approached upfront', 
                                                                       'Form Submitted on Website', 
                                                                       'Email Received'],'Others')


# In[52]:


# Last Notable Activity resonates with the Last Activity and so can be dropped.
lead_df.drop('Last Notable Activity', axis = 1, inplace = True)


# ### <font color = 'Green'> Data Analysis EDA </font>

# In[54]:


## ploting the results on bar plot

ax=(100*lead_df["Converted"].value_counts(normalize=True)).plot.bar(color=["Green","Yellow"],alpha=0.4)

# Adding and formatting title
plt.title("Leads Converted\n", fontdict={'fontsize': 16, 'fontweight' : 12, 'color' : 'Green'})


# Labeling Axes
plt.xlabel('Converted', fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})
plt.ylabel("Percentage Count", fontdict={'fontsize': 12, 'fontweight' : 20, 'color' : 'Brown'})

# modification ticks y axis
ticks=np.arange(0,101,20)
labels=["{:.0f}%".format(i) for i in ticks] 
plt.yticks(ticks,labels)

#xticks
plt.xticks([0,1],["No","Yes"])
plt.xticks(rotation=0)

for p in ax.patches:
    ax.annotate('{:.1f}%'.format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), 
                  ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
    


# In[55]:


# 38% Lead Conversion date is found in the given data.
### Ratio of Data Imbalance
ratio=(lead_df["Converted"].value_counts(normalize=True).loc[0])/(lead_df["Converted"].value_counts(normalize=True).loc[1])

print("Data Imbalance Ratio : {:.2f} : {}".format(ratio,1))


# In[56]:


# plotting countplot for object dtype and histogram for number to get data distribution
categorical_col = lead_df.select_dtypes(include=['category', 'object']).columns.tolist()
plt.figure(figsize=(20,120))

plt.subplots_adjust(wspace=.2,hspace=2)
for i in enumerate(categorical_col):
    plt.subplot(15,2, i[0]+1)
    plt.title("Count plot of {}".format(i[1]),color="green")
    ax=sns.countplot(x=i[1],hue='Converted',data=lead_df,palette = 'pastel6') 
    plt.xticks(rotation=75)
    
    for p in ax.patches:
        text = '{:.1f}%'.format(100*p.get_height()/lead_df.shape[0])
        x = p.get_x() + p.get_width() / 2.
        y = p.get_height()
        ax.annotate(text, (x,y), 
                      ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')


plt.show()

Insights:

Lead Origin: 

Current_occupation: Around 70% of the customers are Unemployed with lead conversion rate (LCR) of 34%. While Working Professional contribute only 7.6(6.9+06)% of total customers with almost 92% lead conversion rate (LCR).

Do Not Email: 92% of the people has opted that they dont want to be emailed about the course.

Lead Source: Google has  Lead Conversion Rate  of 12.9%  , Direct Traffic contributes 9% Lead Conversion Rate which is lower than Google.

Last Activity: 'SMS Sent' has high lead conversion rate of 16.6% contribution from last activities, 'Email Opened' activity contributed 11.5% of last activities performed by the customers.

# In[57]:


plt.figure(figsize=(15, 5))
plt.subplot(1,3,1)
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = lead_df,palette = 'Set2')
plt.subplot(1,3,2)
sns.boxplot(y = 'Page Views Per Visit', x = 'Converted', data = lead_df,palette = 'Set2')
plt.subplot(1,3,3)
sns.boxplot(y = 'Total Time Spent on Website', x = 'Converted', data = lead_df,palette = 'Set2')
plt.show()


# Past Leads who spends more time on Website are successfully converted than those who spends less as seen in the boxplot

# In[59]:


#Checking correlations of numeric values using heatmap
num_cols =["Converted",'TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']
# Size of the figure
plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(data=lead_df[num_cols].corr(),cmap="RdPu",annot=True)
plt.show()


# ## <font color = 'sky blue'> Data Preparation </font>

# In[61]:


#Mapping Binary categorical variables
lead_df['Do Not Email'] = lead_df['Do Not Email'].apply(lambda x: 1 if x =='Yes' else 0)

lead_df['A free copy of Mastering The Interview'] = lead_df['A free copy of Mastering The Interview'].apply(lambda x: 1 if x =='Yes' else 0)


# In[62]:


lead_df.info()


# In[63]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy = pd.get_dummies(lead_df[["Lead Origin","Lead Source","Last Activity"]], drop_first=True)
lead_df = pd.concat([lead_df, dummy], axis = 1)


# In[64]:


dummy = pd.get_dummies(lead_df['What is your current occupation'], prefix  = 'Occupation')
dummy.head()
dummy = dummy.drop(['Occupation_Unknown'], axis = 1)
lead_df = pd.concat([lead_df, dummy], axis = 1)


# In[65]:


lead_df.drop(["Lead Origin","Lead Source","Last Activity","What is your current occupation"],axis =1,inplace = True)


# In[66]:


lead_df.head()


# In[67]:


lead_df.info()


# In[68]:


# Let's see the correlation matrix
plt.figure(figsize = (20,15))        # Size of the figure
sns.heatmap(lead_df.corr(),annot = True)
plt.show()


# In[69]:


lead_df.drop(['Do Not Email','Lead Origin_Lead Import','Page Views Per Visit'],axis= 1, inplace = True)
# Not dropping Lead Origin_Lead Add Form and Lead Source_Reference even when the Corr is 0.85 because in bivariant we found them to have high Lead Conversion Rate


# ## <font color = 'sky blue'> Test - Train Split </font>

# In[71]:


#importing library for splitting dataset
from sklearn.model_selection import train_test_split
# convert Bool data types to int
bool_columns = lead_df.select_dtypes(include='bool').columns
lead_df[bool_columns] = lead_df[bool_columns].astype('int')


# In[72]:


# Putting feature variable to X
X=lead_df.drop('Converted', axis=1)

#checking head of X
X.head()


# In[73]:


# Putting response variable to y
y = lead_df['Converted']

#checking head of y
y.head()


# In[74]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# ## <font color = 'sky blue'> Feature Scaling</font>

# In[76]:


#importing library for feature scaling
from sklearn.preprocessing import StandardScaler


# In[77]:


#scaling of features
scaler = StandardScaler()

num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

#checking X-train dataset after scaling
X_train.head()


# In[78]:


# Checking the Lead Conversion Rate (LCR) - "Converted" is our Target Variable
# We will denote Lead Conversion Rate with 'LCR' as its short form

LCR = (sum(lead_df['Converted'])/len(lead_df['Converted'].index))*100
LCR


# ## <font color = 'sky blue'> Model Building using Stats Model & RFE</font>

# In[80]:


# importing necessary library
import statsmodels.api as sm


# In[81]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, n_features_to_select= 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[82]:


rfe.support_


# In[83]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[84]:


#list of RFE supported columns
col = X_train.columns[rfe.support_]
col


# In[85]:


X_train.columns[~rfe.support_]


# In[86]:


# User defined function for calculating VIFs for variables
def get_vif(model_df):
    X = pd.DataFrame()
    X['Features'] = model_df.columns
    X['VIF'] = [variance_inflation_factor(model_df.values, i) for i in range(model_df.shape[1])]
    X['VIF'] = round(X['VIF'], 2)
    X = X.sort_values(by='VIF', ascending=False)
    X = X.reset_index(drop=True)
    return X


# ## <font color = 'sky blue'> Model fine tuning using GLM & VIF</font>

# ### <font color = 'Green'> Model 1</font>

# In[89]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[90]:


#NOTE : "Occupation_Housewife" column will be removed from model due to high p-value of 0.999, which is above the accepted threshold of 0.05 for statistical significance.


# ### <font color = 'Green'> Model 2</font>

# In[92]:


col


# In[93]:


# Dropping 'Current_occupation_Housewife' column
col = col.drop("Occupation_Housewife")


# In[94]:


# Creating X_train dataframe with variables selected by RFE
X_train_rfe = X_train[col]

# Adding a constant variable 
X_train_sm2 = sm.add_constant(X_train_rfe)

# Create a fitted model
logm2 = sm.GLM(y_train,X_train_sm2,family = sm.families.Binomial()).fit()  

logm2.params


# In[95]:


#Let's see the summary of our logistic regression model
print(logm2.summary())

Model 2 is stable and has significant p-values within the threshold (p-values < 0.05), so we will use it for further analysis.
Now lets check VIFs for these variables to check if there is any multicollinearity which exists among the independent variablesNOTE: No variable needs to be dropped as they all have good VIF values less than 5.

p-values for all variables is less than 0.05
This model looks acceptable as everything is under control (p-values & VIFs).
So we will final our Model 2 for Model Evaluation.
# In[96]:


get_vif(X_train_rfe)


# ## <font color = 'sky blue'> Model Evaluation</font>
Confusion Matrix
Accuracy
Sensitivity and Specificity
Threshold determination using ROC & Finding Optimal cutoff point
Precision and Recall
# ### <font color = 'Green'> Predict the Target Variable in Train set.</font>

# In[99]:


# Getting the predicted values on the train set
y_train_pred = logm2.predict(X_train_sm2)           # giving prob. of getting 1

y_train_pred[:10]


# In[100]:


# for array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[101]:


# Creating a dataframe with the actual churn flag and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final.head()

# y_train.values actual Converted values from df_leads dataset
# y_train_pred probability of Converted values predicted by model


# In[102]:


y_train_pred_final['Predicted'] = y_train_pred_final["Converted_Prob"].map(lambda x: 1 if x > 0.5 else 0)

# checking head
y_train_pred_final.head()


# ### <font color = 'Green'> Confusion Matrix.</font>

# In[104]:


# Confusion matrix  (Actual / predicted)

confusion = metrics.confusion_matrix(y_train_pred_final["Converted"], y_train_pred_final["Predicted"])
print(confusion)


# ### <font color = 'Green'> Accuracy</font>

# In[106]:


# Checking the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final["Converted"], y_train_pred_final["Predicted"]))


# In[107]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# ### <font color = 'Green'> Sensitivity and Specificity</font>

# In[109]:


# Let's see the sensitivity of our logistic regression model
print("Sensitivity :",TP / float(TP+FN))


# In[110]:


# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# In[111]:


# Calculate false postive rate - predicting conversion when customer does not have converted
print(FP/ float(TN+FP))


# In[112]:


# positive predictive value 
print (TP / float(TP+FP))


# In[113]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### <font color = 'Green'> ROC Curve</font>

# In[115]:


# UDF to draw ROC curve 
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[116]:


fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final["Converted"], y_train_pred_final["Converted_Prob"], drop_intermediate = False )


# In[117]:


# Drawing ROC curve for Train Set
draw_roc(y_train_pred_final["Converted"], y_train_pred_final["Converted_Prob"])

NOTE: Area under ROC curve is 0.88 out of 1 which indicates a good predictive model
# #### <font color = 'Megenta'>Finding Optimal Cutoff Point/ Probability</font>

# In[119]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final['Converted_Prob'].map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[120]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final["Converted"], y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[121]:


# Let's plot accuracy sensitivity and specificity for various probabilities.

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# Finding the intersection points of the sensitivity and accuracy curves
sen_interp = interp1d(cutoff_df['prob'], cutoff_df['sensi'], kind='linear')
acc_interp = interp1d(cutoff_df['prob'], cutoff_df['accuracy'], kind='linear')

intersection_1 = np.round(float(fsolve(lambda x : sen_interp(x) - acc_interp(x), 0.5)), 2)

# Find the intersection points of the specificity and accuracy curves
spec_interp = interp1d(cutoff_df['prob'], cutoff_df['speci'], kind='linear')
intersection_2 = np.round(float(fsolve(lambda x : spec_interp(x) - acc_interp(x), 0.5)), 2)

# Calculate the average of the two intersection points
intersection_x = (intersection_1 + intersection_2) / 2

# Interpolate the accuracy, sensitivity, and specificity at the intersection point
accuracy_at_intersection = np.round(float(acc_interp(intersection_x)), 3)
sensitivity_at_intersection = np.round(float(sen_interp(intersection_x)), 2)
specificity_at_intersection = np.round(float(spec_interp(intersection_x)), 2)

# Plot the three curves and add vertical and horizontal lines at intersection point
cutoff_df.plot.line(x='prob', y=['accuracy', 'sensi', 'speci'])
plt.axvline(x=intersection_x, color='grey',linewidth=0.55, linestyle= 'dashdot')
plt.axhline(y=accuracy_at_intersection, color='grey',linewidth=0.55, linestyle='dashdot')

# Adding annotation to display the (x,y) intersection point coordinates 
plt.annotate(f'({intersection_x} , {accuracy_at_intersection})',
             xy=(intersection_x, accuracy_at_intersection),
             xytext=(0,20),
             textcoords='offset points',
             ha='center',
             fontsize=9)

# Displaying the plot
plt.show()

NOTE: 0.36 is the approx. point where all the curves meet, so 0.36 seems to be our Optimal cutoff point for probability threshold .
We will Predict the result based on this Optimal Cutoff at 0.36 instead of 0.5
# In[122]:


y_train_pred_final['final_predicted'] = y_train_pred_final['Converted_Prob'].map( lambda x: 1 if x > 0.36 else 0)

# deleting the unwanted columns from dataframe
y_train_pred_final.drop([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,"Predicted"],axis = 1, inplace = True) 
y_train_pred_final.head()


# ### <font color = 'Green'> Calculating all metrics using confusion matrix for new Train split based on cutoff value 0.36</font> 

# In[124]:


# Checking the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final["Converted"], y_train_pred_final["final_predicted"]))

# or can be found using confusion matrix with formula, lets find all matrix in one go ahead using UDF


# In[125]:


# UDF for all Logistic Regression Metrics
def logreg_all_metrics(confusion_matrix):
    TN =confusion_matrix[0,0]
    TP =confusion_matrix[1,1]
    FP =confusion_matrix[0,1]
    FN =confusion_matrix[1,0]
    
    accuracy = (TN+TP)/(TN+TP+FN+FP)
    sensi = TP/(TP+FN)
    speci = TN/(TN+FP)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP)
    
    #Calculate false postive rate - predicting conversion when customer does not have converted
    FPR = FP/(FP + TN)     
    FNR = FN/(FN +TP)
    
    print ("True Negative                    : ", TN)
    print ("True Positive                    : ", TP)
    print ("False Negative                   : ", FN)
    print ("False Positve                    : ", FP) 
    
    print ("Model Accuracy                   : ", round(accuracy*100,2),'%')
    print ("Model Sensitivity                : ", round(sensi*100,2),'%')
    print ("Model Specificity                : ", round(speci*100,2),'%')
    print ("Model Precision                  : ", round(precision*100,2),'%')
    print ("Model Recall                     : ", round(recall*100,2),'%')
    print ("Model True Positive Rate (TPR)   : ", round(TPR,4))
    print ("Model False Positive Rate (FPR)  : ", round(FPR,4))
    
    


# In[126]:


# Finding Confusion metrics for 'y_train_pred_final' df
confusion_matrix = metrics.confusion_matrix(y_train_pred_final['Converted'], y_train_pred_final['final_predicted'])

#
print("Confusion Matrix")
print(confusion_matrix,"\n")

# Using UDF to calculate all metrices of logistic regression
logreg_all_metrics(confusion_matrix)

print("\n")


# ### <font color = 'Green'> Precision and recall tradeoff</font> 

# In[128]:


# Creating precision-recall tradeoff curve
y_train_pred_final['Converted'], y_train_pred_final['final_predicted']
p, r, thresholds = precision_recall_curve(y_train_pred_final['Converted'], y_train_pred_final['Converted_Prob'])


# In[129]:


# plot precision-recall tradeoff curve
plt.plot(thresholds, p[:-1], "g-", label="Precision")
plt.plot(thresholds, r[:-1], "r-", label="Recall")

# add legend and axis labels

plt.axvline(x=0.405, color='teal',linewidth = 0.55, linestyle='--')
plt.legend(loc='lower left')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')

# Displaying the plot
plt.show()

NOTE: The intersection point of the curve is the threshold value where the model achieves a balance between precision and recall. It can be used to optimise the performance of the model based on business requirement,Here our probability threshold is 0.405 aprrox from above curve.
# In[130]:


# copying df to test model evaluation with precision recall threshold of 0.405
y_train_precision_recall = y_train_pred_final.copy()


# In[131]:


# assigning a feature for 0.41 cutoff from precision recall curve to see which one is best view (sensi-speci or precision-recall)
y_train_precision_recall['precision_recall_prediction'] = y_train_precision_recall['Converted_Prob'].map( lambda x: 1 if x > 0.405 else 0)
y_train_precision_recall.head()


# In[132]:


## Lets see all matrics at 0.405 cutoff in precision-recall view and compare it with 0.345 cutoff from sensi-speci view

# Finding Confusion metrics for 'y_train_precision_recall' df
confusion_matrix = metrics.confusion_matrix(y_train_precision_recall['Converted'], y_train_precision_recall['precision_recall_prediction'])

#
print("Confusion Matrix")
print(confusion_matrix,"\n")
# Using UDF to calculate all metrices of logistic regression
logreg_all_metrics(confusion_matrix)

print("\n")

As we can see in above metrics when we used precision-recall threshold cut-off of 0.405 the values in True Positive Rate ,Sensitivity, Recall have dropped to around 74%, but we need it close to 80% as the Business Objective.
Close to 80% we were getting with the sensitivity-specificity cut-off threshold of 0.36. So, we will go with sensitivity-specificity view for our Optimal cut-off for final predictions.
# ### <font color = 'Green'> Adding Lead Score Feature to Training dataframe</font> 

# In[134]:


# Lets add Lead Score 

y_train_pred_final['Lead_Score'] = y_train_pred_final['Converted_Prob'].map( lambda x: round(x*100))
y_train_pred_final.head()


# ## <font color = 'sky blue'>  Making Predictions on test set</font>

# ### <font color = 'Green'> Scaling Test dataset</font> 

# In[137]:


# fetching int64 and float64 dtype columns from dataframe for scaling
num_cols=X_test.select_dtypes(include=['int64','float64']).columns

# scaling columns
X_test[num_cols] = scaler.transform(X_test[num_cols])

X_test = X_test[col]
X_test.head()


# In[138]:


# Adding contant value
X_test_sm = sm.add_constant(X_test)
X_test_sm.shape


# ### <font color = 'Green'> Prediction on Test Dataset using final model</font> 

# In[140]:


# making prediction using model 2 (final model)
y_test_pred = logm2.predict(X_test_sm)


# In[141]:


y_test_pred[:10]


# In[142]:


# Changing to dataframe of predicted probability
y_test_pred = pd.DataFrame(y_test_pred)
y_test_pred.head()


# In[143]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
y_test_df.head()


# In[144]:


# Removing index for both dataframes to append them side by side 
y_test_pred.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

# Appending y_test_df and y_test_pred
y_pred_final = pd.concat([y_test_df, y_test_pred],axis=1)
y_pred_final.head()


# In[145]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})

# Rearranging the columns

y_pred_final = y_pred_final.reindex(['Converted','Converted_Prob'], axis=1)

y_pred_final.head()


# In[146]:


# taking sensitivity-specificity method at 0.36 probability cutoff during training
y_pred_final['final_predicted'] = y_pred_final['Converted_Prob'].map(lambda x: 1 if x > 0.36 else 0)
y_pred_final.head()


# ### <font color = 'Green'> ROC Curve for Test</font> 

# In[148]:


# Drawing ROC curve for Test Set
fpr, tpr, thresholds = metrics.roc_curve(y_pred_final["Converted"], y_pred_final["Converted_Prob"], drop_intermediate = False )

draw_roc(y_pred_final["Converted"], y_pred_final["Converted_Prob"])


# ### <font color = 'Green'> Test set Model Evaluation</font> 

# In[150]:


# Finding Confusion metrics for 'y_train_pred_final' df
confusion_matrix = metrics.confusion_matrix(y_pred_final['Converted'], y_pred_final['final_predicted'])
print("*"*50,"\n")

#
print("Confusion Matrix")
print(confusion_matrix,"\n")

print("*"*50,"\n")

# Using UDF to calculate all metrices of logistic regression
logreg_all_metrics(confusion_matrix)


# <italic><span style="color:blue">Lead Score: </span></italic> Lead Score is assigned to the customers
# - The customers with a higher lead score have a higher conversion chance 
# - The customers with a lower lead score have a lower conversion chance.

# <italic><span style="color:blue">CONCLUSION </span></italic>
# 
# - Prioritize High-Impact Features:
#     Occupation_Working Professional: With the highest coefficient (3.7478), leads identified as working professionals should be prioritized as they have the highest likelihood of 
#        conversion.
#     Lead Origin_Lead Add Form: This feature also has a strong positive impact (3.6829). Focus on leads originating from the Lead Add Form.
#     Occupation_Businessman: Another significant feature (2.1988). Leads identified as businessmen should be given priority.
# - Enhance Website Engagement:
#     Total Time Spent on Website: This variable shows a positive impact on lead conversion (1.1180). Encourage potential leads to spend more time on the website through engaging content and 
#     interactive features.
# - Utilize Effective Communication Channels:
#     Last Activity_SMS Sent: This has a very high positive impact (2.0291). Continue sending SMS messages to leads as it significantly boosts conversion rates.
#     Last Activity_Email Opened: Also shows a strong positive impact (1.0177). Ensure that email campaigns are effective and track email opens to identify engaged leads.
# - Leverage Multiple Lead Sources:
#     Lead Source_Olark Chat: This source has a significant positive impact (1.2122). Utilize the Olark Chat feature on the website to engage with potential leads.
#     Lead Source_Welingak Website: Although less impactful than others, it still shows a positive effect (1.7369). Continue to optimize this source for lead generation.
# - Address Negative Indicators:
#     Last Activity_Email Bounced: This has a negative impact (-1.0726). Minimize email bounces by maintaining a clean email list and verifying email addresses.
# - Focus on Diverse Occupations:
#     Occupation_Other, Occupation_Student, and Occupation_Unemployed: These categories also show positive impacts. Tailor marketing strategies to effectively engage with these groups.
# - Monitor and Adjust Strategies:
#     Regularly review the model’s performance and adjust strategies based on new data and changing market conditions. This will help in maintaining high conversion rates and adapting to new challenges.
#     

# In[ ]:




