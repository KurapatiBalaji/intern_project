#!/usr/bin/env python
# coding: utf-8

# In[56]:


# Importing the libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Ignore harmless warnings 

import warnings 
warnings.filterwarnings("ignore")

# Set to display all the columns in dataset

pd.set_option("display.max_columns", None)

# Import psql to run queries 

import pandasql as psql


# In[57]:


# Load the Universal bank data

train = pd.read_csv(r"C:\Users\JANU\Desktop\train.csv",header=0)

test = pd.read_csv(r"C:\Users\JANU\Desktop\test.csv",header=0)

# Copy to back-up file

train_01 = train.copy()

test_01 = test.copy()


# In[58]:


train.head()


# In[59]:


test.head()


# In[60]:


# Change the name of variable
train= train.rename(columns = {'capital-gain': 'capital_gain'}, inplace = False)
train= train.rename(columns = {'capital-loss': 'capital_loss'}, inplace = False)
train= train.rename(columns = {'hours-per-week': 'hpw'}, inplace = False)
train= train.rename(columns = {'native-country': 'native_country'}, inplace = False)
train= train.rename(columns = {'income_>50K': 'income_50K'}, inplace = False)


# In[61]:


# Change the name of variable
test= test.rename(columns = {'capital-gain': 'capital_gain'}, inplace = False)
test= test.rename(columns = {'capital-loss': 'capital_loss'}, inplace = False)
test= test.rename(columns = {'hours-per-week': 'hpw'}, inplace = False)
test= test.rename(columns = {'native-country': 'native_country'}, inplace = False)


# In[62]:


train.shape


# In[63]:


test.shape


# In[64]:


train.info()


# In[65]:


train.nunique()


# In[66]:


train.isnull().sum()


# In[67]:


test.isnull().sum()


# In[68]:


train.duplicated().any()


# In[69]:


# Displaying Duplicate values with in dataset

train_dup = train[train.duplicated(keep='last')]

train_dup


# In[70]:


# Remove the identified duplicate records 

train = train.drop_duplicates()

# Display the shape of the dataset

train.shape


# In[71]:


# Re-setting the row index

train = train.reset_index(drop=True)


# In[72]:


#copy file to backup after deletion of duplicate records

train_bk_01 = train.copy()


# In[73]:


test.duplicated().any()


# In[74]:


#SimpleImputer

from sklearn.impute import SimpleImputer

imputer_str = SimpleImputer(missing_values=np.nan,strategy='most_frequent',fill_value=None,verbose=0,copy=True,add_indicator=False)

train ['workclass'] = imputer_str.fit_transform(train [['workclass']])
train ['occupation'] = imputer_str.fit_transform(train [['occupation']])
train ['native_country'] = imputer_str.fit_transform(train [['native_country']])


# In[75]:


train.isnull().sum()


# In[76]:


#Use LabelBinarizer to handle categorical data

from sklearn.preprocessing import LabelBinarizer

LB=LabelBinarizer()

train['gender']=LB.fit_transform(train[['gender']])
train['income_50K']=LB.fit_transform(train[['income_50K']])
test['gender']=LB.fit_transform(test[['gender']])


# In[77]:


#Use LabelEncoder for target variables

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train['workclass'] = le.fit_transform(train['workclass'])
train['education'] = le.fit_transform(train['education'])
train['marital-status'] = le.fit_transform(train['marital-status'])
train['occupation'] = le.fit_transform(train['occupation'])
train['relationship'] = le.fit_transform(train['relationship'])
train['race'] = le.fit_transform(train['race'])
train['native_country'] = le.fit_transform(train['native_country'])

test['workclass'] = le.fit_transform(test['workclass'])
test['education'] = le.fit_transform(test['education'])
test['marital-status'] = le.fit_transform(test['marital-status'])
test['occupation'] = le.fit_transform(test['occupation'])
test['relationship'] = le.fit_transform(test['relationship'])
test['race'] = le.fit_transform(test['race'])
test['native_country'] = le.fit_transform(test['native_country'])


# In[78]:


train.info()


# In[79]:


test.info()


# In[80]:


train.head(10)


# In[81]:


train.shape


# In[82]:


a_ul=round(train.age.mean()+3*train.age.std(),3)
a_ll=round(train.age.mean()-3*train.age.std(),3)
train_01=train[(train.age>a_ll)&(train.age<a_ul)]
train_01.shape


# In[83]:


train_01_el=train[(train.age<a_ll)|(train.age>a_ul)]
train_01_el


# In[84]:


f_ul=round(train_01.fnlwgt.mean()+3*train_01.fnlwgt.std(),3)
f_ll=round(train_01.fnlwgt.mean()-3*train_01.fnlwgt.std(),3)
train_02=train_01[(train_01.fnlwgt>f_ll)&(train_01.fnlwgt<f_ul)]
train_02.shape


# In[85]:


train_02_el=train_01[(train_01.fnlwgt<f_ll)|(train_01.fnlwgt>f_ul)]
train_02_el


# In[86]:


g_ul=round(train_02.capital_gain.mean()+3*train_02.capital_gain.std(),3)
g_ll=round(train_02.capital_gain.mean()-3*train_02.capital_gain.std(),3)
train_03=train_02[(train_02.capital_gain>g_ll)&(train_02.capital_gain<g_ul)]
train_03.shape


# In[87]:


train_03_el=train_02[(train_02.capital_gain<g_ll)|(train_02.capital_gain>g_ul)]
train_03_el


# In[88]:


l_ul=round(train_03.capital_loss.mean()+3*train_03.capital_loss.std(),3)
l_ll=round(train_03.capital_loss.mean()-3*train_03.capital_loss.std(),3)
train_04=train_03[(train_03.capital_loss>l_ll)&(train_02.capital_loss<l_ul)]
train_04.shape


# In[89]:


train_04_el=train_03[(train_03.capital_loss<l_ll)|(train_03.capital_loss>l_ul)]
train_04_el


# In[90]:


n_ul=round(train_04.native_country.mean()+3*train_04.native_country.std(),3)
n_ll=round(train_04.native_country.mean()-3*train_04.native_country.std(),3)
train_05=train_04[(train_04.native_country>n_ll)&(train_04.native_country<n_ul)]
train_05.shape


# In[91]:


train_05_el=train_04[(train_04.native_country<n_ll)|(train_04.native_country>n_ul)]
train_05_el


# In[92]:


test.shape


# In[93]:


a_ul=round(test.age.mean()+3*test.age.std(),3)
a_ll=round(test.age.mean()-3*test.age.std(),3)
test_01=test[(test.age>a_ll)&(test.age<a_ul)]
test_01.shape


# In[94]:


test_01_el=test[(test.age<a_ll)|(test.age>a_ul)]
test_01_el


# In[95]:


f_ul=round(test_01.fnlwgt.mean()+3*test_01.fnlwgt.std(),3)
f_ll=round(test_01.fnlwgt.mean()-3*test_01.fnlwgt.std(),3)
test_02=test_01[(test_01.fnlwgt>f_ll)&(test_01.fnlwgt<f_ul)]
test_02.shape


# In[96]:


test_02_el=test_01[(test_01.fnlwgt<f_ll)|(test_01.fnlwgt>f_ul)]
test_02_el


# In[97]:


g_ul=round(test_02.capital_gain.mean()+3*test_02.capital_gain.std(),3)
g_ll=round(test_02.capital_gain.mean()-3*test_02.capital_gain.std(),3)
test_03=test_02[(test_02.capital_gain>g_ll)&(test_02.capital_gain<g_ul)]
test_03.shape


# In[98]:


test_03_el=test_02[(test_02.capital_gain<g_ll)|(test_02.capital_gain>g_ul)]
test_03_el


# In[99]:


l_ul=round(test_03.capital_loss.mean()+3*test_03.capital_loss.std(),3)
l_ll=round(test_03.capital_loss.mean()-3*test_03.capital_loss.std(),3)
test_04=test_03[(test_03.capital_loss>l_ll)&(test_02.capital_loss<l_ul)]
test_04.shape


# In[100]:


test_04_el=test_03[(test_03.capital_loss<l_ll)|(test_03.capital_loss>l_ul)]
test_04_el


# In[101]:


n_ul=round(test_04.native_country.mean()+3*test_04.native_country.std(),3)
n_ll=round(test_04.native_country.mean()-3*test_04.native_country.std(),3)
test_05=test_04[(test_04.native_country>n_ll)&(test_04.native_country<n_ul)]
test_05.shape


# In[102]:


test_05_el=test_04[(test_04.native_country<n_ll)|(test_04.native_country>n_ul)]
test_05_el


# In[103]:


# Count the target or dependent variable by '0' & '1' and their proportion 
# (> 10 : 1, then the dataset is imbalance data)

income_count = train.income_50K.value_counts()

print('Class 0:',income_count[0])

print('Class 1:', income_count[1])

print('Proportion:', round(income_count[0] / income_count[1], 2), ': 1')

print('Total train records:', len(train))


# In[104]:


col=['workclass','education','marital-status','occupation','relationship','race','native_country']


# In[105]:


plt.figure(figsize=(15,10))
sns.countplot(data = train, x = col[0])


# In[106]:


plt.figure(figsize=(15,10))
sns.countplot(data = train, x = col[1])


# In[107]:


plt.figure(figsize=(15,10))
sns.countplot(data = train, x = col[2])


# In[108]:


plt.figure(figsize=(15,10))
sns.countplot(data = train, x = col[3])


# In[109]:


train['education'].value_counts().plot(kind='pie')
plt.show()


# In[110]:


# Identify the independent and Target (dependent) variables

IndepVar = []
for col in train.columns:
    if col != 'income_50K':
        IndepVar.append(col)

TargetVar = 'income_50K'

x= train[IndepVar]
y= train[TargetVar]


# In[111]:


x


# In[112]:


y


# In[113]:


# Split the data into train and test (random sampling)

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[114]:


cols=['age','workclass','fnlwgt','education','educational-num','marital-status','occupation','relationship','race','capital_gain','capital_loss','hpw','native_country']


# In[115]:


# Scaling the features by using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

mmscaler = MinMaxScaler(feature_range=(0, 1))

x_train[cols] = mmscaler.fit_transform(x_train[cols])

x_train = pd.DataFrame(x_train)

x_test[cols] = mmscaler.fit_transform(x_test[cols])

x_test = pd.DataFrame(x_test)


# In[116]:


x_train


# In[117]:


x_test


# In[118]:


# Load the Results dataset

CSResults = pd.read_csv(r"C:\Users\JANU\Documents\HTResults.csv", header=0)

CSResults.head()


# In[119]:


# Build the Calssification models and compare the results

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb

# Create objects of classification algorithm with default hyper-parameters

ModelLR = LogisticRegression()
ModelDC = DecisionTreeClassifier()
ModelRF = RandomForestClassifier()
ModelET = ExtraTreesClassifier()
ModelKNN = KNeighborsClassifier(n_neighbors=5)
ModelGNB = GaussianNB()
ModelXGB = XGBClassifier(n_estimators=1, max_depth=3, eval_metric='mlogloss')
ModelSVM = SVC(probability=True)
ModelBAG = BaggingClassifier(base_estimator=None, n_estimators=100, max_samples=1.0, max_features=1.0,
                             bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False,
                             n_jobs=None, random_state=None, verbose=0)
ModelGB = GradientBoostingClassifier()

ModelLGB = lgb.LGBMClassifier()

# Evalution matrix for all the algorithms

#MM = [ModelLR, ModelDC, ModelRF, ModelET, ModelKNN, ModelGNB, ModelSVM, ModelXGB, ModelLGB,ModelBAG,ModelGB,ModelLGB]
MM = [ModelLR, ModelDC, ModelRF, ModelET, ModelKNN, ModelGNB,  ModelSVM, ModelXGB, ModelLGB,ModelBAG,ModelGB]
for models in MM:
    
    # Fit the model
    
    models.fit(x_train, y_train)
    
    # Prediction
    
    y_pred = models.predict(x_test)
    y_pred_prob = models.predict_proba(x_test)
    
    # Print the model name
    
    print('Model Name: ', models)
    
    # confusion matrix in sklearn

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # actual values

    actual = y_test

    # predicted values

    predicted = y_pred

    # confusion matrix

    matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
    print('Confusion matrix : \n', matrix)

    # outcome values order in sklearn

    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy

    C_Report = classification_report(actual,predicted,labels=[1,0])

    print('Classification report : \n', C_Report)

    # calculating the metrics

    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);

    # Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
    # A model with a score of +1 is a perfect model and -1 is a poor model

    from math import sqrt

    mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

    print('Accuracy :', round(accuracy*100, 2),'%')
    print('Precision :', round(precision*100, 2),'%')
    print('Recall :', round(sensitivity*100,2), '%')
    print('F1 Score :', f1Score)
    print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
    print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
    print('MCC :', MCC)

    # Area under ROC curve 

    from sklearn.metrics import roc_curve, roc_auc_score

    print('roc_auc_score:', round(roc_auc_score(y_test, y_pred), 3))
    
    # ROC Curve
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    logit_roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, models.predict_proba(x_test)[:,1])
    plt.figure()
    # plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot(fpr, tpr, label= 'Classification Model' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    print('-----------------------------------------------------------------------------------------------------')
    #----------------------------------------------------------------------------------------------------------
    new_row = {'Model Name' : models,
               'True_Positive' : tp, 
               'False_Negative' : fn, 
               'False_Positive' : fp,
               'True_Negative' : tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC':MCC,
               'ROC_AUC_Score':roc_auc_score(y_test, y_pred),
               'Balanced Accuracy':balanced_accuracy}
    CSResults = CSResults.append(new_row, ignore_index=True)


# In[120]:


CSResults


# In[121]:


#GradientBoostingClassifier()
#LGBMClassifier()
#SVC(probability=True)


# In[122]:


# Hyperparameter tuning by GridSearchCV

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
GS_grid = {
    #'bootstrap': [True, False],
    #'max_features': ['auto', 'sqrt', 'log2'],
    'n_estimators': [100, 200, 300, 400, 500],
    #'bootstrap_features':[True, False]
}

# Create object for model

ModelGB = GradientBoostingClassifier()

# Instantiate the grid search model

Grid_search = GridSearchCV(estimator = ModelGB, param_grid = GS_grid, cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data

Grid_search.fit(x_train,y_train)


# In[123]:


# Best parameter from gridseachCV

Grid_search.best_params_


# In[124]:


# Load the Results dataset

CSResults = pd.read_csv(r"C:\Users\JANU\Documents\HTResults.csv", header=0)

CSResults.head()


# In[126]:


# Build the Calssification models and compare the results


from sklearn.ensemble import GradientBoostingClassifier


# Create objects of classification algorithm with default hyper-parameters

ModelGB = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,n_estimators=400, subsample=1.0,criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None,random_state=None,max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False,validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)



 
MM2= [ModelGB]
for models in MM2:
            
    # Train the model training dataset
    
    models.fit(x_train, y_train)
    
    # Prediction the model with test dataset
    
    y_pred = models.predict(x_test)
    y_pred_prob = models.predict_proba(x_test)
    
    # Print the model name
    
    print('Model Name: ', models)
    
    # confusion matrix in sklearn

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    # actual values

    actual = y_test

    # predicted values

    predicted = y_pred

    # confusion matrix

    matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
    print('Confusion matrix : \n', matrix)

    # outcome values order in sklearn

    tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
    print('Outcome values : \n', tp, fn, fp, tn)

    # classification report for precision, recall f1-score and accuracy

    C_Report = classification_report(actual,predicted,labels=[1,0])

    print('Classification report : \n', C_Report)

    # calculating the metrics

    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);

    # Matthews Correlation Coefficient (MCC). Range of values of MCC lie between -1 to +1. 
    # A model with a score of +1 is a perfect model and -1 is a poor model

    from math import sqrt

    mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

    print('Accuracy :', round(accuracy*100, 2),'%')
    print('Precision :', round(precision*100, 2),'%')
    print('Recall :', round(sensitivity*100,2), '%')
    print('F1 Score :', f1Score)
    print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
    print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
    print('MCC :', MCC)

    # Area under ROC curve 

    from sklearn.metrics import roc_curve, roc_auc_score

    print('roc_auc_score:', round(roc_auc_score(actual, y_pred), 3))
    
    # ROC Curve
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    Model_roc_auc = roc_auc_score(actual, y_pred)
    fpr, tpr, thresholds = roc_curve(actual, models.predict_proba(x_test)[:,1])
    plt.figure()
    #
    plt.plot(fpr, tpr, label= 'Classification Model' % Model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    print('-----------------------------------------------------------------------------------------------------')
    #----------------------------------------------------------------------------------------------------------
    new_row = {'Model Name' : models,
               'True_Positive': tp,
               'False_Negative': fn, 
               'False_Positive': fp, 
               'True_Negative': tn,
               'Accuracy' : accuracy,
               'Precision' : precision,
               'Recall' : sensitivity,
               'F1 Score' : f1Score,
               'Specificity' : specificity,
               'MCC': MCC,
               'ROC_AUC_Score':roc_auc_score(actual, y_pred),
               'Balanced Accuracy':balanced_accuracy}
    CSResults=  CSResults.append(new_row, ignore_index=True)


# In[127]:


CSResults


# In[ ]:




