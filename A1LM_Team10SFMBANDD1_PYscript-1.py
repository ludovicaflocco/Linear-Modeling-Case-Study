# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:24:18 2019

@author: Group 10, DAT-5303 - SFMBANDD1
Workdirectory: /Users/Desktop/Hult/Machine_Learning/Assignment/Pregnancy
Purpose: Exploring pregnancy dataset and building model to predict birthweight 

AGENDA:
    1)  Initial Data Set-up
    2)  Data Exploration
    3)  Imputing Missing Values
    4)  Flagging Outliers 
    5)  Correlation Analysis
    6)  Factorization & Dummy Variables
    7)  Feature Engineering
    8)  OLS (Ordinary Least Square) Method Full Model
    9)  Preparing Data for Scikit-learn
    10) OLS LR (Linear Regression) Significant Model
    11) KNN Model
    12) Scikit Learn LR Significant Model
    13) Result Overview
    14) Creation of Charts & Graphs (report writing)
"""

####################################################
# 1) Initial Data Set-up
####################################################

# Loading libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf # regression modeling
import seaborn as sns
import matplotlib.pyplot as plt


# For KNN model
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
from sklearn.model_selection import cross_val_score # k-folds cross validation


#######################


file = 'birthweight_feature_set.xlsx'

birthweight = pd.read_excel(file)



####################################################
# 2) Data Exploration
####################################################

# Extracting column names
birthweight.columns
# mage = mother's age 
# meduc = mother's education
# monpre = month prenatal care began 
# npvis = number of prenatal visits
# fage = father's age
# feduc = father's education 
# omaps = one minute APGAR score 
# fmaps = five minutes APGAR score 
# cigs = average cigarettes per day
# drink = average drinks per week 
# male = 1 if baby is male 
# mwhte = 1 if mother is white
# mblck = 1 if mother is black 
# moth = 1 if mother is other 
# fwhte = 1 if father is white 
# fblck = 1 if father is black 
# goth = 1 if father is other
# bwght = birthweigh in grams 


# Dimensions of the DataFrame
birthweight.shape #18 variables, 196 observations

# Information about each variable 
birthweight.info()

# Descriptive statistics
birthweight.describe().round(2)

# Analysing the target variable
birthweight.sort_values('bwght', ascending = False)



####################################################
# 3) Imputing Missing Values
####################################################

# Analysing missing values
print(
      birthweight
      .isnull()
      .sum()
      )


#######################


# Building a loop for creating flagged columns for missing values

for col in birthweight:  

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if birthweight[col].isnull().any():
        birthweight['m_'+col] = birthweight[col].isnull().astype(int)
        

# Columns with missing values & count of missing values       
# meduc    03
# npvis    03
# feduc    07
     

# Creating a dataset with non-null values       
df_dropped = birthweight.dropna()


#######################


# Creating histograms to analyse distribution of data
sns.distplot(df_dropped['meduc']) # Missing values = Median (skewed distribution)
plt.show()

sns.distplot(df_dropped['npvis']) #Missing values = Mean (normal distribution)
plt.show()

sns.distplot(df_dropped['feduc']) #Missing values = Median (skewed distribution)
plt.show()


#######################


# Imputing npvis missing values with mean

fill = birthweight['npvis'].mean()

birthweight['npvis'] = birthweight['npvis'].fillna(fill)


# Imputing meduc & feduc missing values with median


fill = birthweight['meduc'].median()

birthweight['meduc'] = birthweight['meduc'].fillna(fill)



fill = birthweight['feduc'].median()

birthweight['feduc'] = birthweight['feduc'].fillna(fill)


#######################


# Checking overall dataset to verify no missing values are remaining
print(
      birthweight
      .isnull()
      .any()
      .any()
      )



####################################################
# 4) Flagging Outliers
####################################################

# Analysing variables to decide lower & upper cut-off points

# mage 
plt.subplot(2, 2, 1)
sns.distplot(birthweight['mage'],
             color = 'g')

plt.xlabel('mage')


# meduc
plt.subplot(2, 2, 2)
sns.distplot(birthweight['meduc'],
             color = 'y')

plt.xlabel('meduc')


# monpre
plt.subplot(2, 2, 3)
sns.distplot(birthweight['monpre'],
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('monpre')


# npvis
plt.subplot(2, 2, 4)

sns.distplot(birthweight['npvis'],
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('npvis')


plt.tight_layout()

plt.show()


#######################


# fage
plt.subplot(2, 2, 1)
sns.distplot(birthweight['fage'],
             color = 'g')

plt.xlabel('fage')


# feduc
plt.subplot(2, 2, 2)
sns.distplot(birthweight['feduc'],
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('feduc')


# omaps
plt.subplot(2, 2, 3)

sns.distplot(birthweight['omaps'],
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('omaps')


# fmaps
plt.subplot(2, 2, 4)
sns.distplot(birthweight['fmaps'],
             color = 'y')

plt.xlabel('fmaps')


plt.tight_layout()

plt.show()


#######################


# cigs
plt.subplot(2, 2, 1)
sns.distplot(birthweight['cigs'],
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('cigs')


# drink
plt.subplot(2, 2, 2)

sns.distplot(birthweight['drink'],
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('drink')


# bwght
plt.subplot(2, 2, 3)
sns.distplot(birthweight['bwght'],
             color = 'g')

plt.xlabel('bwght')


plt.tight_layout()

plt.show()


#######################


# Tuning & flagging outliers

mage_low = 20

mage_high = 55

overall_low_meduc = 10

monpre_low = 0

monpre_high = 7

npvis_low = 5

npvis_high = 18

fage_low = 20

fage_high = 62

overall_low_feduc = 7

overall_low_omaps = 4

overall_low_fmaps = 6

overall_cigs = 19

bwght_low = 2500

bwght_high = 4500

overall_drink = 11


#######################


# Creating new variable 'race' (string column for racial data in 6 diff. columns)
# Creating new variable 'cEdu' (combined education of mother and father)

birthweight['race'] = 0
birthweight['cEdu'] = 0
abc = birthweight['cigs']
for val in enumerate(birthweight.loc[ : , 'fwhte']):
      birthweight.loc[val[0], 'race'] =   str(birthweight.loc[val[0], 'mwhte']) + \
                                          str(birthweight.loc[val[0], 'mblck']) + \
                                          str(birthweight.loc[val[0], 'moth']) + \
                                          str(birthweight.loc[val[0], 'fwhte']) + \
                                          str(birthweight.loc[val[0], 'fblck']) + \
                                          str(birthweight.loc[val[0], 'foth'])
      birthweight.loc[val[0], 'cEdu'] =   birthweight.loc[val[0], 'meduc'] + \
                                          birthweight.loc[val[0], 'feduc']


#######################  
               
                         
# Creating outlier flags by building loops for outlier imputation
                                                                                
# mage                                         

birthweight['out_mage'] = 0

for val in enumerate(birthweight.loc[ : , 'mage']):
    
    if val[1] >= mage_high:
        birthweight.loc[val[0], 'out_mage'] = 1
        
    if val[1] <= mage_low:
        birthweight.loc[val[0], 'out_mage'] = -1
        

# meduc

birthweight['out_meduc'] = 0

for val in enumerate(birthweight.loc[ : , 'meduc']):
            
    if val[1] <= overall_low_meduc:
        birthweight.loc[val[0], 'out_meduc'] = -1


# monpre

birthweight['out_monpre'] = 0

for val in enumerate(birthweight.loc[ : , 'monpre']):
    
    if val[1] >= monpre_high:
        birthweight.loc[val[0], 'out_monpre'] = 1
        
    if val[1] <= monpre_low:
        birthweight.loc[val[0], 'out_monpre'] = -1


# npvis

birthweight['out_npvis'] = 0

for val in enumerate(birthweight.loc[ : , 'npvis']):
    
    if val[1] >= npvis_high:
        birthweight.loc[val[0], 'out_npvis'] = 1
        
    if val[1] <= npvis_low:
        birthweight.loc[val[0], 'out_npvis'] = -1
                

# fage

birthweight['out_fage'] = 0

for val in enumerate(birthweight.loc[ : , 'fage']):
    
    if val[1] >= fage_high:
        birthweight.loc[val[0], 'out_fage'] = 1
        
    if val[1] <= fage_low:
        birthweight.loc[val[0], 'out_fage'] = -1
        

# feduc

birthweight['out_feduc'] = 0

for val in enumerate(birthweight.loc[ : , 'feduc']):   
        
    if val[1] <= overall_low_feduc:
        birthweight.loc[val[0], 'out_feduc'] = -1


# omaps

birthweight['out_omaps'] = 0

for val in enumerate(birthweight.loc[ : , 'omaps']):
        
    if val[1] <= overall_low_omaps:
        birthweight.loc[val[0], 'out_omaps'] = -1


# fmaps       

birthweight['out_fmaps'] = 0

for val in enumerate(birthweight.loc[ : , 'fmaps']):

    if val[1] <= overall_low_fmaps:
        birthweight.loc[val[0], 'out_fmaps'] = -1
        

# cigs

birthweight['out_cigs'] = 0

for val in enumerate(birthweight.loc[ : , 'cigs']):
            
    if val[1] >= overall_cigs:
        birthweight.loc[val[0], 'out_cigs'] = 1
        

# bwght

birthweight['out_bwght'] = 0

for val in enumerate(birthweight.loc[ : , 'bwght']):
    
    if val[1] >= bwght_high:
        birthweight.loc[val[0], 'out_bwght'] = 1
        
    if val[1] <= bwght_low:
        birthweight.loc[val[0], 'out_bwght'] = -1
        

# drink

birthweight['out_drink'] = 0

for val in enumerate(birthweight.loc[ : , 'drink']):
            
    if val[1] >= overall_drink:
        birthweight.loc[val[0], 'out_drink'] = 1



####################################################
# 5) Correlation Analysis
####################################################

# Building the correlation analysis
        
df_corr = birthweight.corr().round(2)

print(df_corr)

df_corr.loc['bwght'].sort_values(ascending = False)
        

#######################


# Creating the correlation heatmap

# Using palplot to view a color scheme

sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))
 
df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)

# #Save the plot
# plt.savefig('A1LM_Team10SFMBANDD1_Heatmap.png')
plt.show()



####################################################
# 6) Factorization & Dummy Variables
####################################################

# fmaps
fmaps_dummies = pd.get_dummies(list(birthweight['fmaps']), prefix = 'fmaps', drop_first = True)

# omaps
omaps_dummies = pd.get_dummies(list(birthweight['omaps']), prefix = 'omaps', drop_first = True)

# drink
drink_dummies = pd.get_dummies(list(birthweight['drink']), prefix = 'drink', drop_first = True)

# meduc
meduc_dummies = pd.get_dummies(list(birthweight['meduc']), prefix = 'meduc', drop_first = True)

# feduc
feduc_dummies = pd.get_dummies(list(birthweight['feduc']), prefix = 'feduc', drop_first = True)

# race
race_dummies = pd.get_dummies(list(birthweight['race']), prefix = 'race', drop_first = True)

# npvis
npvis_dummies = pd.get_dummies(list(birthweight['npvis']), prefix = 'npvis', drop_first = True)

# cigs
cigs_dummies = pd.get_dummies(list(birthweight['cigs']), prefix = 'cigs', drop_first = True)


#######################


# Concatenating dummy variables in a dataframe and saving the new dataframe
birthweight_2 = pd.concat(
        [birthweight.loc[:,:],
         fmaps_dummies, drink_dummies, 
         meduc_dummies, race_dummies,
         omaps_dummies, npvis_dummies,
         cigs_dummies, feduc_dummies],
         axis = 1)



####################################################
# 7) Feature Engineering
####################################################

# Creating new variable for combined data of cigs & drink
birthweight_2['cigolic'] = birthweight_2['cigs'] * birthweight_2['drink']

# Creating new variable for combined data of meduc & feduc
birthweight_2['edu'] = birthweight_2['feduc'] * birthweight_2['meduc']

# Creating new variable for combined data of foth & moth
birthweight_2['oth'] = birthweight_2['foth'] * birthweight_2['moth']

# Creating new variable for combined data of male & mwhite
birthweight_2['C_wM'] = birthweight_2['male'] * birthweight_2['mwhte']

# Creating new variable for combined data of male & mblack
birthweight_2['C_bM'] = birthweight_2['male'] * birthweight_2['mblck']

# Creating new variable for combined data of male & moth
birthweight_2['C_oM'] = birthweight_2['male'] * birthweight_2['moth']

# Creating new variable for combined data of male & fblck
birthweight_2['C_bF'] = birthweight_2['male'] * birthweight_2['fblck']



####################################################
# 8) OLS (Ordinary Least Square) Method Full Model
####################################################

# Creating a statsmodel with all possible variables
# Analysing relation of various variables on birhtweight

lm_full = smf.ols(formula = """bwght ~    mage +                                          
                                          monpre +                                          
                                          fage +
                                          birthweight_2['feduc_7.0'] +
                                          birthweight_2['feduc_8.0'] +
                                          birthweight_2['feduc_10.0'] +
                                          birthweight_2['feduc_11.0'] +
                                          birthweight_2['feduc_12.0'] +
                                          birthweight_2['feduc_13.0'] +
                                          birthweight_2['feduc_14.0'] +
                                          birthweight_2['feduc_15.0'] +
                                          birthweight_2['feduc_16.0'] +
                                          birthweight_2['feduc_17.0'] +                                         
                                          birthweight_2['cigs_1'] +
                                          birthweight_2['cigs_2'] +
                                          birthweight_2['cigs_3'] +
                                          birthweight_2['cigs_4'] +
                                          birthweight_2['cigs_5'] +
                                          birthweight_2['cigs_6'] +
                                          birthweight_2['cigs_7'] +
                                          birthweight_2['cigs_8'] +
                                          birthweight_2['cigs_9'] +
                                          birthweight_2['cigs_10'] +
                                          birthweight_2['cigs_11'] +
                                          birthweight_2['cigs_12'] +
                                          birthweight_2['cigs_13'] +
                                          birthweight_2['cigs_14'] +
                                          birthweight_2['cigs_15'] +
                                          birthweight_2['cigs_16'] +
                                          birthweight_2['cigs_17'] +
                                          birthweight_2['cigs_18'] +
                                          birthweight_2['cigs_19'] +
                                          birthweight_2['cigs_20'] +
                                          birthweight_2['cigs_21'] +
                                          birthweight_2['cigs_22'] +
                                          birthweight_2['cigs_23'] +
                                          birthweight_2['cigs_24'] +
                                          birthweight_2['cigs_25'] +                                          
                                          male +
                                          mwhte +
                                          mblck +
                                          moth +
                                          fwhte +
                                          fblck +
                                          foth +                                          
                                          m_meduc +
                                          m_npvis +
                                          m_feduc +                                          
                                          cEdu +
                                          out_mage +
                                          out_meduc +
                                          out_monpre +
                                          out_npvis +
                                          out_fage +
                                          out_feduc +
                                          out_omaps +
                                          out_fmaps +
                                          out_cigs +
                                          out_bwght +
                                          out_drink +
                                          birthweight_2['omaps_3'] +
                                          birthweight_2['omaps_4'] +
                                          birthweight_2['omaps_5'] +
                                          birthweight_2['omaps_6'] +
                                          birthweight_2['omaps_7'] +
                                          birthweight_2['omaps_8'] +
                                          birthweight_2['omaps_9'] +
                                          birthweight_2['omaps_10'] +                                          
                                          birthweight_2['fmaps_6'] +
                                          birthweight_2['fmaps_7'] +
                                          birthweight_2['fmaps_8'] +
                                          birthweight_2['fmaps_9'] +
                                          birthweight_2['fmaps_10'] +
                                          birthweight_2['drink_1'] +
                                          birthweight_2['drink_2'] +
                                          birthweight_2['drink_3'] +
                                          birthweight_2['drink_4'] +
                                          birthweight_2['drink_5'] +
                                          birthweight_2['drink_6'] +
                                          birthweight_2['drink_7'] +
                                          birthweight_2['drink_8'] +
                                          birthweight_2['drink_9'] +
                                          birthweight_2['drink_10'] +
                                          birthweight_2['drink_11'] +
                                          birthweight_2['drink_12'] +
                                          birthweight_2['drink_13'] +
                                          birthweight_2['drink_14'] +
                                          birthweight_2['meduc_10.0'] +
                                          birthweight_2['meduc_11.0'] +
                                          birthweight_2['meduc_12.0'] +
                                          birthweight_2['meduc_13.0'] +
                                          birthweight_2['meduc_14.0'] +
                                          birthweight_2['meduc_15.0'] +
                                          birthweight_2['meduc_16.0'] +
                                          birthweight_2['meduc_17.0'] +
                                          birthweight_2['race_001010'] + 
                                          birthweight_2['race_001100'] +
                                          birthweight_2['race_010001'] +
                                          birthweight_2['race_010010'] +
                                          birthweight_2['race_010100'] +
                                          birthweight_2['race_100100'] +
                                          birthweight_2['npvis_3.0'] +
                                          birthweight_2['npvis_5.0'] +
                                          birthweight_2['npvis_6.0'] +
                                          birthweight_2['npvis_7.0'] +
                                          birthweight_2['npvis_8.0'] +
                                          birthweight_2['npvis_9.0'] +
                                          birthweight_2['npvis_10.0'] +
                                          birthweight_2['npvis_11.0'] +
                                          birthweight_2['npvis_12.0'] +
                                          birthweight_2['npvis_13.0'] +
                                          birthweight_2['npvis_14.0'] +
                                          birthweight_2['npvis_15.0'] +
                                          birthweight_2['npvis_16.0'] +
                                          birthweight_2['npvis_17.0'] +
                                          birthweight_2['npvis_18.0'] +
                                          birthweight_2['npvis_19.0'] +
                                          birthweight_2['npvis_20.0'] +
                                          birthweight_2['npvis_25.0'] +
                                          birthweight_2['npvis_30.0'] +
                                          birthweight_2['npvis_31.0'] +
                                          birthweight_2['npvis_35.0']
                                          """,
                  data = birthweight_2)


#######################


# Fitting results
results = lm_full.fit()

# RSquare value of full LM model
rsq_lm_full = results.rsquared.round(3)

# Printing summary statistics of the model
print(results.summary())


print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")
    


####################################################
# 9) Preparing Data for Scikit-learn
####################################################

## Creating data with selective features 
birthweight_data   = birthweight_2.loc[:,['mage',                                         
                                          'cigs',                                 
                                          'male',
                                          'mwhte',
                                          'mblck',
                                          'moth',
                                          'fwhte',
                                          'fblck',
                                          'foth',
                                          'm_meduc',
                                          'm_npvis',
                                          'm_feduc',
                                          'cEdu',
                                          'C_wM',
                                          'C_bM',
                                          'C_bF',
                                          'oth',
                                          'cigolic',
                                          'edu',
                                          'out_mage',
                                          'out_meduc',
                                          'out_monpre',
                                          'out_npvis',
                                          'out_fage',
                                          'out_feduc',                                          
                                          'out_cigs',
                                          'out_bwght',
                                          'out_drink',
                                          'drink_4',
                                          'drink_5',
                                          'drink_6',
                                          'drink_7',
                                          'drink_8',
                                          'drink_9',
                                          'drink_10',
                                          'drink_11',
                                          'drink_12',
                                          'drink_13',
                                          'drink_14',
                                          'npvis'                                                                                                                  
                                          ]]


#######################


# Preparing the target variable
birthweight_target = birthweight_2.loc[:, 'bwght']


# Preparing test and train datsets
X_train, X_test, y_train, y_test = train_test_split(
            birthweight_data,
            birthweight_target,
            test_size = 0.1,
            random_state = 508)



####################################################
# 10) OLS LR (Linear Regression) Significant Model
####################################################

birthweight_OLS_train = pd.concat([X_train, y_train], axis=1)
birthweight_OLS_test = pd.concat([X_test, y_test], axis=1)

lm_significant = smf.ols(formula = """bwght ~   mage + 
                                                cigs + 
                                                male + 
                                                mwhte +
                                                mblck +
                                                moth +
                                                fwhte +
                                                fblck +
                                                foth +
                                                m_meduc +
                                                m_npvis +
                                                m_feduc +
                                                cEdu +
                                                C_wM +
                                                C_bM +
                                                C_bF +
                                                oth +
                                                cigolic +
                                                edu +
                                                out_mage +
                                                out_meduc +
                                                out_monpre +
                                                out_npvis +
                                                out_fage +
                                                out_feduc +                                          
                                                out_cigs +
                                                out_bwght +
                                                out_drink +
                                                drink_4 +
                                                drink_5 +
                                                drink_6 +
                                                drink_7 +
                                                drink_8 +
                                                drink_9 +
                                                drink_10 +
                                                drink_11 +
                                                drink_12 +
                                                drink_13 +
                                                drink_14 +
                                                npvis
                                                """, 
                  data=birthweight_OLS_train)


#######################


# Fitting results
results = lm_significant.fit()
results.rsquared_adj.round(3)


# Printing summary statistics of the model
print(results.summary())

rsq_lm_significant = results.rsquared.round(3)

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")


    
####################################################
# 11) KNN Model
####################################################

# Initiate list for accuracy
training_accuracy = []
test_accuracy = []


# Define range for accuracy checking
neighbors_settings = range(1, 51)

# Looping to append lists with accuracy
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
    
#######################
    

# Plotting the accuracy graph
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

print(max(test_accuracy))

# checking for best test accuracy
print("Best test accuracy is at N = ",test_accuracy.index(max(test_accuracy)))


########################


# The best results occur when k = 10
# Building a model with k = 10
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = test_accuracy.index(max(test_accuracy)))



#######################


# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)


# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)


# The score is directly comparable to R-Square
print(y_score_knn_optimal)



#######################


# Generating Predictions based on the optimal KNN model
knn_reg_optimal_pred = knn_reg_fit.predict(X_test)

# Predictions
y_pred = knn_reg.predict(X_test)
print(f"""
Test set predictions:
{y_pred.round(2)}
""")

    
    
####################################################
# 12) Scikit Learn LR Significant Model
####################################################

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(
            birthweight_data,
            birthweight_target,
            test_size = 0.1,
            random_state = 508)


# Preparing the model
lr = LinearRegression(fit_intercept = False)

# Fitting the model
lr_fit = lr.fit(X_train, y_train)


# Predictions
lr_pred = lr_fit.predict(X_test)


print(f"""
Test set predictions:
{lr_pred.round(2)}
""")

    
#######################
    
    
# Saving the prediction of results onto an excel (.xlsx) sheets
pd.DataFrame(lr_pred).to_excel('A1LM_Team10SFMBANDD1_PredResults.xlsx', index=True)


#######################


# Scoring the model
y_score_ols_optimal = lr_fit.score(X_test, y_test)


# The score is directly comparable to R-Square
print("Fit score of scikit LR model: ",y_score_ols_optimal)


# Comparing the testing score to the training score

print('Training Score', lr.score(X_train, y_train).round(4))
print('Testing Score:', lr.score(X_test, y_test).round(4))

cv_lr_3 = cross_val_score(lr,
                          birthweight_data,
                          birthweight_target,
                          cv = 3)

print("Cross validation score of LR: ", (pd.np.mean(cv_lr_3)))



####################################################
# 13) Result Overview
####################################################

# Printing model results
print(f"""
Optimal model KNN score:       {y_score_knn_optimal.round(3)}
Optimal scikitLR model score:  {y_score_ols_optimal.round(3)}
CrossValidation (CV 3) score:  {pd.np.mean(cv_lr_3).round(3)}
R-Square OLS Full:             {rsq_lm_full.round(3)}
R-Square OLS Optimal:          {rsq_lm_significant.round(3)}
""")



####################################################
# 14) Creation of Charts & Graphs (report writing)
####################################################

# creating a bar graph comparing avg. birthweight against mage (in quantiles)
quant = birthweight['mage'].quantile([0.25, 0.50, 0.75, 1])
for val in enumerate(birthweight.loc[ : , 'mage']):
      if val[1] <= quant.iloc[0]:
            birthweight.loc[val[0], 'quant'] = 25
            
      elif val[1] <= quant.iloc[1]: 
            birthweight.loc[val[0], 'quant'] = 50
            
      elif val[1] <= quant.iloc[2]:
            birthweight.loc[val[0], 'quant'] = 75
            
      else:
            birthweight.loc[val[0], 'quant'] = 100
            
x = birthweight.groupby('quant')['bwght'].mean()
objects = ('Q1 ' + str(quant.iloc[0]), 'Q2 ' + str(quant.iloc[1]), 'Q3 ' + str(quant.iloc[2]), 'Q4 ' + str(quant.iloc[3]))
y_pos = np.arange(len(objects))
performance = [x.iloc[0], x.iloc[1], x.iloc[2], x.iloc[3]]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Avg. Birthweight')
plt.title("Birthweight Vs. Mother's Age (in quantiles) ")
# #Save the plot
# plt.savefig('A1LM_Team10SFMBANDD1_bwight&mage.png')
plt.show()


# creating a bar graph comparing avg. birthweight against cigs (in quantiles)
quant = birthweight['cigs'].quantile([0.25, 0.50, 0.75, 1])
for val in enumerate(birthweight.loc[ : , 'cigs']):
      if val[1] <= quant.iloc[0]:
            birthweight.loc[val[0], 'quant'] = 25
            
      elif val[1] <= quant.iloc[1]: 
            birthweight.loc[val[0], 'quant'] = 50
            
      elif val[1] <= quant.iloc[2]:
            birthweight.loc[val[0], 'quant'] = 75
            
      else:
            birthweight.loc[val[0], 'quant'] = 100
            
x = birthweight.groupby('quant')['bwght'].mean()
objects = ('Q1 ' + str(quant.iloc[0]), 'Q2 ' + str(quant.iloc[1]), 'Q3 ' + str(quant.iloc[2]), 'Q4 ' + str(quant.iloc[3]))
y_pos = np.arange(len(objects))
performance = [x.iloc[0], x.iloc[1], x.iloc[2], x.iloc[3]]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Avg. Birthweight')
plt.title("Birthweight Vs. Cigarettes (in quantiles) ")
# #Save the plot
# plt.savefig('A1LM_Team10SFMBANDD1_bwight&cigs.png')
plt.show()


# creating a bar graph comparing avg. birthweight against drink (in quantiles)
quant = birthweight['drink'].quantile([0.25, 0.50, 0.75, 1])
for val in enumerate(birthweight.loc[ : , 'drink']):
      if val[1] <= quant.iloc[0]:
            birthweight.loc[val[0], 'quant'] = 25
            
      elif val[1] <= quant.iloc[1]: 
            birthweight.loc[val[0], 'quant'] = 50
            
      elif val[1] <= quant.iloc[2]:
            birthweight.loc[val[0], 'quant'] = 75
            
      else:
            birthweight.loc[val[0], 'quant'] = 100
            
x = birthweight.groupby('quant')['bwght'].mean()
objects = ('Q1 ' + str(quant.iloc[0]), 'Q2 ' + str(quant.iloc[1]), 'Q3 ' + str(quant.iloc[2]), 'Q4 ' + str(quant.iloc[3]))
y_pos = np.arange(len(objects))
performance = [x.iloc[0], x.iloc[1], x.iloc[2], x.iloc[3]]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Avg. Birthweight')
plt.title("Birthweight Vs. Drinks (in quantiles) ")

# #Save the plot
# plt.savefig('A1LM_Team10SFMBANDD1_bwight&drink.png')
plt.show()