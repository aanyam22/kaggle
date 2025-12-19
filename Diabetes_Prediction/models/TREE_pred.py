'''
Import Libraries and Expand DF View
'''
import math
import pandas as pd
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn import tree
from sklearn.tree import _tree

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

TARGET_F = "Survived"

TEST_FILE = "data/test.csv"
TARGET_FILE = "data/train.csv"

test_df = pd.read_csv( TEST_FILE )
train_df = pd.read_csv( TARGET_FILE )


'''
View the dataframe fields and datatypes
'''

print("Train DataFrame")
print(train_df.head().T)
print( train_df.dtypes )
print("\n\n\n")

print("Test DataFrame")
print( test_df.dtypes )
print( test_df.dtypes )
print("\n\n\n")

'''
Cleaning Train Dataframe
'''

train_dt = train_df.dtypes

train_objList = []
train_intList = []
train_floatList = []
train_numList = []

for i in train_dt.index :
   if i in ( [ TARGET_F] ) : continue
   if train_dt[i] in (["object"]) : train_objList.append( i )
   if train_dt[i] in (["float64"]) : train_floatList.append( i )
   if train_dt[i] in (["int64"]) : train_intList.append( i )
   if train_dt[i] in (["float64","int64"]) : train_numList.append( i )

# Dealing with Missing Values
for i in train_numList :
    if train_df[i].isna().sum() == 0 : continue
    FLAG = "M_" + i
    IMP = "IMP_" + i
    train_df[ FLAG ] = train_df[i].isna() + 0
    train_df[ IMP ] = train_df[ i ]
    train_df.loc[ train_df[IMP].isna(), IMP ] = train_df[i].median()
    train_df = train_df.drop( i, axis=1 )

print( train_df.describe().T)
print(train_df.head())

# One Hot Encoding
imputedList = ["Sex", "Embarked"]

for i in imputedList :
    thePrefix = "z_" + i
    y = pd.get_dummies( train_df[i], dtype = int, prefix=thePrefix, dummy_na=False)   
    train_df = pd.concat( [train_df, y], axis=1 )
    train_df = train_df.drop( i, axis=1 )
    
print( train_df.describe().T)
print(train_df.head())

#Dropping Remaining Object Types
train_dt = train_df.dtypes
train_numList = []
train_objList = []

for i in train_dt.index :
    if i in ( [ TARGET_F ] ) : continue
    if train_dt[i] in (["object"]) : train_objList.append( i )
    if train_dt[i] in (["float64","int64"]) : train_numList.append( i )

for i in train_objList:
    train_df = train_df.drop( i, axis=1 )

train_df = train_df.drop( ["M_Age"], axis=1 )

print(train_df.dtypes)
print( train_df.describe().T)
print(train_df.head())


# Outlier Truncation - Disabled

# for i in numList :
#     theMean = train_df[i].mean()
#     theSD = train_df[i].std()
#     theMax = train_df[i].max()
#     theCutoff = round( theMean + 3*theSD )
#     if theMax < theCutoff : continue
#     FLAG = "O_" + i
#     TRUNC = "TRUNC_" + i
#     train_df[ FLAG ] = ( train_df[i] > theCutoff )+ 0
#     train_df[ TRUNC ] = train_df[ i ]
#     train_df.loc[ train_df[TRUNC] > theCutoff, TRUNC ] = theCutoff
#     train_df = train_df.drop( i, axis=1 )


'''
Cleaning Test Dataframe
'''

test_dt = test_df.dtypes

test_objList = []
test_intList = []
test_floatList = []
test_numList = []

for i in test_dt.index :
   if test_dt[i] in (["object"]) : test_objList.append( i )
   if test_dt[i] in (["float64"]) : test_floatList.append( i )
   if test_dt[i] in (["int64"]) : test_intList.append( i )
   if test_dt[i] in (["float64","int64"]) : test_numList.append( i )

# Dealing with Missing Values
for i in test_numList :
    if test_df[i].isna().sum() == 0 : continue
    FLAG = "M_" + i
    IMP = "IMP_" + i
    test_df[ FLAG ] = test_df[i].isna() + 0
    test_df[ IMP ] = test_df[ i ]
    test_df.loc[ test_df[IMP].isna(), IMP ] = test_df[i].median()
    test_df = test_df.drop( i, axis=1 )

print( test_df.describe().T)
print(test_df.head())


# One Hot Encoding
imputedList2 = ["Sex", "Embarked"]

for i in imputedList2 :
    thePrefix = "z_" + i
    y = pd.get_dummies( test_df[i], dtype = int, prefix=thePrefix, dummy_na=False)   
    test_df = pd.concat( [test_df, y], axis=1 )
    test_df = test_df.drop( i, axis=1 )

print( test_df.describe().T)
print(test_df.head())

#Dropping Remaining Object Types
test_dt = test_df.dtypes
test_numList = []
test_objList = []

for i in test_dt.index :
    if test_dt[i] in (["object"]) : test_objList.append( i )
    if test_dt[i] in (["float64","int64"]) : test_numList.append( i )

for i in test_objList:
    test_df = test_df.drop( i, axis=1 )
    
test_df = test_df.drop( ["M_Age"], axis=1 )
test_df = test_df.drop( ["M_Fare"], axis=1 )

print(test_df.dtypes)
print( test_df.describe().T)
print(test_df.head())


"""
SPLIT DATA FOR TRAINING
"""
features = ["z_Sex_male", "z_Sex_female", "Pclass", "Fare", "IMP_Age"]

X = train_df[features].copy()
Y = train_df[ [TARGET_F] ]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=5)


"""
MODEL ACCURACY METRICS
"""

def getProbAccuracyScores( NAME, MODEL, X, Y ) :
    pred = MODEL.predict( X )
    probs = MODEL.predict_proba( X )
    acc_score = metrics.accuracy_score(Y, pred)
    p1 = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve( Y, p1)
    auc = metrics.auc(fpr,tpr)
    return [NAME, acc_score, fpr, tpr, auc]

def print_ROC_Curve( TITLE, LIST ) :
    fig = plt.figure(figsize=(6,4))
    plt.title( TITLE )
    for theResults in LIST :
        NAME = theResults[0]
        fpr = theResults[2]
        tpr = theResults[3]
        auc = theResults[4]
        theLabel = "AUC " + NAME + ' %0.2f' % auc
        plt.plot(fpr, tpr, label = theLabel )
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def print_Accuracy( TITLE, LIST ) :
    print( TITLE )
    print( "======" )
    for theResults in LIST :
        NAME = theResults[0]
        ACC = theResults[1]
        print( NAME, " = ", ACC )
    print( "------\n\n" )


def getTreeVars( TREE, varNames ) :
    tree_ = TREE.tree_
    varName = [ varNames[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature ]
    nameSet = set()
    for i in tree_.feature :
        if i != _tree.TREE_UNDEFINED :
            nameSet.add( i )
    nameList = list( nameSet )
    parameter_list = list()
    for i in nameList :
        parameter_list.append( varNames[i] )
    return parameter_list


"""
SINGLE TREE CLASSIFIER MODEL
"""

WHO = "TREE"

CLM = tree.DecisionTreeClassifier( max_depth= 3, random_state=5 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

feature_cols = list( X.columns.values )
# tree.export_graphviz(CLM,out_file='tree_f.dot',filled=True, rounded=True, feature_names = feature_cols, impurity=False, class_names=["Good","Bad"]  )
vars_tree_flag = getTreeVars( CLM, feature_cols ) 

#to view tree, use the following command in terminal: dot -Tsvg tree_f.dot > output.svg

TREE_CLM = TEST_CLM.copy()



"""
Final Prediction on Test Data
"""
test_df = test_df.rename( columns={ "IMP_Fare" : "Fare" } )
test_data = test_df[features].copy()

test_predictions = CLM.predict(test_data)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': test_predictions})
output.to_csv('submissions/TREE_submission.csv', index=False)