'''
Import Libraries and Expand DF View
'''
import math
import pandas as pd
import numpy as np
from operator import itemgetter


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


from sklearn import tree
from sklearn.tree import _tree

from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier 

from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

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
Exploring the data categorical and numerical data
'''

print( train_df.describe().T)
print("\n\n\n")

train_dt = train_df.dtypes

objList = []
intList = []
floatList = []
numList = []

for i in train_dt.index :
   if i in ( [ TARGET_F] ) : continue
   if train_dt[i] in (["object"]) : objList.append( i )
   if train_dt[i] in (["float64"]) : floatList.append( i )
   if train_dt[i] in (["int64"]) : intList.append( i )
   if train_dt[i] in (["float64","int64"]) : numList.append( i )

#Creating lists of interests
demographics = ["Pclass", "Age", "Sex"]
family = ["SibSp", "Parch"]
income = ["Fare", "Embarked"]

for i in demographics :
   print(" Class = ", i )
   g = train_df.groupby( i )
   x = g[ TARGET_F ].mean()
   print( "Survived Titanic", x )
  

for i in family :
   print("Variable=",i )
   g = train_df.groupby( TARGET_F )
   x = g[ i ].mean()
   print( "Survived Titanic Probab", x )
   

for i in floatList :
   print("Variable=",i )
   g = train_df.groupby( TARGET_F )
   x = g[ i ].mean()
   print( "Survived Titanic Probab", x )



'''
Exploring the data visually 
'''

# for i in demographics:
#    sns.histplot(
#     x=train_df[ i ], hue=train_df[ TARGET_F ],
#     multiple="stack",
#     palette="light:m_r",
#     edgecolor=".3",
#     linewidth=.5)
#    plt.xlabel( i )
#    plt.show()


# for i in income:
#    sns.histplot(
#     x=train_df[ i ], hue=train_df[ TARGET_F ],
#     multiple="stack",
#     palette="light:m_r",
#     edgecolor=".3",
#     linewidth=.5)
#    plt.xlabel( i )
#    plt.show()



'''
One Hot Encoding and Impute Categorical Value
'''

# for i in objList :
#    print( i )
#    print( train_df[i].unique() )
#    g = train_df.groupby( i )
#    print( g[i].count() )
#    print( "MOST COMMON = ", train_df[i].mode()[0] )   
#    print( "MISSING = ", train_df[i].isna().sum() ) 
#    print( "\n\n")

# Dealing with Missing Values
for i in numList :
    if train_df[i].isna().sum() == 0 : continue
    FLAG = "M_" + i
    IMP = "IMP_" + i
    train_df[ FLAG ] = train_df[i].isna() + 0
    train_df[ IMP ] = train_df[ i ]
    train_df.loc[ train_df[IMP].isna(), IMP ] = train_df[i].median()
    train_df = train_df.drop( i, axis=1 )

for i in objList :
    if train_df[i].isna().sum() == 0 : continue
    NAME = "IMP_"+i
    train_df[NAME] = train_df[i]
    train_df[NAME] = train_df[NAME].fillna("MISSING")
    train_df = train_df.drop( i, axis=1 )

print( train_df.describe().T)
print(train_df.head())

# One Hot Encoding

imputedList = ["Sex", "IMP_Embarked"]

for i in imputedList :
    thePrefix = "z_" + i
    y = pd.get_dummies( train_df[i], dtype = int, prefix=thePrefix, dummy_na=False)   
    train_df = pd.concat( [train_df, y], axis=1 )
    train_df = train_df.drop( i, axis=1 )
    
print( train_df.describe().T)
print(train_df.head())



"""
Remove Outliers
"""

print( train_df.describe().T)

dt = train_df.dtypes
numList = []
objList = []

for i in dt.index :
    print(i, dt[i])
    if i in ( [ TARGET_F ] ) : continue
    if dt[i] in (["object"]) : objList.append( i )
    if dt[i] in (["float64","int64"]) : numList.append( i )

for i in numList :
    theMean = train_df[i].mean()
    theSD = train_df[i].std()
    theMax = train_df[i].max()
    theCutoff = round( theMean + 3*theSD )
    if theMax < theCutoff : continue
    FLAG = "O_" + i
    TRUNC = "TRUNC_" + i
    train_df[ FLAG ] = ( train_df[i] > theCutoff )+ 0
    train_df[ TRUNC ] = train_df[ i ]
    train_df.loc[ train_df[TRUNC] > theCutoff, TRUNC ] = theCutoff
    train_df = train_df.drop( i, axis=1 )

for i in objList:
    train_df = train_df.drop( i, axis=1 )

print(train_df.dtypes)
print( train_df.describe().T)
print(train_df.head())



"""
SPLIT DATA
"""

X = train_df.copy()
X = X.drop( TARGET_F, axis=1 )
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

def getEnsembleTreeVars( ENSTREE, varNames ) :
    importance = ENSTREE.feature_importances_
    index = np.argsort(importance)
    theList = []
    for i in index :
        imp_val = importance[i]
        if imp_val > np.average( ENSTREE.feature_importances_ ) :
            v = int( imp_val / np.max( ENSTREE.feature_importances_ ) * 100 )
            theList.append( ( varNames[i], v ) )
    theList = sorted(theList,key=itemgetter(1),reverse=True)
    return theList

def getCoefLogit( MODEL, TRAIN_DATA ) :
    varNames = list( TRAIN_DATA.columns.values )
    coef_dict = {}
    coef_dict["INTERCEPT"] = MODEL.intercept_[0]
    for coef, feat in zip(MODEL.coef_[0],varNames):
        coef_dict[feat] = coef
    print("\SURVIVAL PROBABILITY")
    print("---------")
    print("Total Variables: ", len( coef_dict ) )
    for i in coef_dict :
        print( i, " = ", coef_dict[i]  )


"""
DECISION TREE
"""

WHO = "TREE"

CLM = tree.DecisionTreeClassifier( max_depth= 3, random_state=5 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

# print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

feature_cols = list( X.columns.values )
# tree.export_graphviz(CLM,out_file='tree_f.dot',filled=True, rounded=True, feature_names = feature_cols, impurity=False, class_names=["Good","Bad"]  )
vars_tree_flag = getTreeVars( CLM, feature_cols ) 

#to view tree, use the following command in terminal: dot -Tsvg tree_f.dot > output.svg

TREE_CLM = TEST_CLM.copy()


"""
RANDOM FOREST
"""

WHO = "RF"

CLM = RandomForestClassifier(max_depth=5, random_state=5)
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

# print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

feature_cols = list( X.columns.values )
vars_RF_flag = getEnsembleTreeVars( CLM, feature_cols )

RF_CLM = TEST_CLM.copy()


"""
GRADIENT BOOSTING
"""

WHO = "GB"

CLM = GradientBoostingClassifier(random_state=5)
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

# print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )


feature_cols = list( X.columns.values )
vars_GB_flag = getEnsembleTreeVars( CLM, feature_cols )

GB_CLM = TEST_CLM.copy()


"""
REGRESSION ALL VARIABLES
"""

WHO = "REG_ALL"

CLM = LogisticRegression( solver='newton-cg', max_iter=1000 )
CLM = CLM.fit( X_train, Y_train[ TARGET_F ] )

TRAIN_CLM = getProbAccuracyScores( WHO + "_Train", CLM, X_train, Y_train[ TARGET_F ] )
TEST_CLM = getProbAccuracyScores( WHO, CLM, X_test, Y_test[ TARGET_F ] )

# print_ROC_Curve( WHO, [ TRAIN_CLM, TEST_CLM ] ) 
print_Accuracy( WHO + " CLASSIFICATION ACCURACY", [ TRAIN_CLM, TEST_CLM ] )

REG_ALL_CLM = TEST_CLM.copy()


"""
COMPARE MODEL ACCURACIES
"""

ALL_CLM = [ TREE_CLM, RF_CLM, GB_CLM, REG_ALL_CLM]

ALL_CLM = sorted( ALL_CLM, key = lambda x: x[4], reverse=True )
print_ROC_Curve( WHO, ALL_CLM ) 

ALL_CLM = sorted( ALL_CLM, key = lambda x: x[1], reverse=True )
print_Accuracy( "ALL CLASSIFICATION ACCURACY", ALL_CLM )
