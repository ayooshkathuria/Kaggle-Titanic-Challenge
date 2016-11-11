#My solution to Kaggle Titanic Machine Learning from Disaster Challenge
#We have to predict survival for passengers onboard rms Titanic using various features describing passengers

#The stuff that we are going to need. 
import matplotlib.pyplot as plt
import pandas as pd    #Dataframe library to manipulate data
import numpy as np      
import re  #We'll be using regular expressions to extract the titles from people's names. Like Mr, Mrs, Count etc
from sklearn.cross_validation import KFold   #for k-fold cross-validation
from sklearn import cross_validation as cv      #cross-validation
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import xgboost 
from sklearn.grid_search import GridSearchCV   #Support for Hyper-parameter Tuning


#Kaggle provides us with two files. train.csv for training our classifier and test.csv 
#for generating submissions. We read the data in two seperate pandas dataframes
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Make a copy of test for later use. 
test_orig = test[:]

#In order to avoid duplication of code owing to applying same operations to both dataframes
#we combine the test and train dataframes into one. We'll split them later at time of training.
seperator = train.shape[0] #get the length of training data to slie the combined data frame later
frames = [train, test]
titanic = pd.concat(frames)

#Cabin has too many missing values. Isn't very important to survial, so we drop it.
titanic.drop(["Cabin"], axis = 1, inplace = True)

#Senisble imputation of missing Fare values
median_fare = titanic.loc[(titanic["Pclass"] == 3) & (titanic["Embarked"] == "S") & (titanic["Age"] >= 55)].dropna()["Fare"].median()
titanic["Fare"] = titanic["Fare"].fillna(median_fare)

def get_title(name):
    """
    Use a regular expression to search for a title.  Titles always consist of
    capital and lowercase letters, and end with a period.
    
    Takes a name as input and returns the title string as output
    """

    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


titanic["Title"] = titanic["Name"].apply(get_title)  #We dropped "Name" earlier. So, we use original data.

#Condense the title into smaller, and more meaningful categories.
Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royal",
                        "Don":        "Royal",
                        "Sir" :       "Royal",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "Countess":   "Royal",
                        "Dona":       "Royal",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royal"

                        }

def titlemap(x):
    return Title_Dictionary[x]


titanic["Title"] = titanic["Title"].apply(titlemap)

#Fill in the missing age values by categorising the data and imputing the missing Age values 
#in a particular category by the median Age of that category.
#We could replace the age by the median but that would rob our dataset of precious variance 
#which is important in training our classifier to perform better.

def fillAges(row):
    if not(np.isnan(row['Age'])):
        return row['Age']
    
    if row['Sex']=='female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return 30
        elif row['Title'] == 'Mrs':
            return 45
        elif row['Title'] == 'Officer':
            return 49
        elif row['Title'] == 'Royalty':
            return 39

    elif row['Sex']=='female' and row['Pclass'] == 2:
        if row['Title'] == 'Miss':
            return 20
        elif row['Title'] == 'Mrs':
            return 30

    elif row['Sex']=='female' and row['Pclass'] == 3:
        if row['Title'] == 'Miss':
            return 18
        elif row['Title'] == 'Mrs':                
            return 31

    elif row['Sex']=='male' and row['Pclass'] == 1:
        if row['Title'] == 'Master':
            return 6
        elif row['Title'] == 'Mr':
            return 41.5
        elif row['Title'] == 'Officer':
            return 52
        elif row['Title'] == 'Royalty':
            return 40

    elif row['Sex']=='male' and row['Pclass'] == 2:
        if row['Title'] == 'Master':
            return 2
        elif row['Title'] == 'Mr':
            return 30
        elif row['Title'] == 'Officer':
                return 41.5

    elif row['Sex']=='male' and row['Pclass'] == 3:
        if row['Title'] == 'Master':
            return 6
        elif row['Title'] == 'Mr':
            return 26
        
titanic["Age"] = titanic.apply(fillAges, axis = 1)

#a Rare title indicates towards a higher chances of survival and hence, more chnaces or survival
#We denote Rare titles by 1 and The common ones by 0
def isRare(title):
    if title == "Mr" or title == "Mrs" or title == "Master" or title == "Miss":
        return 0
    return 1

titanic["Title"] = titanic["Title"].apply(isRare)

#Combing Siblings, Spouses, Parents or children onboard to a single Family variable
titanic["Family"] = titanic["Parch"] + titanic["SibSp"]

#Being a child improves your chances of survival. 
titanic["Child"] = 0
titanic.loc[titanic["Age"] <= 18, "Child"] = 1


#Sex is non-numeric data which can't be handled by our classifier. 
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0    #set male to 0 and female to 1
titanic.loc[titanic["Sex"] == "female", "Sex"] =1

titanic["Sex"] = titanic["Sex"].astype(int)


#Embarked is non-numeric data. Therefore we are going to build a couple of categorical variables to
#represent embarked
titanic["Q"] = 0
titanic.loc[titanic["Embarked"] == "Q", "Q"] = 1

titanic["S"] = 0
titanic.loc[titanic["Embarked"] == "S", "S"] = 1

# predictors = ["Age", "Q", "S", "Fare", "Pclass", "Sex", "Family", "Title", "Child"] 0.7655

#The predictors that we are going to use
predictors = ["Q", "S", "Fare", "Pclass", "Sex", "Family", "Title", "Child"]

#Break the combined data set into test and train data
target = titanic["Survived"].iloc[:seperator]
train = titanic[predictors][:seperator]
test = titanic[predictors][seperator:]


#Build an ensemble of classifiers. Hyper-parameters chosen through cross validation
xgb = xgboost.XGBClassifier(learning_rate = 0.05, n_estimators=500);
svmc = svm.SVC(C = 5, probability = True)

#fit the data
xgb.fit(train, target)
svmc.fit(train, target)

xgb_preds = xgb.predict_proba(test).transpose()[1]
svmc_preds = svmc.predict_proba(test).transpose()[1]

#Assign different weightages to the classifiers
ensemble_preds = xgb_preds*0.75 + svmc_preds*0.25

for x in range(len(ensemble_preds)):
    if ensemble_preds[x] >= 0.5:
        ensemble_preds[x] = 1
    else:
        ensemble_preds[x] = 0



results  = ensemble_preds.astype(int)

#Generate the final submission file.
submission = pd.DataFrame({"PassengerId": test_orig["PassengerId"], "Survived": results}) 
submission.to_csv("kaggle1.csv", index=False)



