import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn as sns

from itertools import cycle
from sklearn.preprocessing import label_binarize, StandardScaler

from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFromModel
from sklearn.decomposition import PCA

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier,  VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,MultinomialNB

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

data = pd.read_csv('CleanData.csv',index_col=0)
y=data['rank3']
data = data.drop('Num_review_scale.1', axis=1)
X=data[[
       'Hong Kong Style', 'Japanese', 'Western', 'decor', 'service', 'overall', 'mins_walk', 'alipay', 'apple pay', 'openrice pay', 'wechat', 'Alcoholic Drinks', 'Delivery', '$101-200', '$201-400', '$51-100', 'Below $50', 'rank_foodtype', 'Central', 'Kowloon Bay', 'North Point', 'Tsim Sha Tsui', 'Tsuen Wan', 'Yau Ma Tei', 'Yuen Long', 'happy_scale', 'sad_scale', 'Num_review_scale', 'num_branch_scale', 'num_payment_scale', 'num_facility_scale', 'visa', '10% Service Charge', 'Num_review_scale', '$401-800'
             ]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

RF= RandomForestClassifier(random_state=42,max_depth= 11, min_samples_leaf=5, min_samples_split=10, n_estimators= 150)
RF.fit(X_train, y_train)

Bag1= BaggingClassifier(base_estimator=DecisionTreeClassifier(),max_samples=0.5, n_estimators=51)
Bag1.fit(X_train, y_train)

Bag2= BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=71)
Bag2.fit(X_train, y_train)

Boost=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),algorithm='SAMME', learning_rate=0.0001, n_estimators=1)
Boost.fit(X_train, y_train)

estimators=[ ('Bag2', Bag2),('Bag',Bag1),('Boost',Boost),('RF',RF)]

for i in range (6):
    max = 0
    ls = []
    for a in range(1,4):
        for b in range(1,4):
            for c in range(1,4):
                for d in range(1,4):
                    ensemble = VotingClassifier(estimators, voting='soft', weights=[a,b,c,d])
                    ensemble.fit(X_train, y_train)

                    ensemble_pred = ensemble.predict(X_test)
                    Accuracy_ensemble = accuracy_score (ensemble_pred,y_test)
                    if Accuracy_ensemble>=max:
                        max = Accuracy_ensemble
                        ls = [a,b,c,d]

    print(max)
    print(ls)
