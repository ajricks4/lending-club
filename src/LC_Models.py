import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pdp
import matplotlib as mpl
mpl.rcParams['font.size'] = 15.0
import imp
plt.style.use('seaborn-darkgrid')
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRFClassifier,XGBClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score,precision_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from numpy.linalg import svd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
from os import listdir
import src.LC_Clean_Assist as LCC
import src.LC_Plotter as LCP
import src.LC_Transformer as LCT
import src.LC_Models as LCM
imp.reload(LCP)
imp.reload(LCC)
imp.reload(LCT)
imp.reload(LCM)
from sklearn.feature_selection import SelectKBest


def lc_evaluate_model(y_true,y_preds,df_extra):
    predictions = pd.DataFrame({'y_preds':y_preds,'y_true':y_true})
    evals = pd.concat([df_extra,predictions],axis=1)
    evals['investments'] = evals['y_preds'] * evals['loan_amount']
    evals['profit_loss'] = evals['y_preds'] * evals['loan_payoff']
    overall_return = round((evals['profit_loss'].sum() / evals['investments'].sum() - 1) * 100 ,1)
    return overall_return



def lc_mult_evaluate(scaled_df,model):

    model_df = scaled_df
    accs = []
    precs= []
    rets = []
    props = []
    for i in list(np.linspace(0.01, 3.0,25)):
        print('Testing with proportion {}'.format(i))
        print('\n')
        X_train, X_test, y_train, y_test, train_loan_data, test_loan_data = LCT.lc_balance_sets(model_df,i)
        model.fit(X_train,y_train)
        y_preds = model.predict(X_test)
        acc, prec, ret = lc_score(y_test,y_preds,test_loan_data)
        accs.append(acc)
        precs.append(prec)
        rets.append(ret)
        props.append(i)
    return pd.DataFrame({'Proportions':props,'Accuracy':accs,'Precision':precs,'Returns':rets})





"""
def grid_search(model, param_grid,X_train,y_train,cv=5):
    grid_search = GridSearchCV(model,param_grid,cv=cv)
    grid_search.fit(X_train,y_train)
    return grid_search.best_params_, grid_search.best_estimator_
"""

def lc_log_grid(X_train,y_train):
    pass


def lc_rfc_grid(X_train,y_train):
    pass


def lc_xgb_grid(X_train,y_train):
    pass


def lc_score(y_true,y_preds,test_loan_data):
    print(confusion_matrix(y_true,y_preds))
    prec = precision_score(y_true,y_preds)
    acc = accuracy_score(y_true,y_preds)
    overall_return = lc_evaluate_model(y_true,y_preds, test_loan_data)
    print('\n')
    print('{}: {}'.format('accuracy',acc))
    print('{}: {}'.format('precision',prec))
    print('Return: {}'.format(overall_return))
    return acc,prec,overall_return
