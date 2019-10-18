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
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split,RandomizedSearchCV
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


def lc_defaults_quick_eval(X_train,y_train,X_test,y_test,test_loan_data):
    logm = LogisticRegression(solver='liblinear')
    rfc = RandomForestClassifier(n_estimators=10)
    gbc = GradientBoostingClassifier()
    xgb = XGBClassifier()
    logm.fit(X_train,y_train)
    rfc.fit(X_train,y_train)
    gbc.fit(X_train,y_train)
    xgb.fit(X_train,y_train)
    y_preds_logm = logm.predict(X_test)
    y_preds_rfc = rfc.predict(X_test)
    y_preds_gbc = gbc.predict(X_test)
    y_preds_xgb = xgb.predict(X_test)
    print('Logistic Regression')
    logm_acc, logm_prec, logm_ret = lc_score(y_test,y_preds_logm,test_loan_data)
    print('\n')
    print('Random Forest')
    rfc_acc, rfc_prec, rfc_ret = lc_score(y_test,y_preds_rfc,test_loan_data)
    print('\n')
    print('Gradient Boosting')
    gbc_acc, gbc_prec, gbc_ret = lc_score(y_test,y_preds_gbc,test_loan_data)
    print('\n')
    print('XGradient Boosting')
    xgb_acc, xgb_prec, xgb_ret = lc_score(y_test,y_preds_xgb,test_loan_data)
    logm_stats = [round(logm_acc,2),round(logm_prec,2),round(logm_ret/100 + 1,3)]
    rfc_stats = [round(rfc_acc,2),round(rfc_prec,2),round(rfc_ret/100+1,3)]
    gbc_stats = [round(gbc_acc,2),round(gbc_prec,2),round(gbc_ret/100+1,3)]
    xgb_stats = [round(xgb_acc,2),round(xgb_prec,2),round(xgb_ret/100+1,3)]
    return logm_stats, rfc_stats, gbc_stats, xgb_stats

def lc_defaults_quick_eval(X_train,y_train,X_test,y_test,test_loan_data):
    logm = LogisticRegression(solver='liblinear')
    rfc = RandomForestClassifier(n_estimators=10)
    gbc = GradientBoostingClassifier()
    xgb = XGBClassifier()
    logm.fit(X_train,y_train)
    rfc.fit(X_train,y_train)
    gbc.fit(X_train,y_train)
    xgb.fit(X_train,y_train)
    y_preds_logm = logm.predict(X_test)
    y_preds_rfc = rfc.predict(X_test)
    y_preds_gbc = gbc.predict(X_test)
    y_preds_xgb = xgb.predict(X_test)
    print('Logistic Regression')
    logm_acc, logm_prec, logm_ret = lc_score(y_test,y_preds_logm,test_loan_data)
    print('\n')
    print('Random Forest')
    rfc_acc, rfc_prec, rfc_ret = lc_score(y_test,y_preds_rfc,test_loan_data)
    print('\n')
    print('Gradient Boosting')
    gbc_acc, gbc_prec, gbc_ret = lc_score(y_test,y_preds_gbc,test_loan_data)
    print('\n')
    print('XGradient Boosting')
    xgb_acc, xgb_prec, xgb_ret = lc_score(y_test,y_preds_xgb,test_loan_data)
    logm_stats = [round(logm_acc,2),round(logm_prec,2),round(logm_ret/100 + 1,3)]
    rfc_stats = [round(rfc_acc,2),round(rfc_prec,2),round(rfc_ret/100+1,3)]
    gbc_stats = [round(gbc_acc,2),round(gbc_prec,2),round(gbc_ret/100+1,3)]
    xgb_stats = [round(xgb_acc,2),round(xgb_prec,2),round(xgb_ret/100+1,3)]
    return logm_stats, rfc_stats, gbc_stats, xgb_stats


def lc_randomized_search(X_train,y_train,X_test, y_test,test_loan_data):
    logm = LogisticRegression(solver='liblinear')
    rfc = RandomForestClassifier(n_estimators=10)
    gbc = GradientBoostingClassifier()
    xgb = XGBClassifier()
    logm_params = {'penalty':['l1','l2'],'C':np.arange(1,10,1)}
    rfc_params = { 'max_depth': [3,5,10,15,20],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4,6,10],
     'min_samples_split': [2, 5, 7,10],
     'n_estimators': [10,20,100]}
    gbc_params = {'n_estimators':[10,50],'learning_rate':[0.1,0.01],'max_depth':[3,6]}
    xgb_params = {'n_estimators':[10,50],'learning_rate':[0.1,0.01],'max_depth':[3,6]}
    logm_random = RandomizedSearchCV(logm,logm_params,cv=3,n_iter=5,n_jobs=-1)

    rfc_random = RandomizedSearchCV(rfc,rfc_params,cv=3,n_iter=5,n_jobs=-1)

    gbc_random = RandomizedSearchCV(gbc,gbc_params,cv=3,n_iter=5,n_jobs=-1)

    xgb_random = RandomizedSearchCV(xgb,xgb_params,cv=3,n_iter=5,n_jobs=-1)

    logm_random.fit(X_train,y_train)
    print('Completed fit for Logistic Regression')
    rfc_random.fit(X_train,y_train)
    print('Completed fit for Random Forest')
    gbc_random.fit(X_train,y_train)
    print('Completed fit for Gradient Boosting')
    xgb_random.fit(X_train,y_train)
    print('Completed fit for XGradient Boosting')
    logm_y_preds = logm_random.predict(X_test)
    rfc_y_preds = rfc_random.predict(X_test)
    gbc_y_preds = gbc_random.predict(X_test)
    xgb_y_preds = xgb_random.predict(X_test)
    print('Logistic Regression')
    print('Best_Params: {}'.format(logm_random.best_params_))
    logm_stats = lc_score(y_test,logm_y_preds,test_loan_data)
    print('\n')
    print('Random Forest')
    print('Best Params: {}'.format(rfc_random.best_params_))
    rfc_stats = lc_score(y_test,rfc_y_preds,test_loan_data)
    print('\n')
    print('Gradient Boosting')
    print('Best Params: {}'.format(gbc_random.best_params_))
    gbc_stats = lc_score(y_test,gbc_y_preds,test_loan_data)
    print('\n')
    print('XGradientBoosting')
    print('Best Params: {}'.format(xgb_random.best_params_))
    xgb_stats = lc_score(y_test,xgb_y_preds,test_loan_data)
    return logm_stats, rfc_stats, gbc_stats, xgb_stats, logm_random, rfc_random, gbc_random, xgb_random



def lc_predict_probas_evaluator(model, y_test,loan_test_data):
    pass
