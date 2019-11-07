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
    evals['investments'] = round(evals['y_preds'] * evals['loan_amount'],3)
    evals['profit_loss'] = round(evals['y_preds'] * evals['loan_payoff'],3)
    evals = evals.groupby('term').sum()
    overall_return = round((evals['profit_loss'].sum() / evals['investments'].sum() - 1) * 100 ,1)
    deployed_capital = evals['investments'].sum()
    returned_capital = evals['profit_loss'].sum()
    t_36_pl = evals.loc[36,'profit_loss']
    t_36_deployed = evals.loc[36,'investments']
    t_36_rets = round((t_36_pl / t_36_deployed - 1) * 100,2)
    t_60_pl = evals.loc[60,'profit_loss']
    t_60_deployed = evals.loc[60,'investments']
    t_60_rets = round((t_60_pl / t_60_deployed - 1) * 100,2)

    return overall_return, deployed_capital, returned_capital, t_36_rets,t_36_deployed,t_36_pl,t_60_rets,t_60_deployed,t_60_pl



def lc_mult_evaluate(scaled_df,model):

    model_df = scaled_df
    accs = []
    precs= []
    rets = []
    props = []
    for i in list(np.linspace(0.01, 1.5,15)):
        print('\n')
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

def lc_proportion_grid_search(scaled_df,grid_model,rang=list(np.arange(0.025,0.51,0.025))):
    model = grid_model
    model_df = scaled_df
    accs = []
    precs = []
    rets = []
    props = []
    t_36_returns = []
    t_36_deployed_capital = []
    t_36_returned_capital = []
    t_60_returns = []
    t_60_deployed_capital = []
    t_60_returned_capital = []
    optimized_parameters = []
    deployed = []
    returned = []
    profits = []
    for i in rang:
        X_train, X_test, y_train, y_test, train_loan_data, test_loan_data = LCT.lc_balance_sets(model_df,i)
        model.fit(X_train,y_train)
        y_preds = model.best_estimator_.predict(X_test)
        prec = precision_score(y_test,y_preds)
        acc = accuracy_score(y_test,y_preds)
        overall_return, deployed_capital, returned_capital, t_36_rets,t_36_deployed,t_36_pl,t_60_rets,t_60_deployed,t_60_pl = lc_evaluate_model(y_test,y_preds,test_loan_data)
        profit = returned_capital - deployed_capital
        print('Proportion: {}'.format(i))
        print('\n')
        print('Best Parameters: {}'.format(model.best_params_))
        print('Returns: {}'.format(overall_return))
        print('Profit: {}'.format(round(profit,2)))
        print('Accuracy: {}'.format(acc))
        print('Precision: {}'.format(prec))
        print('Deployed Capital: {}'.format(deployed_capital))
        print('Returned Capital: {}'.format(returned_capital))
        print('36 Month Returns: {}, 36 Month Deployed Capital: {} 36 Month Returned Capital: {}'.format(t_36_rets,round(t_36_deployed,1),round(t_36_pl,1)))
        print('60 Month Returns: {}, 60 Month Deployed Capital: {} 60 Month Returned Capital: {}'.format(t_60_rets,round(t_60_deployed,1),round(t_60_pl,1)))
        print(confusion_matrix(y_test,y_preds))
        print('---------------------------------------')
        print('\n')
        accs.append(acc)
        precs.append(prec)
        rets.append(overall_return)
        props.append(i)
        optimized_parameters.append(model.best_params_)
        deployed.append(deployed_capital)
        returned.append(returned_capital)
        t_36_returns.append(t_36_rets)
        t_36_deployed_capital.append(t_36_deployed)
        t_36_returned_capital.append(t_36_pl)
        t_60_returns.append(t_60_rets)
        t_60_deployed_capital.append(t_60_deployed)
        t_60_returned_capital.append(t_60_pl)
        profits.append(profit)
    return pd.DataFrame({'Proportions':props,'Returns':rets,'Profits':profits,'Accuracy':accs,'Precision':precs,'Best_Params':optimized_parameters,'Deployed_Capital':deployed,'Returned_Capital':returned,'36 Month Returns':t_36_returns,'36 Month Deployed Capital':t_36_deployed_capital,'36 Month Returned Capital':t_36_returned_capital,'60 Month Returns':t_60_returns,'60 Month Deployed Capital':t_60_deployed_capital,'60 Month Returned Capital':t_60_returned_capital })








def lc_score(y_true,y_preds,test_loan_data):
    print(confusion_matrix(y_true,y_preds))
    prec = precision_score(y_true,y_preds)
    acc = accuracy_score(y_true,y_preds)
    overall_return, deployed_capital, returned_capital, t_36_rets,t_36_deployed,t_36_pl,t_60_rets,t_60_deployed,t_60_pl = lc_evaluate_model(y_true,y_preds, test_loan_data)
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


def lc_randomized_search(X_train,y_train,X_test, y_test,test_loan_data,scoring):
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
    logm_random = RandomizedSearchCV(logm,logm_params,cv=3,n_iter=5,n_jobs=-1,scoring=scoring)

    rfc_random = RandomizedSearchCV(rfc,rfc_params,cv=3,n_iter=5,n_jobs=-1,scoring=scoring)

    gbc_random = RandomizedSearchCV(gbc,gbc_params,cv=3,n_iter=5,n_jobs=-1,scoring=scoring)

    xgb_random = RandomizedSearchCV(xgb,xgb_params,cv=3,n_iter=5,n_jobs=-1,scoring=scoring)

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



def lc_predict_probas_evaluator(model, X_test, y_test,loan_test_data):
    probs_class_1 = list(model.predict_proba(X_test)[:,1])
    predictions = pd.DataFrame({'Class1':probs_class_1,'y_true':y_test})
    evals = pd.concat([loan_test_data,predictions],axis=1)
    rets = []
    for i in list(np.linspace(0.1,0.7,100)):
        evals['threshold_'+str(i)] = evals['Class1'].apply(lambda x: 1 if x > i else 0)
        ret = (np.sum(evals['threshold_'+str(i)] * evals['loan_payoff']) / np.sum((evals['loan_amount'] * evals['threshold_'+str(i)])) - 1)
        rets.append(round(ret,3))
    return list(np.linspace(0.1,0.7,100)),rets
