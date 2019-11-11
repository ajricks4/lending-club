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
import ast


def lc_evaluate_model(y_true,y_preds,df_extra):
    """
    Evaluates a model by comparing the model's predictions to the actual loan outcomes and calculating the returns.

    Args:
        y_true (pandas Series): actual loan outcomes.
        y_preds (pandas Series): predicted loan outcomes.
        df_extra (pandas dataframe): dataframe including the loan term, loan amount and loan payoff.

    Returns:
        overall_return (float): Overall Return percentage
        deployed_capital (float): Dollar amount invested in loans.
        returned_capital (float): Returned Capital.
        t_36_rets (float): Return percentage on 36 month term loans.
        t_36_deployed (float): Capital invested in 36 month term loans
        t_36_pl (float): Profit on 36 month term loans.
        t_60_rets (float): Return percentage on 60 month term loans.
        t_60_deployed (float): Capital invested in 60 month term loans.
        t_60_pl (float): Profit on 60 month term loans.
    """
    predictions = pd.DataFrame({'y_preds':y_preds,'y_true':y_true})
    evals = pd.concat([df_extra,predictions],axis=1)
    evals['investments'] = round(evals['y_preds'] * evals['loan_amount'],3)
    evals['profit_loss'] = round(evals['y_preds'] * evals['loan_payoff'],3)
    evals = evals.groupby('term').sum()
    overall_return = round((evals['profit_loss'].sum() / evals['investments'].sum() - 1) * 100 ,4)
    deployed_capital = evals['investments'].sum()
    returned_capital = evals['profit_loss'].sum()
    t_36_pl = evals.loc[36,'profit_loss']
    t_36_deployed = evals.loc[36,'investments']
    t_36_rets = round((t_36_pl / t_36_deployed - 1) * 100,4)
    t_60_pl = evals.loc[60,'profit_loss']
    t_60_deployed = evals.loc[60,'investments']
    t_60_rets = round((t_60_pl / t_60_deployed - 1) * 100,4)

    return overall_return, deployed_capital, returned_capital, t_36_rets,t_36_deployed,t_36_pl,t_60_rets,t_60_deployed,t_60_pl


def lc_proportion_grid_search(scaled_df,grid_model,rang=list(np.arange(0.025,0.51,0.025))):
    """
    Performs a gridsearch on each proportion of the majority : minority class specified in the rang variable.

    Args:
        scaled_df (pandas dataframe):
        grid_model (sklearn model):
        rang (list):

    Returns:
        data_table (pandas dataframe): DataFrame with detailed information on the gridsearch at each proportion.
    """
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
        print('Profit: {}'.format(round(profit,3)))
        print('Accuracy: {}'.format(acc))
        print('Precision: {}'.format(prec))
        print('Deployed Capital: {}'.format(deployed_capital))
        print('Returned Capital: {}'.format(returned_capital))
        print('36 Month Returns: {}, 36 Month Deployed Capital: {} 36 Month Returned Capital: {}'.format(t_36_rets,round(t_36_deployed,2),round(t_36_pl,2)))
        print('60 Month Returns: {}, 60 Month Deployed Capital: {} 60 Month Returned Capital: {}'.format(t_60_rets,round(t_60_deployed,2),round(t_60_pl,2)))
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
    data_table =  pd.DataFrame({'Proportions':props,'Returns':rets,'Profits':profits,'Accuracy':accs,'Precision':precs,'Best_Params':optimized_parameters,'Deployed_Capital':deployed,'Returned_Capital':returned,'36 Month Returns':t_36_returns,'36 Month Deployed Capital':t_36_deployed_capital,'36 Month Returned Capital':t_36_returned_capital,'60 Month Returns':t_60_returns,'60 Month Deployed Capital':t_60_deployed_capital,'60 Month Returned Capital':t_60_returned_capital })
    return data_table

def calculate_sharpe_ratios(pca_df,sharpe_matrix,rfo=0.0,rf36 = 0.0,rf60=0.0):
    """
    Calculates the data necessary to compute the sharpe ratios by re-running a gridsearched model 50 times to obtain
    the robustness of the model's predictive ability.

    Args:
        pca_df (pandas DataFrame): dataframe obtained from cleaning, scaling and applying PCA to the Lending Club dataset.
        sharpe_matrix (pandas DataFrame): dataframe containing the best models to obtain the sharpe ratios for.
        rfo (float): risk free return.
        rf36 (float): risk free return on 36 month assets.
        rf60 (float): risk free return on 60 month assets.

    Returns:
        pd.DataFrame(dat) (pandas dataframe): dataframe with data necessary to calculate Sharpe ratios.
    """
    df = sharpe_matrix
    model_list = []
    params = []
    proportions = []
    ret_list = []
    ret_list_36 = []
    ret_list_60 = []
    deployed_cap = []
    deployed_cap_36 = []
    deployed_cap_60 = []
    for i, j in zip(['best','36m','60m'] * 3,['LogisticRegression'] * 3 + ['RandomForestClassifier'] * 3 + ['GradientBoostingClassifier']*3):
        model_list.append(j + '___' + i)
        param_dict = ast.literal_eval(df[(df['measure'] ==i) & (df['model']==j)]['Best_Params'].values[0])
        p = df[(df['measure'] ==i) & (df['model']==j)]['Proportions'].values[0]
        params.append(param_dict)
        proportions.append(p)
        rets = []
        rets_36 = []
        rets_60 = []
        d_ = []
        d_36 = []
        d_60 = []
        for k in range(50):
            print('Running iteration: {}'.format(k+1))
            X_train, X_test, y_train, y_test, train_loan_data, test_loan_data = LCT.lc_balance_sets(pca_df,p)
            if j == 'LogisticRegression':
                mod = LogisticRegression(**param_dict)
            elif j =='RandomForestClassifier':
                mod = RandomForestClassifier(**param_dict)
            else:
                mod = GradientBoostingClassifier(**param_dict)
            mod.fit(X_train,y_train)
            y_preds = mod.predict(X_test)
            overall_return, deployed_capital, returned_capital, t_36_rets,t_36_deployed,t_36_pl,t_60_rets,t_60_deployed,t_60_pl = lc_evaluate_model(y_test,y_preds,test_loan_data)
            rets.append(overall_return)
            rets_36.append(t_36_rets)
            rets_60.append(t_60_rets)
            d_.append(deployed_capital)
            d_36.append(t_36_deployed)
            d_60.append(t_60_deployed)
        ret_list.append(rets)
        ret_list_36.append(rets_36)
        ret_list_60.append(rets_60)
        deployed_cap.append(d_)
        deployed_cap_36.append(d_36)
        deployed_cap_60.append(d_60)
        print('\n')
        print('Completed batch: {}'.format(j+ '___'+i))
        print('-------------------------------------')
    dat = {'Model_list':model_list,'Proportions':proportions,'Parameters':params,'Return_List':ret_list,
            'Return36_List':ret_list_36,'Return60_List':ret_list_60,'Deployed_Capital':deployed_cap,
            'Deployed_Capital_36':deployed_cap_36,'Deployed_Capital_60':deployed_cap_60}
    return pd.DataFrame(dat)

def sharpe_models_large_p(pca_df,combined_df,rfo=0.0,rf36 = 0.0,rf60=0.0):
    """
    Calculates the data necessary to compute the sharpe ratios by re-running a gridsearched model 50 times to obtain
    the robustness of the model's predictive ability.

    Args:
        pca_df (pandas DataFrame): dataframe obtained from cleaning, scaling and applying PCA to the Lending Club dataset.
        combined_df (pandas DataFrame): dataframe containing all models that the grid search ran on.
        rfo (float): risk free return.
        rf36 (float): risk free return on 36 month assets.
        rf60 (float): risk free return on 60 month assets.

    Returns:
        pd.DataFrame(dat) (pandas dataframe): dataframe with data necessary to calculate Sharpe ratios.
    """
    mods = ['LogisticRegression']*3 + ['RandomForestClassifier']*3 + ['GradientBoostingClassifier']*3
    rands = []
    props = []
    while len(rands) < len(mods):
        x = np.random.rand()
        if x < 0.2:
            continue
        else:
            rands.append(x)
    model_list = []
    params = []
    proportions = []
    ret_list = []
    ret_list_36 = []
    ret_list_60 = []
    deployed_cap = []
    deployed_cap_36 = []
    deployed_cap_60 = []
    for m,r in zip(mods,rands):
        model_list.append(m + '___' + str(r))
        data_row = combined_df[(combined_df['model'] == m) & (combined_df['Proportions'] - r < 0.01)].tail(1)
        p = combined_df['Proportions'].values[0]
        param_dict = ast.literal_eval(data_row['Best_Params'].values[0])
        params.append(param_dict)
        proportions.append(p)
        rets = []
        rets_36 = []
        rets_60 = []
        d_ = []
        d_36 = []
        d_60 = []
        for k in range(50):
            print('Running iteration: {}'.format(k+1))
            if m == 'LogisticRegression':
                mod = LogisticRegression(**param_dict)
            elif m == 'RandomForestClassifier':
                mod = RandomForestClassifier(**param_dict)
            else:
                mod = GradientBoostingClassifier(**param_dict)
            X_train, X_test, y_train, y_test, train_loan_data, test_loan_data = LCT.lc_balance_sets(pca_df,p)
            mod.fit(X_train,y_train)
            y_preds = mod.predict(X_test)
            overall_return, deployed_capital, returned_capital, t_36_rets,t_36_deployed,t_36_pl,t_60_rets,t_60_deployed,t_60_pl = lc_evaluate_model(y_test,y_preds,test_loan_data)
            rets.append(overall_return)
            rets.append(overall_return)
            rets_36.append(t_36_rets)
            rets_60.append(t_60_rets)
            d_.append(deployed_capital)
            d_36.append(t_36_deployed)
            d_60.append(t_60_deployed)
        ret_list.append(rets)
        ret_list_36.append(rets_36)
        ret_list_60.append(rets_60)
        deployed_cap.append(d_)
        deployed_cap_36.append(d_36)
        deployed_cap_60.append(d_60)
        print('\n')
        print('Completed batch: {}'.format(m+ '___'+str(r)))
        print('-------------------------------------')
    dat = {'Model_list':model_list,'Proportions':proportions,'Parameters':params,'Return_List':ret_list,
            'Return36_List':ret_list_36,'Return60_List':ret_list_60,'Deployed_Capital':deployed_cap,
            'Deployed_Capital_36':deployed_cap_36,'Deployed_Capital_60':deployed_cap_60}
    return pd.DataFrame(dat)


def str_list_to_list(x):
    """
    Converts json string containing list to list.

    Args:
        x (string): json string.

    Returns:
        arr (numpy array): array of original values.
    """
    k = [float(j) for j in x.strip('[]').split(',')]
    arr = np.array(k)
    arr =  arr[(~np.isnan(arr)) & (arr != 0 )]
    return arr

def sharpe_calc(rets,rf):
    """
    Calculates the sharpe ratio.

    Args:
        rets (arr): array with returns data
        rf (float): risk free asset return.

    Returns:
        sharpe ratio (float): sharpe ratio defined as (mean(returns) - risk_free_return) / standard_deviation(returns)
    """
    if len(rets) == 0:
        return 0
    else:
        return np.mean(rets) / np.std(rets)


def sharpe_calc_df(df_orig):
    """
    Modifies the dataframe to clean up the sharpe matrix data to workable data as well as compute Sharpe Ratios.

    Args:
        df_orig (pandas DataFrame): DataFrame produced from running either the calculate_sharpe_ratios or sharpe_models_large_p functions.

    Returns:
        df (pandas DataFrame):
    """
    df = df_orig
    df['Return_List'] = df['Return_List'].apply(lambda x: str_list_to_list(x))
    df['Return36_List'] = df['Return36_List'].apply(lambda x: str_list_to_list(x))
    df['Return60_List'] = df['Return60_List'].apply(lambda x: str_list_to_list(x))
    df['Avg_Return'] = df['Return_List'].apply(lambda x: 0 if len(x) ==0 else np.mean(x))
    df['Avg_Return_36'] = df['Return36_List'].apply(lambda x: 0 if len(x) ==0 else np.mean(x))
    df['Avg_Return_60'] = df['Return60_List'].apply(lambda x: 0 if len(x) ==0 else np.mean(x))
    df['Sharpe_Overall'] = df['Return_List'].apply(lambda x: 0 if len(x) < 2 else sharpe_calc(x,0.0))
    df['Sharpe_36'] = df['Return36_List'].apply(lambda x: 0 if len(x) < 2 else sharpe_calc(x,0.0))
    df['Sharpe_60'] = df['Return60_List'].apply(lambda x: 0 if len(x) < 2 else sharpe_calc(x,0.0))
    df['Deployed_Capital'] = df['Deployed_Capital'].apply(lambda x: str_list_to_list(x))
    df['Deployed_Capital_36'] = df['Deployed_Capital_36'].apply(lambda x: str_list_to_list(x))
    df['Deployed_Capital_60'] = df['Deployed_Capital_60'].apply(lambda x: str_list_to_list(x))
    return df


def final_system(pca_df,model_36m,model_60m,p_36,p_60):
    """
    Creates final prediction system based on two models.

    Args:
        pca_df (pandas DataFrame): dataframe containing scaled and cleaned data.
        model_36m (sklearn model): model to be used for predictions on 36 month term loans.
        model_60m (sklearn model): model to be used for predictions on 60 month term loans.
        p_36 (float): proportion for model_36m.
        p_60 (float): proportion for model_60m.

    Returns:
        overall_return (float): Overall Return percentage
        deployed_capital (float): Dollar amount invested in loans.
        returned_capital (float): Returned Capital.
        t_36_rets (float): Return percentage on 36 month term loans.
        t_36_deployed (float): Capital invested in 36 month term loans
        t_36_pl (float): Profit on 36 month term loans.
        t_60_rets (float): Return percentage on 60 month term loans.
        t_60_deployed (float): Capital invested in 60 month term loans.
        t_60_pl (float): Profit on 60 month term loans.
    """
    X_train, X_test, y_train, y_test = train_test_split(pca_df.iloc[:,:-1], pca_df.iloc[:,-1], test_size=0.1)
    train_set = pd.concat([X_train,y_train],axis=1)
    p1 = p_36
    p2 = p_60
    class_0 = train_set[train_set['loan_status'] == 0]
    size = int(p1 * class_0.shape[0])
    class_1 = train_set[train_set['loan_status'] == 1].iloc[:size,:]
    train_set = pd.concat([class_0,class_1])
    X_train_36m = train_set.iloc[:,:-1]
    y_train_36m = train_set.iloc[:,-1]
    class_0 = train_set[train_set['loan_status'] == 0]
    size = int(p2 * class_0.shape[0])
    class_1 = train_set[train_set['loan_status'] == 1].iloc[:size,:]
    train_set = pd.concat([class_0,class_1])
    X_train_60m = train_set.iloc[:,:-1]
    y_train_60m = train_set.iloc[:,-1]
    test_loan_data = X_test[['term','loan_amount','loan_payoff']]
    X_train_36m.drop(['term','loan_amount','loan_payoff'],axis=1,inplace=True)
    X_train_60m.drop(['term','loan_amount','loan_payoff'],axis=1,inplace=True)
    model_36m.fit(X_train_36m,y_train_36m)
    model_60m.fit(X_train_60m,y_train_60m)
    X_test.drop(['term','loan_amount','loan_payoff'],axis=1,inplace=True)
    preds = pd.DataFrame({'36M_Preds':model_36m.predict(X_test),'60M_Preds':model_60m.predict(X_test),'Actual':y_test})
    combined_preds = pd.concat([preds,test_loan_data],axis=1)
    combined_preds['Y_preds'] = combined_preds.apply(lambda row: row['36M_Preds'] if row['term'] == 36 else row['60M_Preds'],axis=1)
    overall_return, deployed_capital, returned_capital, t_36_rets,t_36_deployed,t_36_pl,t_60_rets,t_60_deployed,t_60_pl = LCM.lc_evaluate_model(combined_preds['Y_preds'].values,y_test,test_loan_data)
    return overall_return, deployed_capital, returned_capital, t_36_rets,t_36_deployed,t_36_pl,t_60_rets,t_60_deployed,t_60_pl
