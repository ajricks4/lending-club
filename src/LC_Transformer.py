from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
import pandas as pd
from sklearn.decomposition import PCA, NMF

def lc_transform(df_original):
    df = df_original.copy()
    num_features = ['issue_d_year','funded_amnt','int_rate','installment',
    'FICO','DTI','annual_income','emp_length',
    'negative_activity','inq_last_6mths','delinq_2yrs','open_accts',
    'mortgage_accts','tot_cur_bal','revolving','revol_util']

    cat_features = ['grade','term','purpose','application_type',
    'home_ownership','verified']
    sc = StandardScaler()
    X1 = sc.fit_transform(df[num_features])
    ohe = OneHotEncoder()
    X2 = ohe.fit_transform(df[cat_features])
    categories = []
    for i in ohe.categories_:
        categories += list(i)
    test_cat = pd.DataFrame(X2.todense(),columns=categories)
    test_num = pd.DataFrame(X1,columns=num_features)
    output = df[['term','funded_amnt','total_pymnt','loan_status',]]
    output[['loan_amount','loan_payoff']] = output[['funded_amnt','total_pymnt']]
    output = output[['term','loan_amount','loan_payoff','loan_status']]
    scaled_df = pd.concat([test_num,test_cat,output],axis=1)
    return scaled_df

def lc_stratified_train_test_split(data,data_x, data_y, state=101, size = 0.2, splits=1):
    split = StratifiedShuffleSplit(n_splits=splits,test_size=size,random_state=state)
    for i, j in split.split(data_x,data_y):
        strat_train_set = data.iloc[i]
        strat_test_set = data.iloc[j]

    return strat_train_set, strat_test_set

def lc_create_sets(scaled_df):
    x_cols = list(scaled_df.columns)
    x_cols.remove('loan_status')
    x_cols.remove('loan_amount')
    x_cols.remove('loan_payoff')
    strat_train, strat_valid = lc_stratified_train_test_split(scaled_df,scaled_df[x_cols],scaled_df['loan_status'])
    strat_train, strat_test = lc_stratified_train_test_split(strat_train,strat_train[x_cols],strat_train['loan_status'])
    return strat_train, strat_test, strat_valid, x_cols

def lc_organize(scaled_df):
    strat_train, strat_test, strat_valid, x_cols = lc_create_sets(scaled_df)
    strat_train_x = strat_train[x_cols]
    strat_train_y = strat_train['loan_status']
    strat_train_extra = strat_train[['loan_amount','loan_payoff']]
    strat_test_x = strat_test[x_cols]
    strat_test_y = strat_test['loan_status']
    strat_test_extra = strat_test[['loan_amount','loan_payoff']]
    strat_valid_x = strat_valid[x_cols]
    strat_valid_y = strat_valid['loan_status']
    strat_valid_extra = strat_valid[['loan_amount','loan_payoff']]
    return strat_train_x, strat_train_y, strat_train_extra, strat_test_x, strat_test_y, strat_test_extra, strat_valid_x,strat_valid_y, strat_valid_extra


def lc_balance_sets(scaled_df,proportion = 1):
    X_train, X_test, y_train, y_test = train_test_split(scaled_df.iloc[:,:-1], scaled_df.iloc[:,-1], test_size=0.1, random_state=42)
    train_set = pd.concat([X_train,y_train],axis=1)



    class_0 = train_set[train_set['loan_status'] == 0]
    size = int(proportion * class_0.shape[0])
    class_1 = train_set[train_set['loan_status'] == 1].iloc[:size,:]
    train_set = pd.concat([class_0,class_1])
    X_train = train_set.iloc[:,:-1]
    y_train = train_set.iloc[:,-1]
    train_loan_data = X_train[['term','loan_amount','loan_payoff']]
    X_train.drop(['term','loan_amount','loan_payoff'],axis=1,inplace=True)
    test_loan_data = X_test[['term','loan_amount','loan_payoff']]
    X_test.drop(['term','loan_amount','loan_payoff'],axis=1,inplace=True)
    return X_train, X_test, y_train, y_test, train_loan_data, test_loan_data

def get_pca_df(scaled_df,n_comp):
    pca = PCA(n_components=n_comp)
    principalComponents = pca.fit_transform(scaled_df.iloc[:,:-4])
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc' + str(i) for i in range(1,n_comp+1)])
    pca_df = pd.concat([principalDf,scaled_df.iloc[:,-4:]],axis=1)
    return pca_df


def get_sharpe_models(combined):
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    for i in models:
        df = combined[combined['model'] ==i]
        max_ret_df = df[df['Returns']==df['Returns'].max()].tail(1)[['model','Best_Params','Proportions']]
        max_36_df = df[df['36 Month Returns'] == df['36 Month Returns'].max()].tail(1)[['model','Best_Params','Proportions']]
        max_60_df = df[df['60 Month Returns'] == df['60 Month Returns'].max()].tail(1)[['model','Best_Params','Proportions']]
        idx_df = pd.DataFrame({'measure':['best','36m','60m']})
        p_df = pd.concat([max_ret_df,max_36_df,max_60_df]).reset_index().drop('index',axis=1)
        if i == 'LogisticRegression':
            log_df = pd.concat([idx_df,p_df],axis=1)
        elif i == 'RandomForestClassifier':
            rfc_df = pd.concat([idx_df,p_df],axis=1)
        else:
            gbc_df = pd.concat([idx_df,p_df],axis=1)
    return pd.concat([log_df,rfc_df,gbc_df])
