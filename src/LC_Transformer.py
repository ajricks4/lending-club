from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
import pandas as pd
from sklearn.decomposition import PCA, NMF

def lc_transform(df_original):
    """
    Function that takes the cleaned Lending Club DataFrame and transforms it into a
    scaled dataframe.

    Args:
        df_original (pandas DataFrame): pandas DataFrame of the cleaned Lending Club Data

    Returns:
        scaled_df (pandas DataFrame): pandas DataFrame with transformed data columns including one
                                      hot encoding and standarding the data.
    """
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

def lc_balance_sets(scaled_df,proportion = 1.0):
    """
    Function that takes in a dataframe and prepares a train-test-split
    with a specified proportion of the majority class to the minority class.

    Args:
        scaled_df (pandas DataFrame): DataFrame outputted from the lc_transform function.
        proportion (float): proportion of the majority class to the minority class.

    Returns:
        X_train (pandas DataFrame): X training data
        y_train (pandas DataFrame): y training data
        X_test (pandas DataFrame): X test data
        y_test (pandas DataFrame): y test data
        train_loan_data (pandas DataFrame): training loan data including loan length, loan amount, and loan payoff.
        test_loan_data (pandas DataFrame): test loan data including loan length, loan amount, and loan payoff.
    """
    X_train, X_test, y_train, y_test = train_test_split(scaled_df.iloc[:,:-1], scaled_df.iloc[:,-1], test_size=0.1)
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
    """
    Takes in a scaled dataframe and computes the PCA eigenmatrix.

    Args:
        scaled_df (pandas dataframe): dataframe with scaled data
        n_comp (int): numer of components (eigenvectors)

    Returns:
        pac_df (pandas dataframe): eigenmatrix
    """
    pca = PCA(n_components=n_comp)
    principalComponents = pca.fit_transform(scaled_df.iloc[:,:-4])
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc' + str(i) for i in range(1,n_comp+1)])
    pca_df = pd.concat([principalDf,scaled_df.iloc[:,-4:]],axis=1)
    return pca_df


def get_compiled_models(combined):
    """
    Obtains the models and model data with the highest blended returns, 36 month returns or 60 month returns.

    Args:
        combined (pandas DataFrame): dataframe combined from concatentating several output dataframes from the
                                    lc_proportion_grid_search function in the LC_Models.py module.

    Returns:
        compiled (pandas DataFrame): dataframe containing 9 models selected based on scoring the best returns across several
                                     criteria.
    """
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
    compiled = pd.concat([log_df,rfc_df,gbc_df])
    return compiled
