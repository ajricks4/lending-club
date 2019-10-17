from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
import pandas as pd

def lc_transform(df_original):
    df = df_original.copy()
    num_features = ['issue_d_year','funded_amnt','int_rate','installment',
    'FICO','DTI','annual_income','emp_length','earliest_cr_line',
    'negative_activity','inq_last_6mths','delinq_2yrs','open_accts',
    'mortgage_accts','tot_cur_bal','revolving','revol_util']

    cat_features = ['grade','addr_state','term','purpose','application_type',
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
    output = df[['funded_amnt','total_pymnt','loan_status']]
    output[['loan_amount','loan_payoff']] = output[['funded_amnt','total_pymnt']]
    output = output[['loan_amount','loan_payoff','loan_status']]
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


def lc_balance_sets(scaled_df):
    X_train, X_test, y_train, y_test = train_test_split(scaled_df.iloc[:,:-1], scaled_df.iloc[:,-1], test_size=0.1, random_state=42)
    train_set = pd.concat([X_train,y_train],axis=1)
    class_0 = train_set[train_set['loan_status'] == 0]
    class_1 = train_set[train_set['loan_status'] == 1].iloc[:class_0.shape[0],:]
    train_set = pd.concat([class_0,class_1])
    X_train = train_set.iloc[:,:-1]
    y_train = train_set.iloc[:,-1]
    train_loan_data = X_train[['loan_amount','loan_payoff']]
    X_train.drop(['loan_amount','loan_payoff'],axis=1,inplace=True)
    test_loan_data = X_test[['loan_amount','loan_payoff']]
    X_test.drop(['loan_amount','loan_payoff'],axis=1,inplace=True)
    return X_train, X_test, y_train, y_test, train_loan_data, test_loan_data
