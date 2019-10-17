from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import pandas as pd

def lc_transform(df_original):
    df = df_original.copy()
    num_features = ['issue_d_year','funded_amnt','int_rate','installment',
    'FICO','DTI','annual_income','emp_length','earliest_cr_line',
    'negative_activity','inq_last_6mths','delinq_2yrs','open_accts',
    'mortgage_accts','tot_cur_bal','revolving','revol_util']

    cat_features = ['grade','sub_grade','term','purpose','application_type',
    'home_ownership','addr_state','verified']
    sc = StandardScaler()
    X1 = sc.fit_transform(df[num_features])
    ohe = OneHotEncoder()
    X2 = ohe.fit_transform(df[cat_features])
    categories = []
    for i in ohe.categories_:
        categories += list(i)
    test_cat = pd.DataFrame(X2.todense(),columns=categories)
    test_num = pd.DataFrame(X1,columns=num_features)
    output = df[['loan_status','funded_amnt','total_pymnt']]
    output[['loan_amount','loan_payoff']] = output[['funded_amnt','total_pymnt']]
    output.drop(['funded_amnt','total_pymnt'],axis=1,inplace=True)
    scaled_df = pd.concat([test_num,test_cat,output],axis=1)
    return scaled_df

def lc_stratified_train_test_split(data,data_x, data_y, state=42, size = 0.2, splits=1):
    split = StratifiedShuffleSplit(n_splits=splits,test_size=size,random_state=state)
    for i, j in split.split(data_x,data_y):
        strat_train_set = data.iloc[i]
        strat_test_set = data.iloc[j]

    return strat_train_set, strat_test_set

def lc_create_sets(scaled_df):
    x_cols = list(scaled_df.columns)
    x_cols.remove('loan_status')
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
