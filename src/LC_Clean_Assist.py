import pandas as pd


def clean_lc_for_plotting(df_original):
    """
    Takes in original LC data and returns a cleaned DataFrame for plotting

    Params:
    df_original (pd.DataFrame): lending club dataframe

    Returns
    df (pd.DataFrame)
    """
    df = df_original.copy()
    df.drop(index = df[df['loan_amnt'].isnull()].index,inplace=True)
    df.drop('Unnamed: 0',inplace=True,axis = 1)
    df['loan_status'] = df['loan_status'].apply(lambda x: clean_up_status(x))
    df['charged_off_amnt'] = df.apply(lambda row: get_charge_off(row),axis=1)
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    df['term'] = df['term'].apply(lambda x: int(str(x).split(' ')[1]))
    df['int_rate'] = df['int_rate'].apply(lambda x: float(str(x).split('%')[0]) / 100)
    df['emp_length'] = df['emp_length'].apply(lambda x: clean_emp_length(x))
    return df


def clean_lc_for_models(df_original):
    """
    Takes in the dataframe produced by clean_lc_for_plotting
    and returns a cleaned dataframe

    Params:
    df_original (pd.DataFrame): dataframe returned by the clean_lc_for_plotting function

    Returns:
    df (pd.DataFrame)
    """
    df = df_original[(df_original['loan_status']=='Fully Paid') | (df_original['loan_status']=='Charged Off')]
    drop_cols = ['acc_now_delinq','acc_open_past_24mths','bc_open_to_buy','bc_util',
                'collection_recovery_fee','collections_12_mths_ex_med','id','last_credit_pull_d',
                'last_fico_range_high','last_fico_range_low','last_pymnt_amnt','last_pymnt_d',
                'member_id','policy_code','mo_sin_old_il_acct','mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op',
                'mo_sin_rcnt_tl','mths_since_rcnt_il','mths_since_recent_bc_dlq',
                'mths_since_recent_inq','mths_since_recent_revol_delinq','next_pymnt_d',
                'sec_app_earliest_cr_line','sec_app_inq_last_6mths','deferral_term','debt_settlement_flag',
                'debt_settlement_flag_date','settlement_status','settlement_date','settlement_amount',
                'settlement_percentage','settlement_term','payment_plan_start_date',
                'orig_projected_additional_accrued_interest','total_il_high_credit_limit','total_bc_limit',
                'tot_hi_cred_lim','sec_app_open_act_il','tot_coll_amt','open_acc_6m','open_act_il',
                'open_il_12m','open_il_24m','total_bal_il','open_rv_12m','open_rv_24m','max_bal_bc',
                'total_cu_tl','num_actv_bc_tl','num_actv_rev_tl','num_bc_sats','num_bc_tl',
                'num_il_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_tl_120dpd_2m',
                'num_tl_30dpd','pct_tl_nvr_dlq','pymnt_plan','url','zip_code','loan_amnt','funded_amnt_inv',
                'out_prncp','out_prncp_inv','total_pymnt_inv','total_rec_prncp','total_rec_int','il_util',
                'avg_cur_bal','sec_app_num_rev_accts','total_rev_hi_lim','total_acc','total_rec_late_fee'
                ] + [j for j in df.columns if 'hardship' in j.lower()]
    df.drop(drop_cols,axis=1,inplace=True)
    df['FICO'] = df.apply(lambda row: clean_ij_fico(row),axis=1)
    df.drop(['fico_range_low','fico_range_high','sec_app_fico_range_low','sec_app_fico_range_high'],axis=1,inplace=True)
    df['DTI'] = df.apply(lambda row: clean_ij_dti(row),axis=1)
    df.drop(['dti','dti_joint'],axis=1,inplace=True)
    df['annual_income'] = df.apply(lambda row: clean_ij_inc(row),axis=1)
    df.drop(['annual_inc','annual_inc_joint'],axis=1,inplace=True)
    df['mortgage_accts'] = df.apply(lambda row: clean_ij_mort(row),axis=1)
    df.drop(['mort_acc','sec_app_mort_acc'],inplace=True,axis=1)
    df['verified'] = df.apply(lambda row: clean_ij_ver_stat(row),axis=1)
    df.drop(['verification_status','verification_status_joint'],inplace=True,axis=1)
    df['open_accts'] = df.apply(lambda row: clean_ij_open_acc(row),axis=1)
    df.drop(['open_acc','sec_app_open_acc'],inplace=True,axis=1)

    return df







def clean_ij_fico(row):
    if row['application_type'] == 'Individual':
        return (row['fico_range_low'] + row['fico_range_high']) / 2
    else:
        return (row['fico_range_low'] + row['fico_range_high'] + row['sec_app_fico_range_low'] + row['sec_app_fico_range_high'])  /4

def clean_ij_dti(row):
    if row['application_type'] == 'Individual':
        return row['dti']
    else:
        return row['dti_joint']

def clean_ij_inc(row):
    if row['application_type'] == 'Individual':
        return row['annual_inc']
    else:
        return row['annual_inc_joint']

def clean_ij_mort(row):
    if row['application_type'] == 'Individual':
        return row['mort_acc']
    else:
        return (row['mort_acc'] + row['sec_app_mort_acc']) / 2
def clean_ij_ver_stat(row):
    if row['application_type'] == 'Individual':
        return row['verification_status']
    else:
        if ('Not Verified' in row['verification_status']) or ('Not Verified' in row['verification_status_joint']):
            return 'Not Verified'
        elif ('Source Verified' in row['verification_status']) or ('Source Verified' in row['verification_status_joint']):
            return 'Source Verified'
        else:
            return 'Verified'
def clean_ij_open_acc(row):
    if row['application_type'] == 'Individual':
        return row['open_acc']
    else:
        return (row['open_acc'] + row['sec_app_open_acc'])/2








def clean_up_status(x):
    if str(x)[:4] == 'Does':
        return str(x).split(':')[-1]
    else:
        return x

def get_charge_off(row):
    if row['loan_status'] == 'Charged Off':
        return  row['loan_amnt'] - row['total_rec_prncp']
    else:
        return 0

def clean_emp_length(x):
    if str(x) == 'nan':
        return 0
    else:
        return int(str(x).strip(' +< years'))
