import pandas as pd
import numpy as np



def clean_lc_for_plotting(df_original):
    """
    Takes in the original Lending Club  data and returns a
    cleaned DataFrame for plotting and early data exploration.

    Args:
        df_original (pandas DataFrame): Lending Club dataframe containing loan data

    Returns:
        df (pandas DataFrame): Dataframe with columns cleaned for the LC_Plotter.py module
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

    Args:
        df_original (pd.DataFrame): Dataframe returned by the clean_lc_for_plotting function

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
                'avg_cur_bal','sec_app_num_rev_accts','total_rev_hi_lim','total_acc','total_rec_late_fee',
                'emp_title','desc','title','mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog',
                'delinq_amnt','mths_since_recent_bc','num_sats','num_tl_op_past_12m','percent_bc_gt_75','tax_liens',
                'total_bal_ex_mort','sec_app_revol_util','sec_app_collections_12_mths_ex_med','sec_app_mths_since_last_major_derog',
                'initial_list_status','all_util','inq_fi','inq_last_12m','pub_rec_bankruptcies','recoveries','charged_off_amnt'
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
    df['revolving'] = df.apply(lambda row: clean_ij_revol(row),axis=1)
    df.drop(['revol_bal','revol_bal_joint'],inplace=True,axis=1)
    df[['pub_rec','chargeoff_within_12_mths','num_accts_ever_120_pd','num_tl_90g_dpd_24m','sec_app_chargeoff_within_12_mths']].fillna(0,inplace=True)
    df['negative_activity'] = df.apply(lambda row: clean_ij_pub_rec(row),axis=1)
    df.drop(['pub_rec','chargeoff_within_12_mths','num_accts_ever_120_pd','num_tl_90g_dpd_24m','sec_app_chargeoff_within_12_mths'],axis=1,inplace=True)
    df['issue_d_year'] = df['issue_d'].apply(lambda x: int(str(x).split('-')[0]))
    df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x == 'Fully Paid' else 0)
    df['delinq_2yrs'].fillna(0,inplace=True)
    df['tot_cur_bal'].fillna(np.mean(df['tot_cur_bal']),inplace=True)
    df['FICO'].fillna(np.mean(df['FICO']),inplace=True)
    df['DTI'].fillna(np.mean(df['DTI']),inplace=True)
    df['annual_income'].fillna(np.mean(df['annual_income']),inplace=True)
    df['mortgage_accts'].fillna(0,inplace=True)
    df['open_accts'].fillna(0,inplace=True)
    df['revolving'].fillna(0,inplace=True)
    df['earliest_cr_line'] = df.apply(lambda row: row['issue_d_year'] if str(row['earliest_cr_line']).lower() == 'nan' else row['earliest_cr_line'],axis=1)
    df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda x: int(str(x).lower().strip('abcdefghijklmnopqrstuvwxyz-')))
    df['inq_last_6mths'].fillna(0,inplace=True)
    df['revol_util'].fillna('0.00%',inplace=True)
    df['negative_activity'].fillna(0,inplace=True)
    df['revol_util'] = df['revol_util'].apply(lambda x: float(str(x).strip('%')))
    df.drop('issue_d',axis=1,inplace=True)
    df_final = df[['issue_d_year','grade','sub_grade','funded_amnt','term','int_rate','installment',
                'purpose','application_type','FICO','DTI','annual_income','emp_length','home_ownership',
                'addr_state','earliest_cr_line','negative_activity','inq_last_6mths','delinq_2yrs',
                'verified','open_accts','mortgage_accts','tot_cur_bal', 'revolving','revol_util','loan_status',
                'total_pymnt']]
    return df_final

def clean_ij_pub_rec(row):
    """
    Function to be used within a .apply(lambda row: clean_ij_pub_rec(row))

    Args:
        row (pandas DataFrame row): Row within the initially cleaned data

    Returns:
        bad_num (float): Count of public record or otherwise negative credit related events.
    """
    if row['application_type'] == 'Individual':
        bad_num =  row['pub_rec'] + row['chargeoff_within_12_mths'] + row['num_accts_ever_120_pd'] + row['num_tl_90g_dpd_24m']
        return bad_num
    else:
        bad_num = row['pub_rec'] + row['chargeoff_within_12_mths'] + row['num_accts_ever_120_pd'] + row['num_tl_90g_dpd_24m'] +row['sec_app_chargeoff_within_12_mths']
        return bad_num

def clean_ij_fico(row):
    """
    Function to be used within a .apply(lambda row: clean_ij_pub_rec(row))

    Args:
        row (pandas DataFrame row): Row within the initially cleaned data

    Returns:
        fico (float): Average of different FICO measures.
    """
    if row['application_type'] == 'Individual':
        fico =  (row['fico_range_low'] + row['fico_range_high']) / 2
    else:
        fico =  (row['fico_range_low'] + row['fico_range_high'] + row['sec_app_fico_range_low'] + row['sec_app_fico_range_high'])  /4
    return fico

def clean_ij_dti(row):
    """
    Function to be used within a .apply(lambda row: clean_ij_pub_rec(row))

    Args:
        row (pandas DataFrame row): Row within the initially cleaned data

    Returns:
        DTI (float): Debt to income ratio of borrower.
    """
    if row['application_type'] == 'Individual':
        DTI =  row['dti']
    else:
        DTI =  row['dti_joint']
    return DTI

def clean_ij_inc(row):
    """
    Function to be used within a .apply(lambda row: clean_ij_pub_rec(row))

    Args:
        row (pandas DataFrame row): Row within the initially cleaned data

    Returns:
        income (float): Income of borrower.
    """
    if row['application_type'] == 'Individual':
        income = row['annual_inc']
    else:
        income =  row['annual_inc_joint']
    return income

def clean_ij_mort(row):
    """
    Function to be used within a .apply(lambda row: clean_ij_pub_rec(row))

    Args:
        row (pandas DataFrame row): Row within the initially cleaned data

    Returns:
        mort_acc (float): Count of mortgage accounts.
    """
    if row['application_type'] == 'Individual':
        mort_acc =  row['mort_acc']
    else:
        mort_acc =  (row['mort_acc'] + row['sec_app_mort_acc']) / 2
    return mort_acc

def clean_ij_ver_stat(row):
    """
    Function to be used within a .apply(lambda row: clean_ij_pub_rec(row))

    Args:
        row (pandas DataFrame row): Row within the initially cleaned data

    Returns:
        status (string): Description of the verification status of an applicant.
    """
    if row['application_type'] == 'Individual':
        status =  row['verification_status']
    else:
        if (row['verification_status'] == 'Not Verified') or (row['verification_status_joint'] == 'Not Verified'):
            status =  'Not Verified'
        elif (row['verification_status'] == 'Verified') and (row['verification_status_joint'] == 'Verified'):
            status =  'Verified'
        else:
            status =  'Source Verified'
    return status

def clean_ij_open_acc(row):
    """
    Function to be used within a .apply(lambda row: clean_ij_pub_rec(row))

    Args:
        row (pandas DataFrame row): Row within the initially cleaned data

    Returns:
        open_acc (float): Count of open accounts.
    """
    if row['application_type'] == 'Individual':
        open_acc =  row['open_acc']
    else:
        open_acc =  (row['open_acc'] + row['sec_app_open_acc'])/2
    return open_acc

def clean_ij_revol(row):
    """
    Function to be used within a .apply(lambda row: clean_ij_pub_rec(row))

    Args:
        row (pandas DataFrame row): Row within the initially cleaned data

    Returns:
        revol (float): Revolving balance.
    """
    if row['application_type'] == 'Individual':
        revol =  row['revol_bal']
    else:
        revol =  row['revol_bal_joint']
    return revol



def clean_up_status(x):
    """
    Function that cleans up the status string of loans that include "Does not comply"

    Args:
        x (string): Loan status

    Returns:
        stat (string): Cleaned up loan status string
    """
    if str(x)[:4] == 'Does':
        stat =  str(x).split(':')[-1]
    else:
         stat = x
    return stat

def get_charge_off(row):
    """
    Function that takes in pandas dataframe row.

    Args:
        row (pandas DataFrame): row of the Lending Club DataFrame

    Returns:
        num (float): dollar amount that charged-off
    """
    if row['loan_status'] == 'Charged Off':
        num =  row['loan_amnt'] - row['total_rec_prncp']
    else:
        num =  0
    return num

def clean_emp_length(x):
    """
    Function that cleans the employment length column.

    Args:
        x (string): string that describes the employment length of borrower.

    Returns:
        detail (int): Years borrower has worked at current job.
    """
    if str(x) == 'nan':
        detail =  0
    else:
        detail =  int(str(x).strip(' +< years'))
    return detail
