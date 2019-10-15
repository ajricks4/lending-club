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
    pass










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
