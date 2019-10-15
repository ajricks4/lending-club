import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pdp
import matplotlib as mpl
import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)





def plot_loan_breakdown_pie(df):
    """
    Plots pie chart of Lending Club's loan originations broken down by
    loan status and saves image to './images/lc_loan_status.png'

    Params:
    df (pd.DataFrame): dataframe of Lending Club loans

    Returns:
    """
    mpl.rcParams['font.size'] = 30
    pie_df = df.groupby('loan_status').sum()[['loan_amnt']]
    pie_df.reset_index(inplace=True)
    bad_loan_sum = pie_df.sort_values(by='loan_amnt').head(4)['loan_amnt'].sum()
    pie_df.loc[pie_df.shape[0]+1] = ['Late, In Default / Grace Period',bad_loan_sum]
    pie_df = pie_df[(pie_df['loan_status']=='Charged Off') | (pie_df['loan_status'] == 'Current') | (pie_df['loan_status'] == 'Fully Paid') | (pie_df['loan_status'] == 'Charged Off') | (pie_df['loan_status'] == 'Late, In Default / Grace Period')]
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(1,1,1)
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.array([0,2,4,8]))
    explode = [0.1 for i in range(pie_df.shape[0])]
    ax.pie(pie_df['loan_amnt'],autopct=apc,labels=pie_df['loan_status'],explode=explode,textprops={'size':45},colors=colors)
    plt.title('Lending Club Loan Status Across ${}B Loans'.format(round(pie_df['loan_amnt'].sum()/1000000000,1)),fontsize = 45,fontweight='bold')
    plt.savefig('images/lc_loan_status.png')
    plt.show()

def plot_grade_breakdown_pie(df):
    """
    !!! Note that this function requires that the charged_off_amnt column has been created
    Plots pie chart of Lending Club's loan originations by grade with a further
    breakdown showing the amounts by status: fully paid off, current, charged off, late / default / in grace period.
    The image is saved to './images/lc_grade_pie.png'

    Params:
    df (pd.DataFrame): dataframe of Lending Club loans

    Returns:
    """
    grade_pie = df.groupby('grade').sum()[['loan_amnt','total_rec_prncp','out_prncp','delinq_amnt','charged_off_amnt']]
    grade_pie.reset_index(inplace=True)
    grade_pie['bad_debt'] = grade_pie['delinq_amnt'] + grade_pie['charged_off_amnt']
    grade_pie.drop(['delinq_amnt','charged_off_amnt'],inplace=True, axis=1)
    amounts = []
    for i in range(grade_pie.shape[0]):
        for j in range(2,grade_pie.shape[1]):
            amounts.append(grade_pie.iloc[i,j])
    labels = ['Paid-Off','Outs','Bad Debt']* 4
    labels += ['','',''] * 3
    fig = plt.figure(figsize= (25,25))

    size = 0.3

    ax = fig.add_subplot(1,1,1)
    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.array([0,4,8,12,4,8,12]))
    inner_colors = cmap(np.array([1,2,3,5,6,7,9,10,11,13,14,15,5,6,7,9,10,11,13,14,15]))

    ax.pie(grade_pie['loan_amnt'], radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'),startangle=90,labels=grade_pie['grade'],textprops={'size':30,'color':'blue'},autopct = apc,pctdistance=0.9)

    ax.pie(amounts, radius=1-size, colors=inner_colors,
       wedgeprops=dict(width=0.35, edgecolor='w'),startangle=90,labels=labels,textprops={'size':30,'color':'black'},autopct = apc,pctdistance = 0.575,labeldistance=0.8)

    plt.title('Lending Club Loans By Grade and Status',fontsize = 35,fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/lc_grade_pie.png')
    plt.show()


def apc(x):
    if round(x,1) < 2.5:
        return ''
    else:
        return '{}%'.format(round(x,1))

def choro_debt_state(df):
    """
    Plots choromap at state level of debt originations and chargeoffs

    Params:
    df (pd.DataFrame): Lending Club dataframe
    """
    outcome_df = df[(df['loan_status'] == 'Charged Off') | (df['loan_status'] == 'Fully Paid') ]
    outcome_df = outcome_df.groupby(['addr_state','loan_status']).sum()['loan_amnt']
    outcome_df = pd.DataFrame(outcome_df).reset_index()
    outcome_df['loan_amnt'] = round(outcome_df['loan_amnt'] / 1000000,1)
    total_loans = outcome_df.groupby('addr_state').sum()['loan_amnt']
    text = []
    for i in total_loans.index:
        charge = outcome_df[(outcome_df['addr_state'] == i) & (outcome_df['loan_status'] =='Charged Off')]['loan_amnt'].values[0]
        paid = outcome_df[(outcome_df['addr_state'] == i) & (outcome_df['loan_status'] =='Fully Paid')]['loan_amnt'].values[0]
        text.append('{}: <br> Fully Paid: ${}M  <br> Charged Off: ${}M'.format(i,paid,charge))
    data = dict(type='choropleth',locations=total_loans.index,
            locationmode='USA-states',colorscale='Blues',
           z = total_loans.values,colorbar = {'title':'$ Millions'},
            text = text)
    layout = dict(geo={'scope':'usa'})

    choromap = go.Figure(data=[data],layout=layout)
    choromap.update_layout(title_text='Lending Club Loan Originations Per State',geo_scope='usa')
    plotly.offline.plot(choromap, filename = './images/state_choro_loan_total.png', auto_open=False)
    iplot(choromap,validate=False, filename='./images/state_choro',image='png')

def choro_debt_state_count(df):
    """
    Plots choromap at state level of debt originations and chargeoffs

    Params:
    df (pd.DataFrame): Lending Club dataframe
    """

    outcome_df = df.groupby(['addr_state','loan_status']).count()['loan_amnt'].reset_index()
    outcome_df['loan_amnt'] = round(outcome_df['loan_amnt'] / 1000,1)
    total_loans = outcome_df.groupby('addr_state').sum()['loan_amnt']
    text = []
    for i in total_loans.index:
        charge = outcome_df[(outcome_df['addr_state'] == i) & (outcome_df['loan_status'] =='Charged Off')]['loan_amnt'].values[0]
        paid = outcome_df[(outcome_df['addr_state'] == i) & (outcome_df['loan_status'] =='Fully Paid')]['loan_amnt'].values[0]
        try:
            current = outcome_df[(outcome_df['addr_state'] == i) & (outcome_df['loan_status'] =='Current')]['loan_amnt'].values[0]
        except:
            current = 0
        try:
            others = outcome_df[(outcome_df['addr_state'] == i) & (outcome_df['loan_status'] !='Current') & (outcome_df['loan_status'] !='Fully Paid') & (outcome_df['loan_status'] !='Charged Off')]['loan_amnt'].values[0]
        except:
            others=0
        text.append('{}: <br> Fully Paid: {}K  <br> Current: {}K <br> Charged Off: {}K <br> Late / In Default / Grace Period {}K'.format(i,paid,current,charge,others))
    data = dict(type='choropleth',locations=total_loans.index,
            locationmode='USA-states',colorscale='Blues',
           z = total_loans.values,colorbar = {'title':'Loan Count (thousands)'},
            text = text)
    layout = dict(geo={'scope':'usa'})

    choromap = go.Figure(data=[data],layout=layout)
    choromap.update_layout(title_text='Lending Club Loan Count Per State',geo_scope='usa')
    plotly.offline.plot(choromap, filename = './images/state_choro_loan_count_total.png', auto_open=False)
    iplot(choromap,validate=False, filename='./images/state_choro',image='png')

def lc_time_series(df1):
    """
    Plots an overlay of cumulative loan growth and lending club stock price over lending club
    loans originated per month

    Params:
    df (pd.DataFrame): lending club data

    Returns:

    """
    df = df1.copy()
    df['issue_d'] = pd.to_datetime(df['issue_d'])
    by_date = df.groupby('issue_d').sum()['loan_amnt'].reset_index()
    by_date['loan_amnt'] = round(by_date['loan_amnt'] / 1000000,1)
    lc_stock = pd.read_csv('data/LC.csv')[['Adj Close','Date']]
    lc_stock['Date'] = pd.to_datetime(lc_stock['Date'])
    lc_stock = lc_stock[(lc_stock['Date'] <= by_date['issue_d'].max()) & (lc_stock['Date'] > by_date['issue_d'].min())]
    fig = plt.figure(figsize=(40,40))
    ax = fig.add_subplot(1,1,1)
    ax.plot(by_date['issue_d'],by_date['loan_amnt'])
    plt.title('Lending Club Loan Origination Per Month',fontsize=80,fontweight='bold')
    plt.xlabel('Date',fontsize=50)
    plt.ylabel('Loans $ Millions',fontsize=50)
    ax.tick_params(axis="x", labelsize=50)
    ax.tick_params(axis="y", labelsize=50)
    plt.scatter(by_date[by_date['issue_d'] == '2014-08-01']['issue_d'],by_date[by_date['issue_d'] == '2014-08-01']['loan_amnt'], marker='o', color='r',s=5000,linewidths=10)
    a = plt.axes([.2,.6,.25,.25],facecolor='w')
    #'left,bottom.widt,height'
    plt.plot(by_date['issue_d'],np.cumsum(by_date['loan_amnt']))
    plt.title('Cumulative Loan Growth',fontsize=50,fontweight='bold')
    plt.xlabel('Date',fontsize=45)
    plt.ylabel('Loan $ Millions',fontsize=45)
    a.tick_params(axis="x", labelsize=35)
    a.tick_params(axis="y", labelsize=35)
    plt.scatter(by_date[by_date['issue_d'] == '2014-08-01']['issue_d'],np.cumsum(by_date['loan_amnt'])[86], marker='o', color='r',s=5000,linewidths=10)
    b = plt.axes([.2,.25,.25,.25])
    plt.plot(lc_stock['Date'],lc_stock['Adj Close'])
    plt.title('Lending Club Stock Price 2014 - Current',fontsize=50,fontweight='bold')
    plt.xlabel('Date',fontsize=45)
    plt.ylabel('Price',fontsize=45)
    b.axes.get_xaxis().set_ticks([lc_stock['Date'][i] for i in range(lc_stock.shape[0]) if i % 18 ==0])
    b.tick_params(axis="x", labelsize=35)
    b.tick_params(axis="y", labelsize=35)
    plt.savefig('images/lc_time_series.png')
    plt.show()


def lc_individual_profile(df_original):
    """
    From the cleaned dataframe, create a few plots to detail the
    profile of individuals

    Params:
    df_original (pd.DataFrame):  cleaned lending club data frame

    Returns:

    """
    df = df_original.copy()
    df['loan_outcome'] = df['loan_status'].apply(lambda x: 1 if (x=='Current') or (x=='Fully Paid') else 0)
    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(2,2,1)
    ax1 = sns.violinplot(df['loan_amnt'])
    plt.xlabel('Loan Amount',fontsize=20)
    plt.ylabel('Distribution',fontsize=20)
    plt.title('Loan Amount',fontsize=25,fontweight='bold')
    ax1.tick_params('x',labelsize=15)
    ax1.tick_params('y',labelsize=15)
    ax2 = fig.add_subplot(2,2,2)
    ax2 = sns.distplot(df[df['fico_range_low'].isnull() == False]['fico_range_low'])
    ax2.tick_params('x',labelsize=15)
    ax2.tick_params('y',labelsize=15)
    plt.xlabel('Fico Score',fontsize=20)
    plt.ylabel('Frequency',fontsize=20)
    plt.title('Distribution of Fico Range Low',fontsize=20,fontweight='bold')
    ax3 = fig.add_subplot(2,2,3)
    ax3 = sns.distplot(df['emp_length'])
    ax3.tick_params('x',labelsize=15)
    ax3.tick_params('y',labelsize=15)
    plt.xlabel('Years Working at Current Job',fontsize=20)
    plt.title('Employment Length',fontsize=25,fontweight='bold')
    ax4 = fig.add_subplot(2,2,4)
    labels = list(df['home_ownership'].value_counts().index[:-3]) + ['','','']
    explode = [0.1 for i in range(len(labels))]
    ax4.pie(df['home_ownership'].value_counts(),labels = labels,autopct=apc,textprops={'size':20,'color':'black'},explode=explode)
    plt.title('Living Location Status',fontsize=25,fontweight='bold')

    plt.savefig('images/borrower_profile.png')
    plt.show()
