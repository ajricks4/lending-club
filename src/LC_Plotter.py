import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pdp
import matplotlib as mpl
#import plotly.graph_objs as go
#import plotly
#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)
import src.LC_Models as LCM
from sklearn.decomposition import PCA, NMF





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
    pie_df.loc[pie_df.shape[0]+1] = ['Troubled Debt',bad_loan_sum]
    pie_df = pie_df[(pie_df['loan_status']=='Charged Off') | (pie_df['loan_status'] == 'Current') | (pie_df['loan_status'] == 'Fully Paid') | (pie_df['loan_status'] == 'Charged Off') | (pie_df['loan_status'] == 'Troubled Debt')]
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(1,1,1)
    cmap = plt.get_cmap("tab20")
    colors = cmap(np.array([0,2,4,8]))
    explode = [0.1 for i in range(pie_df.shape[0])]
    ax.pie(pie_df['loan_amnt'],autopct=apc,labels=pie_df['loan_status'],explode=explode,textprops={'size':60,'weight':'bold'},colors=colors)
    plt.title('Lending Club Loan Status Across ${}B Loans'.format(round(pie_df['loan_amnt'].sum()/1000000000,1)),fontsize = 70,fontweight='bold')
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
    labels = ['Paid','Outs','Bad']* 4
    labels += ['','',''] * 3
    fig = plt.figure(figsize= (25,25))

    size = 0.3

    ax = fig.add_subplot(1,1,1)
    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.array([0,4,8,12,4,8,12]))
    inner_colors = cmap(np.array([1,2,3,5,6,7,9,10,11,13,14,15,5,6,7,9,10,11,13,14,15]))

    ax.pie(grade_pie['loan_amnt'], radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'),startangle=90,labels=grade_pie['grade'],textprops={'size':40,'color':'blue','weight':'bold'},autopct = apc,pctdistance=0.9)

    ax.pie(amounts, radius=1-size, colors=inner_colors,
       wedgeprops=dict(width=0.35, edgecolor='w'),startangle=90,labels=labels,textprops={'size':40,'color':'black','weight':'bold'},autopct = apc,pctdistance = 0.60,labeldistance=0.835)

    plt.title('Lending Club Loans By Grade and Status',fontsize = 45,fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/lc_grade_pie.png')
    plt.show()


def apc(x):
    """
    Sets apc for use in a pie chart
    Args:
        x (float): percentage of total

    Returns:
        display (string): display of perentage if above 2.5%.
    """
    if round(x,1) < 2.5:
        display =  ''
    else:
        display =  '{}%'.format(round(x,1))
    return display

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
    fig = plt.figure(figsize=(40,40))
    ax1 = fig.add_subplot(2,2,1)
    ax1 = sns.violinplot(df['loan_amnt'])
    plt.xlabel('Loan Amount',fontsize=50)
    plt.ylabel('Distribution',fontsize=50)
    plt.title('Loan Amount',fontsize=50,fontweight='bold')
    ax1.tick_params('x',labelsize=50)
    ax1.tick_params('y',labelsize=50)
    ax2 = fig.add_subplot(2,2,2)
    ax2 = sns.distplot(df[df['fico_range_low'].isnull() == False]['fico_range_low'])
    ax2.tick_params('x',labelsize=50)
    ax2.tick_params('y',labelsize=50)
    plt.xlabel('Fico Score',fontsize=50)
    plt.ylabel('Frequency',fontsize=50)
    plt.title('Distribution of Fico Range Low',fontsize=50,fontweight='bold')
    ax3 = fig.add_subplot(2,2,3)
    ax3 = sns.distplot(df['emp_length'])
    ax3.tick_params('x',labelsize=50)
    ax3.tick_params('y',labelsize=50)
    plt.xlabel('Years Working at Current Job',fontsize=50)
    plt.title('Employment Length',fontsize=50,fontweight='bold')
    ax4 = fig.add_subplot(2,2,4)
    labels = list(df['home_ownership'].value_counts().index[:-3]) + ['','','']
    explode = [0.1 for i in range(len(labels))]
    ax4.pie(df['home_ownership'].value_counts(),labels = labels,autopct=apc,textprops={'size':50,'color':'black'},explode=explode)
    plt.title('Living Location Status',fontsize=50,fontweight='bold')

    plt.savefig('images/borrower_profile.png')
    plt.tight_layout()
    plt.show()


def lc_returns(df):
    """
    Createds dataframe displaying returns grouped by Lending Club grades.

    Args:
        df (pandas DataFrame): Lending Club Data

    Returns:
        grouped (pandas DataFrame): DataFrame displaying the returns by grade.
    """
    grouped = df.groupby(['grade','term']).sum()[['funded_amnt','total_pymnt']]
    grouped.reset_index(inplace=True)
    grouped['return'] = (grouped['total_pymnt'] / grouped['funded_amnt'] - 1)*100
    grouped['annualized_pct'] = ((((grouped['return']/100)+1) ** (1 / (grouped['term']/12))) - 1)*100
    grouped['return_formatted'] = grouped['return'].apply(lambda x: str(round(x,1)) + '%')
    grouped['annualized_return_formatted'] = grouped['annualized_pct'].apply(lambda x: str(round(x,1)) + '%')
    grouped['return'] = grouped['return'].apply(lambda x: round(x,1))
    grouped['annualized_pct'] = grouped['annualized_pct'].apply(lambda x: round(x,1))
    grouped.drop(['funded_amnt','total_pymnt'],axis=1,inplace=True)
    return grouped

def lc_plot_returns(returns,title,filepath):
    """
    Plots the returns by Lending Club grades.

    Args:
        returns (pandas DataFrame): DataFrame with returns grouped by Lending Club Grade.
        title (string): Title to add to the plot.
        filepath (string): filepath to save image into.

    Returns:

    """
    fig = plt.figure(figsize=(30,30))
    ax1 = fig.add_subplot(1,1,1)
    ax1.bar(returns['grade'],returns['return'])
    plt.title(title,fontsize=40,fontweight='bold')
    plt.xlabel('Grade',fontdict={'size':40})
    plt.ylabel('Return',fontdict={'size':40})
    ax1.tick_params('x',labelsize=35)
    ax1.tick_params('y',labelsize=35)
    plt.savefig(filepath)
    plt.show()

def lc_plot_annualized_returns(returns,title,filepath):
    """
    Plots the annualized returns by Lending Club grades.

    Args:
        returns (pandas DataFrame): DataFrame with returns grouped by Lending Club Grade.
        title (string): Title to add to the plot.
        filepath (string): filepath to save image into.

    Returns:

    """
    fig = plt.figure(figsize=(30,30))
    ax2 = fig.add_subplot(1,1,1)
    ax2.bar(returns['grade'],returns['annualized_pct'])
    plt.title(title,fontsize=40,fontweight='bold')
    plt.xlabel('Grade',fontdict={'size':40})
    plt.ylabel('Annualized Return',fontdict={'size':40})
    ax2.tick_params('x',labelsize=35)
    ax2.tick_params('y',labelsize=35)
    plt.savefig(filepath)
    plt.show()


def lc_proportions_time(df):
    """
    Plots the proportion of Fully Paid Off loans vs. Charged Off loans over time.

    Args:
        df (pandas DataFrame): Lending Club data.

    Returns:

    """
    cohort_df = (df.groupby(['issue_d_year','loan_status']).sum()['funded_amnt'] / 1000000).reset_index()
    cohort_df['proportion'] = cohort_df.apply(lambda row: prop_cohort(row,cohort_df),axis=1)
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    ax.plot(cohort_df[cohort_df['loan_status'] ==1]['issue_d_year'],cohort_df[cohort_df['loan_status'] ==1]['proportion'],label='Fully Paid',c='blue')
    ax.plot(cohort_df[cohort_df['loan_status'] ==0]['issue_d_year'],cohort_df[cohort_df['loan_status'] ==0]['proportion'],label='Charged-Off ',c='orangered')
    plt.xlabel('Year',fontsize=30,fontweight='bold')
    plt.ylabel('Proportion',fontsize=30,fontweight='bold')
    plt.title('Loan Outcome Proportions',fontsize=40,fontweight='bold')
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    plt.legend(prop={'size': 25})
    plt.savefig('images/loan_outcome_proportions.png')
    plt.show()


def prop_cohort(row,cohort_df):
    """
    Returns data column specifying the proportion of the given loan class as a percentage of total loans that reached maturity.

    Args:
        row (pandas DataFrame row): DataFrame row
        cohort_df (pandas DataFrame): dataframe with only one type of loan outcome.

    Returns:
        prop (float): proportion of loans that the loan class comprised.
    """
    yr = row['issue_d_year']
    yr_amnt = cohort_df[cohort_df['issue_d_year'] == yr]['funded_amnt'].sum()
    prop =  row['funded_amnt'] / yr_amnt
    return prop

def pca_plotter(pca_df):
    """
    Plots the total variance explained by the eigenmatrix generated by PCA.

    Args:
        pca_df (pandas DataFrame): dataframe generated by performing PCA on the Lending Club dataset.

    Returns:

    """
    scaled_df = pca_df
    pca = PCA(n_components=40)
    principalComponents = pca.fit_transform(scaled_df.iloc[:,:-4])
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc' + str(i) for i in range(1,41)])
    pca_df = pd.concat([principalDf,scaled_df.iloc[:,-4:]],axis=1)
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.cumsum(pca.explained_variance_ / sum(pca.explained_variance_)),label='Cumulative Explained Variance')
    plt.title('PCA - Explained Variance',fontsize=40,fontweight='bold')
    plt.xlabel('Number of Components',fontsize=30,fontweight='bold')
    plt.ylabel('Variance',fontsize=30,fontweight='bold')
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    ax.axhline(0.95,c='r',label = '95% Variance')
    plt.legend(prop={'size':25})
    plt.savefig('images/pca_explained_variance.png')
    plt.show()




def plot_36m_returns(returns_df):
    """
    Plots the 36 month returns by proportion.

    Args:
        returns_df (pandas DataFrame): Dataframe containing the gridsearch data on each proportion.

    Returns:

    """
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax.scatter(returns_df[returns_df['model']==models[i]]['Proportions'],returns_df[returns_df['model']==models[i]]['36 Month Returns'],label=models[i],c=colors[i])
    plt.legend(prop={'size': 25})
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    plt.ylabel('Returns (%)',fontsize=30,fontweight='bold')
    plt.xlabel('Proportion of Majority to Minority Class',fontsize=30,fontweight='bold')
    plt.title('36 Month Returns by Model and Proportion',fontsize=40,fontweight='bold')
    plt.savefig('images/returns_36m.png')
    plt.show()

def plot_60m_returns(returns_df):
    """
    Plots the 60 month returns by proportion.

    Args:
        returns_df (pandas DataFrame): Dataframe containing the gridsearch data on each proportion.

    Returns:

    """
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    fig = plt.figure(figsize=(20,15))
    ax2 = fig.add_subplot(1,1,1)
    for j in range(len(models)):
        ax2.scatter(returns_df[returns_df['model']==models[j]]['Proportions'],returns_df[returns_df['model']==models[j]]['60 Month Returns'],label=models[j],c=colors[j])
    plt.legend(prop={'size': 25})
    ax2.tick_params('x',labelsize=25)
    ax2.tick_params('y',labelsize=25)
    plt.ylabel('Returns (%)',fontsize=30,fontweight='bold')
    plt.xlabel('Proportion of Majority to Minority Class',fontsize=30,fontweight='bold')
    plt.title('60 Month Returns by Model and Proportion',fontsize=40,fontweight='bold')
    plt.savefig('images/returns_60m.png')
    plt.show()

def plot_rets_v_acc(return_df):
    """
    Plots comparison of returns vs accuracy.

    Args:
        returns_df (pandas DataFrame): Dataframe containing the gridsearch data on each proportion.

    Returns:

    """
    fig = plt.figure(figsize=(20,15))
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    ax1 = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax1.scatter(return_df[return_df['model']==models[i]]['Accuracy'],return_df[return_df['model']==models[i]]['Returns'],c=colors[i],label=models[i])
    plt.legend(prop={'size': 35})
    ax1.tick_params('x',labelsize=25)
    ax1.tick_params('y',labelsize=25)
    plt.ylabel('Returns (%)',fontsize=30,fontweight='bold')
    plt.xlabel('Accuracy',fontsize=30,fontweight='bold')
    plt.title('Blended Returns vs. Accuracy',fontsize=40,fontweight='bold')
    plt.savefig('images/returns_v_acc.png')
    plt.show()

def plot_rets_v_prec(return_df):
    """
    Plots comparison of returns vs precision.

    Args:
        returns_df (pandas DataFrame): Dataframe containing the gridsearch data on each proportion.

    Returns:

    """
    fig = plt.figure(figsize=(20,15))
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    ax1 = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax1.scatter(return_df[return_df['model']==models[i]]['Accuracy'],return_df[return_df['model']==models[i]]['Returns'],c=colors[i],label=models[i])
    plt.legend(prop={'size': 35})
    ax1.tick_params('x',labelsize=25)
    ax1.tick_params('y',labelsize=25)
    plt.ylabel('Returns (%)',fontsize=30,fontweight='bold')
    plt.xlabel('Precision',fontsize=30,fontweight='bold')
    plt.title('Blended Returns vs. Precision',fontsize=40,fontweight='bold')
    plt.savefig('images/returns_v_prec')
    plt.show()




def plot_prec_by_prop(return_df):
    """
    Plots comparison of returns vs proportions.

    Args:
        returns_df (pandas DataFrame): Dataframe containing the gridsearch data on each proportion.

    Returns:

    """
    fig = plt.figure(figsize=(20,15))
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    ax = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax.scatter(return_df[return_df['model'] == models[i]]['Proportions'],return_df[return_df['model'] == models[i]]['Precision'],label=models[i],c=colors[i])
    plt.legend(prop={'size':25})
    plt.xlabel('Proportion',fontsize=30,fontweight='bold')
    plt.ylabel('Precision',fontsize=30,fontweight='bold')
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    plt.title('Precision vs. Proportion',fontsize=40,fontweight='bold')
    plt.savefig('images/prec_prop.png')

    plt.show()

def plot_36m_deployed(returns_df):
    """
    Plots 36 month returns against the capital invested in 36 month term loans.

    Args:
        returns_df (pandas dataframe): Dataframe containing the gridsearch data on each proportion.

    Returns:

    """
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax.scatter(returns_df[returns_df['model']==models[i]]['Deployed_Capital'] / 1000000,returns_df[returns_df['model']==models[i]]['36 Month Returns'],label=models[i],c=colors[i])
    plt.legend(prop={'size': 25})
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    plt.ylabel('Returns (%)',fontsize=35,fontweight='bold')
    plt.xlabel('Deployed Capital ($ in millions)',fontsize=35,fontweight='bold')
    plt.title('36 Month Returns on Deployed Capital',fontsize=40,fontweight='bold')
    plt.savefig('images/returns_36m_deployed.png')
    plt.show()

def plot_60m_deployed(returns_df):
    """
    Plots 60 month returns against the capital invested in 60 month term loans.

    Args:
        returns_df (pandas dataframe): Dataframe containing the gridsearch data on each proportion.

    Returns:

    """
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax.scatter(returns_df[returns_df['model']==models[i]]['Deployed_Capital'] / 1000000,returns_df[returns_df['model']==models[i]]['60 Month Returns'],label=models[i],c=colors[i])
    plt.legend(prop={'size': 25})
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    plt.ylabel('Returns (%)',fontsize=35,fontweight='bold')
    plt.xlabel('Deployed Capital ($ in millions)',fontsize=35,fontweight='bold')
    plt.title('60 Month Returns on Deployed Capital',fontsize=40,fontweight='bold')
    plt.savefig('images/returns_60m_deployed.png')
    plt.show()

def profits_v_deployed(returns_df):
    """
    Plots profits against the capital invested in loans.

    Args:
        returns_df (pandas dataframe): Dataframe containing the gridsearch data on each proportion.

    Returns:

    """
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax.scatter(returns_df[returns_df['model']==models[i]]['Deployed_Capital'] / 1000000,returns_df[returns_df['model']==models[i]]['Profits'] / 1000000,c=colors[i],label=models[i])
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    plt.legend(prop={'size': 25})
    plt.ylabel('Profits ($ in millions)',fontsize=30,fontweight='bold')
    plt.xlabel('Deployed Capital ($ in millions)',fontsize=30,fontweight='bold')
    plt.title('Profits By Model',fontsize=40,fontweight='bold')
    plt.savefig('images/profits_v_deployed.png')
    plt.show()

def get_list_vals(x):
    """
    Extracts values from a json style list.

    Args:
        x (string): string containg list of values.

    Returns:
        arr (np.array): numpy array that contains values from the original list and removes nan values.
    """
    arr = np.array([float(j) for j in x.strip('[]').split(',')])
    arr =  arr[~np.isnan(arr)]
    return arr

def plot_sharpe(df_original):
    """
    Creates four plots: Sharpe Ratio vs. 36 Month Deployed, Sharpe Ratio vs. 60 Month Deployed, Sharpe Ratio vs. 36 Month Returns,
    and Sharpe Ratio vs. 60 Month Returns.

    Args:
        df_original (pandas dataframe): dataframe created from the LC_Transformer sharpe_calc function.

    Returns:

    """
    df = df_original
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['b','g','r']
    df['Avg_Deployed_36'] = df['Deployed_Capital_36'].apply(lambda x: np.mean(x) /1000000)
    df['Avg_Deployed_60'] = df['Deployed_Capital_60'].apply(lambda x: np.mean(x)/1000000)
    df['Model'] = df['Model_list'].apply(lambda x: x.split('_')[0])
    fig = plt.figure(figsize=(20,20))
    """
    ax1 = fig.add_subplot(2,2,1)
    for m,c in zip(models,colors):
        ax1.scatter(df[df['Model']==m]['Avg_Deployed_36'],df[df['Model']==m]['Sharpe_36'],c=c,label=m)
    plt.title('Sharpe Ratio vs. 36 Month Deployed',fontsize=35,fontweight='bold')
    ax1.tick_params('x',labelsize=25)
    ax1.tick_params('y',labelsize=25)
    plt.xlabel('Deployed Capital in Millions',fontsize=30,fontweight='bold')
    plt.ylabel('Sharpe Ratio',fontsize=30,fontweight='bold')
    plt.legend(prop={'size': 20})
    ax2 = fig.add_subplot(2,2,2)
    for m,c in zip(models,colors):
        ax2.scatter(df[df['Model']==m]['Avg_Deployed_36'],df[df['Model']==m]['Sharpe_60'],c=c,label=m)
    plt.title('Sharpe Ratio vs. 60 Month Deployed',fontsize=40,fontweight='bold')
    ax2.tick_params('x',labelsize=25)
    ax2.tick_params('y',labelsize=25)
    plt.xlabel('Deployed Capital in Millions',fontsize=30,fontweight='bold')
    plt.ylabel('Sharpe Ratio',fontsize=30,fontweight='bold')
    plt.legend(prop={'size': 20})
    """
    ax3 = fig.add_subplot(2,1,1)
    for m,c in zip(models,colors):
        ax3.scatter(df[df['Model'] ==m]['Avg_Return_36'],df[df['Model']==m]['Sharpe_36'],c=c,label=m)
    plt.title('Sharpe Ratio vs. 36 Month Returns',fontsize=40,fontweight='bold')
    ax3.tick_params('x',labelsize=25)
    ax3.tick_params('y',labelsize=25)
    plt.xlabel('Returns',fontsize=30,fontweight='bold')
    plt.ylabel('Sharpe Ratio',fontsize=30,fontweight='bold')
    plt.legend(prop={'size': 20})
    ax4 = fig.add_subplot(2,1,2)
    for m,c in zip(models,colors):
        ax4.scatter(df[df['Model'] ==m]['Avg_Return_60'],df[df['Model']==m]['Sharpe_60'],c=c,label=m)
    plt.title('Sharpe Ratio vs. 60 Month Returns',fontsize=40,fontweight='bold')
    ax4.tick_params('x',labelsize=25)
    ax4.tick_params('y',labelsize=25)
    plt.xlabel('Returns',fontsize=30,fontweight='bold')
    plt.ylabel('Sharpe Ratio',fontsize=30,fontweight='bold')
    plt.legend(prop={'size': 20})
    plt.tight_layout()
    plt.savefig('images/sharpe_plot.png')
    plt.show()
