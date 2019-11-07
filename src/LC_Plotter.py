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


def lc_choose_proportion(df,filepath):
    output_df = df.iloc[1:,:]
    fig = plt.figure(figsize=(20,20))
    ax1 = fig.add_subplot(3,2,1)
    ax1.scatter(output_df['Proportions'],output_df['Accuracy'])
    ax1.tick_params('x',labelsize=15)
    ax1.tick_params('y',labelsize=15)
    plt.xlabel('Proportions - Fully Paid to Charged-Off',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.title('Accuracy vs Proportions',fontsize=20,fontweight='bold')
    ax2 = fig.add_subplot(3,2,2)
    ax2.scatter(output_df['Proportions'],output_df['Precision'])
    ax2.tick_params('x',labelsize=15)
    ax2.tick_params('y',labelsize=15)
    plt.ylabel('Precision',fontsize=20)
    plt.xlabel('Proportions - Fully Paid to Charged-Off',fontsize=20)
    plt.title('Precision vs Proportions',fontsize=20,fontweight='bold')
    ax3 = fig.add_subplot(3,2,3)
    ax3.scatter(output_df['Proportions'],output_df['Returns'])
    ax3.tick_params('x',labelsize=15)
    ax3.tick_params('y',labelsize=15)
    plt.xlabel('Proportions - Fully Paid to Charged-Off',fontsize=20)
    plt.ylabel('Returns',fontsize=20)
    plt.title('Returns vs Proportions',fontsize=20,fontweight='bold')
    ax4 = fig.add_subplot(3,2,4)
    ax4.scatter(output_df['Accuracy'],output_df['Returns'])
    ax4.tick_params('x',labelsize=15)
    ax4.tick_params('y',labelsize=15)
    plt.xlabel('Accuracy',fontsize=20)
    plt.ylabel('Returns',fontsize=20)
    plt.title('Returns vs Accuracy',fontsize=20,fontweight='bold')
    ax5 = fig.add_subplot(3,2,5)
    ax5.scatter(output_df['Precision'],output_df['Returns'])
    ax5.tick_params('x',labelsize=15)
    ax5.tick_params('y',labelsize=15)
    plt.xlabel('Precision',fontsize=20)
    plt.ylabel('Returns',fontsize=20)
    plt.title('Returns vs Precision',fontsize=20,fontweight='bold')
    ax6 = fig.add_subplot(3,2,6)
    ax6.hist(output_df['Returns'])
    ax6.tick_params('x',labelsize=15)
    ax6.tick_params('y',labelsize=15)
    plt.xlabel('Returns')
    plt.ylabel('Counts')
    plt.title('Distribution of Returns',fontsize=20,fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath)



def lc_defaults_iplot(X_train,y_train,X_test,y_test,test_loan_data):
    logm_stats, rfc_stats, gbc_stats, xgb_stats = LCM.lc_defaults_quick_eval(X_train,y_train,X_test,y_test,test_loan_data)
    acc = []
    prec = []
    rets = []
    count = 0
    for i,j,k,l in zip(logm_stats,rfc_stats,gbc_stats,xgb_stats):
        if count ==0:
            acc.append(i)
            acc.append(j)
            acc.append(k)
            acc.append(l)
        elif count == 1:
            prec.append(i)
            prec.append(j)
            prec.append(k)
            prec.append(l)
        else:
            rets.append(i)
            rets.append(j)
            rets.append(k)
            rets.append(l)
        count += 1
    fig = plt.figure(figsize = (40,40))
    ax = fig.add_subplot(1,1,1)
    ax.bar(height=acc,width=0.2,color='green')
    ax.bar(height=prec,width=0.2,color='blue')
    ax.bar(height=rets, width=0.2,color='red')
    plt.show()




def lc_ml(logm,rfc, gbc,xgb,filename,title):
    logm_stats = list(logm).copy()
    logm_stats[-1] = round((logm_stats[-1] / 100) + 1,3)
    rfc_stats = list(rfc).copy()
    rfc_stats[-1] = round((rfc_stats[-1] / 100) + 1,3)
    gbc_stats = list(gbc).copy()
    gbc_stats[-1] = round((gbc_stats[-1] / 100) + 1,3)
    xgb_stats = list(xgb).copy()
    xgb_stats[-1] = round((xgb_stats[-1] / 100) + 1,3)
    data = [
    go.Scatterpolar(
        mode='lines+markers',
      r = xgb_stats,
      theta = ['XGB Acc.','XGB Prec.','XGB Ret.'],
      fill = 'toself',
      name = '',
        line = dict(
        color = "#63AF63"
      ),
      marker = dict(
        color = "#B3FFB3",
        symbol = "square",
        size = 12
      ),
      subplot = "polar",
    ),
        go.Scatterpolar(
        mode='lines+markers',
      r = logm_stats,
      theta = ['LogReg Acc.','LogReg Prec.','LogReg Ret.'],
      fill = 'toself',
      name = '',
        line = dict(
        color = "#63AF63"
      ),
      marker = dict(
        color = "#B3FFB3",
        symbol = "square",
        size = 12
      ),
      subplot = "polar3",
    ),
        go.Scatterpolar(
        mode='lines+markers',
      r = rfc_stats,
      theta = ['RFC Acc.','RFC Prec.','RFC Ret.'],
      fill = 'toself',
      name = '',
        line = dict(
        color = "#63AF63"
      ),
      marker = dict(
        color = "#B3FFB3",
        symbol = "square",
        size = 12
      ),
      subplot = "polar2",
    ),
    go.Scatterpolar(
        mode='lines+markers',
      r = gbc_stats,
      theta = ['GBC Acc.','GBC Prec.','GBC Ret.'],
      fill = 'toself',
      name = '',
        line = dict(
        color = "#63AF63"
      ),
      marker = dict(
        color = "#B3FFB3",
        symbol = "square",
        size = 12
      ),
      subplot = "polar4",
    )
    ]

    layout = go.Layout(
        title="",
        showlegend = False,
         paper_bgcolor = "rgb(255, 248, 243)",
        polar = dict(
          domain = dict(
            x = [0.6,1.0],
            y = [0.0,0.4]
          ),
          radialaxis = dict(
            tickfont = dict(
              size = 8
            )
          ),
          angularaxis = dict(
            tickfont = dict(
              size = 12
            ),
            rotation = 90,
            direction = "counterclockwise"
          )
        ),
        polar2 = dict(
          domain = dict(
            x = [0.6,1],
            y = [0.6,1]
          ),
          radialaxis = dict(
            tickfont = dict(
              size = 8
            )
          ),
          angularaxis = dict(
            tickfont = dict(
              size = 12
            ),
            rotation = 90,
            direction = "counterclockwise"
          ),
        ),
        polar3 = dict(
          domain = dict(
            x = [0.0,0.4],
            y = [0.6,1.0]
          ),
          radialaxis = dict(
            tickfont = dict(
              size = 8
            )
          ),
          angularaxis = dict(
            tickfont = dict(
              size = 12
            ),
            rotation = 90,
            direction = "counterclockwise"
          )
        ),
        polar4 = dict(
          domain = dict(
            x = [0.0,0.4],
            y = [0.0,0.4]
          ),
          radialaxis = dict(
            tickfont = dict(
              size = 8

            )
          ),
          angularaxis = dict(
            tickfont = dict(
              size = 12
            ),
            rotation = 90,
            direction = "counterclockwise"
          )
        )

    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(title_text=title)
    plotly.offline.plot(fig,filename='random_ml_models',auto_open=False)
    iplot(fig,image_width=1000,image_height=1000,filename=filename,image='png')



def lc_returns_vs_thresholds(models, X_test, y_test,loan_test_data,filepath):
    ret_list = []
    for model in models:
        x, rets = LCM.lc_predict_probas_evaluator(model, X_test, y_test,loan_test_data)
        ret_list.append(rets)
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,ret_list[0],label='Logistic Regression')
    ax.plot(x,ret_list[1],label='Random Forest')
    ax.plot(x,ret_list[2],label='Gradient Boosting')
    ax.plot(x,ret_list[3],label='XGradient Boosting')
    plt.title('Returns vs. Threshold',fontsize=20,fontweight='bold')
    plt.xlabel('Thresholds',fontsize=20)
    plt.ylabel('Returns',fontsize=20)
    ax.tick_params('x',labelsize=20)
    ax.tick_params('y',labelsize=20)
    plt.savefig(filepath)
    plt.legend(loc='lower right',prop={'size': 25})
    plt.show()


def lc_proportions_time(df):
    cohort_df = (df.groupby(['issue_d_year','loan_status']).sum()['funded_amnt'] / 1000000).reset_index()
    cohort_df['proportion'] = cohort_df.apply(lambda row: prop_cohort(row,cohort_df),axis=1)
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(cohort_df[cohort_df['loan_status'] ==1]['issue_d_year'],cohort_df[cohort_df['loan_status'] ==1]['proportion'],label='Fully Paid',c='blue')
    ax.plot(cohort_df[cohort_df['loan_status'] ==0]['issue_d_year'],cohort_df[cohort_df['loan_status'] ==0]['proportion'],label='Charged-Off ',c='orangered')
    plt.xlabel('Year',fontsize=15)
    plt.ylabel('Proportion',fontsize=15)
    plt.title('Loan Outcome Proportions')
    ax.tick_params('x',labelsize=15)
    ax.tick_params('y',labelsize=15)
    plt.legend()
    plt.savefig('images/loan_outcome_proportions.png')
    plt.show()


def prop_cohort(row,cohort_df):
    yr = row['issue_d_year']
    yr_amnt = cohort_df[cohort_df['issue_d_year'] == yr]['funded_amnt'].sum()
    return row['funded_amnt'] / yr_amnt

def pca_plotter(pca_df):
    scaled_df = pca_df
    pca = PCA(n_components=40)
    principalComponents = pca.fit_transform(scaled_df.iloc[:,:-4])
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pc' + str(i) for i in range(1,41)])
    pca_df = pd.concat([principalDf,scaled_df.iloc[:,-4:]],axis=1)
    plt.plot(np.cumsum(pca.explained_variance_ / sum(pca.explained_variance_)))
    plt.title('PCA - Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance')
    plt.axhline(0.95,c='r',label = '95% Variance')
    plt.savefig('images/pca_explained_variance.png')
    plt.show()




def plot_36m_returns(returns_df):
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax.scatter(returns_df[returns_df['model']==models[i]]['Proportions'],returns_df[returns_df['model']==models[i]]['36 Month Returns'],label=models[i],c=colors[i])
    plt.legend(prop={'size': 25})
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    plt.ylabel('Returns (%)',fontsize=25)
    plt.xlabel('Proportion of Majority to Minority Class',fontsize=25)
    plt.title('36 Month Returns by Model and Proportion',fontsize=35,fontweight='bold')
    plt.savefig('images/returns_36m.png')
    plt.show()

def plot_60m_returns(returns_df):
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    fig = plt.figure(figsize=(20,15))
    ax2 = fig.add_subplot(1,1,1)
    for j in range(len(models)):
        ax2.scatter(returns_df[returns_df['model']==models[j]]['Proportions'],returns_df[returns_df['model']==models[j]]['60 Month Returns'],label=models[j],c=colors[j])
    plt.legend(prop={'size': 25})
    ax2.tick_params('x',labelsize=25)
    ax2.tick_params('y',labelsize=25)
    plt.ylabel('Returns (%)',fontsize=25)
    plt.xlabel('Proportion of Majority to Minority Class',fontsize=25)
    plt.title('60 Month Returns by Model and Proportion',fontsize=35,fontweight='bold')
    plt.savefig('images/returns_60m.png')
    plt.show()

def plot_rets_by_score(return_df):
    fig = plt.figure(figsize=(35,20))
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    ax1 = fig.add_subplot(2,1,1)
    for i in range(len(models)):
        ax1.scatter(return_df[return_df['model']==models[i]]['Accuracy'],return_df[return_df['model']==models[i]]['Returns'],c=colors[i],label=models[i])
    plt.legend(prop={'size': 35})
    ax1.tick_params('x',labelsize=30)
    ax1.tick_params('y',labelsize=30)
    plt.ylabel('Returns (%)',fontsize=35)
    plt.xlabel('Accuracy',fontsize=35)
    plt.title('Blended Returns vs. Accuracy',fontsize=40,fontweight='bold')
    ax2 = fig.add_subplot(2,1,2)
    for j in range(len(models)):
        ax2.scatter(return_df[return_df['model']==models[j]]['Precision'],return_df[return_df['model']==models[j]]['Returns'],c=colors[j],label=models[j])
    plt.legend(prop={'size': 35})
    ax2.tick_params('x',labelsize=30)
    ax2.tick_params('y',labelsize=30)
    plt.ylabel('Returns (%)',fontsize=35)
    plt.xlabel('Precision',fontsize=35)
    plt.title('Blended Returns vs. Precision',fontsize=40,fontweight='bold')
    plt.savefig('images/rets_scoring_metrics.png')
    plt.tight_layout()
    plt.show()

def plot_prec_by_prop(return_df):
    fig = plt.figure(figsize=(20,15))
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    ax = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax.scatter(return_df[return_df['model'] == models[i]]['Proportions'],return_df[return_df['model'] == models[i]]['Precision'],label=models[i],c=colors[i])
    plt.legend(prop={'size':25})
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    plt.savefig('images/prec_v_prop.png')
    plt.title('Precision vs. Proportion',fontsize=25,fontweight='bold')
    plt.show()

def plot_36m_deployed(returns_df):
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax.scatter(returns_df[returns_df['model']==models[i]]['Deployed_Capital'] / 1000000,returns_df[returns_df['model']==models[i]]['36 Month Returns'],label=models[i],c=colors[i])
    plt.legend(prop={'size': 25})
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    plt.ylabel('Returns (%)',fontsize=25)
    plt.xlabel('Deployed Capital ($ in millions)',fontsize=25)
    plt.title('36 Month Returns by Model',fontsize=35,fontweight='bold')
    plt.savefig('images/returns_36m_deployed.png')
    plt.show()

def plot_60m_deployed(returns_df):
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax.scatter(returns_df[returns_df['model']==models[i]]['Deployed_Capital'] / 1000000,returns_df[returns_df['model']==models[i]]['60 Month Returns'],label=models[i],c=colors[i])
    plt.legend(prop={'size': 25})
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    plt.ylabel('Returns (%)',fontsize=25)
    plt.xlabel('Deployed Capital ($ in millions)',fontsize=25)
    plt.title('60 Month Returns by Model',fontsize=35,fontweight='bold')
    plt.savefig('images/returns_60m_deployed.png')
    plt.show()

def profits_v_deployed(returns_df):
    models = ['LogisticRegression','RandomForestClassifier','GradientBoostingClassifier']
    colors = ['blue','green','red']
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1,1,1)
    for i in range(len(models)):
        ax.scatter(returns_df[returns_df['model']==models[i]]['Deployed_Capital'] / 1000000,returns_df[returns_df['model']==models[i]]['Profits'] / 1000000,c=colors[i],label=models[i])
    ax.tick_params('x',labelsize=25)
    ax.tick_params('y',labelsize=25)
    plt.legend(prop={'size': 25})
    plt.ylabel('Profits ($ in millions)',fontsize=25)
    plt.xlabel('Deployed Capital ($ in millions)',fontsize=25)
    plt.title('60 Month Returns by Model',fontsize=35,fontweight='bold')
    plt.savefig('images/profits_v_deployed.png')
    plt.show()
