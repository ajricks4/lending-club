# Lending Club Loan Analysis

## Overview

Lending Club is a peer-to-peer lending company founded in 2007 by John Donovan and Renaud Laplanche. The Lending Club platform allows borrowers to create unsecured personal loans between $1,000 and $40,000. Investors earn money by selecting loans to fund and Lending Club earns money by charging borrowers an origination fee and investors a service fee. From Q2'2007 - Q2'2019, the Lending Club platform originated ~ $38.1 billion in loans, most recently originating ~ $2.2B in Q2'19. 

## Data 

Lending Club has made their data public [here](https://help.lendingclub.com/hc/en-us/articles/216127307-Data-Dictionaries). The dataset spans ~ 2.5 million loans across 150 features and includes a data dictionary to refer to each term. 

## A First Look

#### Historical Performance

Because we have loans dating back to 2007, we can start by viewing a time series of Lending Club's loans originated per month. The below plot also overlays a plot of the cumulative loan growth as well as Lending Club's stock (NYSE:LC) to see whether loan growth has translated into market capitalization. The below plot also shows (red marker) the date at which Lending Club IPO'ed. 

<img src="./images/lc_time_series.png"/>



Post-IPO, Lending CLub's strongest month was Mar'16 when it originated $949.4M in loans on its platform. Since IPO, Lending Club has posted strong yearly loan origination numbers as seen below:
<p align="center">
  
|Year|Amount|
|:---|:-----|
|2015|$6.4B|
|2016|$6.4B|
|2017|$6.6B|
|2018|$7.9B|
|2019|$4.1B|

</p>

#### Understanding the Loan Mix

The loans on Lending Club are classified in 5 categories: Fully Paid, Current, Charged Off, Late (31-120 days), In Grace Period, Late (16-30 days), Does not meet the credit policy. Status:Fully Paid, Does not meet the credit policy. Status:Charged Off, Default. Below, we have grouped up the loan categories including late, in grace period and in default to show overall, what the Lending Club loan platform has originated.

<p align="left">
<img src="./images/lc_loan_status.png" width = 7500 />
</p>

As we can see here, ~85% of Lending Club loans fall under the Fully Paid or Current category, indicating that a sizable chunk of the loans available to investors are performing. To further break these loans down, we can analyze how Lending Club grades loans. 

Lending Club, at loan origination, gives a loan a grade of A to G and a subgrade of 1 to 5 indicating the financial rating of the borrower based on a proprietary model. The model outputs a number ranging from 1 - 25 which is then translated to a grade and a subgrade. For example, a model rank of 6 would correspond to B1. To dive further into how Lending Club loans perform by grade, we have the below plot:

<img src="./images/lc_grade_pie.png"/>

The exact $ (expressed in millions )values are as follows:
<p align="center">
  
| Grade| Principal Paid Off| Outstanding| Bad Debt|
|:-----|:------------------|:-----------|:--------|
| A    | $5,050.3M         | $2,440.4M  | $146.6M |
| B    | $7,196.7M         | $2,884.8M  | $553.5M |
| C    |$6,896.6M          |$2,801.4M   |$1,047.8M|
| D    |$3,399.7M          |$1,415.8M   |$838.5M  |
| E    |$1,518.0| $338.7M | $561.5M|
| F    |$469.9M|$79.4M|$250.6M|
| G    |$135.3M|$25.8M|$87.1M|

</p>

Speficially, we can see that Lending Club does a good job rating its loans, with each consecutive grade slightly producing more bad debt than the previous grade. 

#### Geographies

To get a sense of how Lending Club's loans are dispersed throughout the U.S., we can view the two below plots:
(Please note that the plots below are .png file representations of an interactive .html file. The html files are saved in the images repository and can be re-created using code found in the src repository)

<img src="./images/state_loan_originations.png"/>
<img src="./images/state_loan_count.png"/>

The two plots confirm what may have already been suspected. The largest loan volume is coming from CA, TX, NY, and FL. However, when we look average loan size, we find the following:
<p align="center">
  
|State| Avg Loan Size|
|-----|--------------|
|CA| $15.5K |
|FL| $14.6K|
|NY| $15.0K|
|TX| $15.9K|

</p>
Which are considerably less than AK, with an avg loan size of $17.4K.

#### Individuals

Now looking at the profile of individual borrowers, we produce the following plots:

<img src="./images/borrower_profile.png"/>

From this, we can conclude the following points:
-  Loan amounts spike around large round numbers and have more density in the < $20,000.00 range
-  The FICO distribution of funded loans has a clear cutoff at ~650 and is right skewed with a large mass less than 700 (considered as the cutoff between a good vs mediocre credit score)
-  A large % of the borrowers have been at their job for a long time, as represented by the large column at the 10 tick. The data lumps in any worker who has worked over 10 years at their current role with those that have been at their current role for exactly 10 years.
-  Almost half of all borrowers have a mortgage with another ~ 40% renting. Only 11.2% own their property, meaning that most borrowers in the data set will have a monthly obligation to rent or a mortgage which will have a large effect on underwriting and credit risk rating.




## Evaluating Returns

grade_returns.png
grade_annualized_returns.png

Because Lending Club asigns grades to their loans, we can evaluate the returns by grade on both an actual return and an annualized return basis. Also, because the loans can only take two different term lengths, 36 or 60 months, we can further divide the grades by term length. We hypothesize that the highest returns should be earned by 60 month rated B-D loans as these will have a greater return compared to their respective 36 month counterparts to compensate for the additional holding time required. Also, we expect these loans to outperform others by grade because A loans will have lower rates of return because they are considered the safest and the loans rated E-G will be expected to have high chargeoff rates, diminishing returns. 

<img src="./images/grade_returns.png"/>

<img src="./images/grade_annualized_returns.png"/>

From the two plots, we can see that the highest performing group is the 36 month term B loans with a return of 5.7% and an annualized return of 1.9%. Most surprisingly, we find that the rate of return of Lending Club loans by grade and term length are extremely low compared to the risk in investing in unsecured peer-to-peer loans. We will benchmark our algorithms against these rates of returns:

|Grade| Term | Return | Annualized Return|
|-----|------|--------|------------------|
|A|36|5.2%|1.7%|
|A|60|3.3%|0.7%|
|B|36|5.7%|1.9%|
|B|60|3.5%|0.7%|
|C|36|4.0%|1.3%|
|C|60|2.5%|0.5%|
|D|36|2.0%|0.7%|
|D|60|1.4%|0.3%|
|E|36|0.0%|0.0%|
|E|60|1.2%|0.2%|
|F|36|-1.2%|-0.4%|
|F|60|-1.7%|-0.3%|
|G|36|-8.8%|-3.0%|
|G|60|-5.3%|-1.1%|

## Data Cleaning / Transformation

The data cleaning process included the following:
-  Removing the loans that have not yet paid off / charged-off. We only want to look at the loans that have completed their maturity so that we can see how a loan performed over the course of the term. Including loans that are currently being paid off would throw off any algorithms as loans that may charge-off in the future may still presently be current.

-  Cleaning Features: The majority of this work involved dropping features that would not have been available at the time of loan underwriting as well as converting variables to usable numerical types, whether in the form of scaled numerical features or as a one-hot encoded feature for categorical data. Further, because applications can take on the categories: Individual or Joint, we need to adjust other features that are dependent on this category. 

The initial cleaned data set yielded the following features: 
-  Issue year: (Numerical) Year loan was issued
-  Grade: (Categorical) Lending Club grade assigned to the loan
-  Subgrade: (Categorical) Lending Club sub-grade assigned to the loan
-  Funded amount: (Numerical) Loan amount funded
-  Term: (Categorical) Term length of loan, can only take on values of either 36 months or 60 months
-  Interest rate: Interest rate applied to the loan
-  Installment: (Numerical) Monthly payment if the loan is funded
-  Purpose: (Categorical) Purpose of the loan, includes credit card consolidation, deb
-  Application type: (Categorical) Type of application, can take on either Individual or Joint Application
-  FICO: (Numerical) FICO score of borrower, created by averaging the low and high range of the borrower's FICO scores
-  DTI: (Numerical) Debt to Income Ratio of the borrower based on the monthly debt payments charged to the borrower versus -  monthly income
-  Annual income: (Numerical) Annual salary of borrower
-  Employment length: (Numerical) Years in current role
-  Home ownership: (Categorical) Indicates whether a borrower owns a home, is paying off a mortgage, rents or has some other living situation
-  Address state: (Categorical) Indicates state the borrower is applying from
-  Earliest credit line: (Numerical) The earliest year the borrower had a credit line 
-  Negative activity: (Numerical) Combination of counts of public record bankruptics and other credit adverse events
-  Inquiries within the last 6 months: (Numerical) Count of times borrower's credit report was inquired upon within the last 6 months
-  Delinquencies within the last 6 months: (Numerical) Count of delinquencies within the last 6 months
-  Verified: (Categorical) Indicates whether the borrower's credit information was Verified, Source Verified, or Not Verified
-  Open accounts: (Numerical) Count of open accounts the borrower has
-  Mortgage Accounts: (Numerical) Count of open mortgage accounts the borrower has
-  Total Current Balance: (Numerical) Total outstanding credit of borrower
-  Revolving: (Numerical) Measures total revolving credit balance
-  Revolving Utilization: (Numerical) Percentage measuring 
-  Loan Status: (Categorical) Indicates whether loan was Fully Paid or Charged-Off
-  Total Payment: (Numerical) Sum of payments borrower made on credit

Note that the Loan Status feature is picked as our target variable and the Total Payment feature was kept in the dataset so as to calculate investing returns after training an algorithm.

To transform the data, we apply Sklearn's StandardScaler and OneHotEncoder on the numerical and categorical features respectively. Now it is time to start Machine Learning!

## Training Models

####  Setting up the Training / Testing Split

Our data is split roughly 80 / 20 amongst Fully Paid and Charged-Off loans. We want to start by observing how a basic model will predict with the proportion of the training set balanced and altered. Because we have the historical loan amounts and payoffs as well, we can see how changing the balance split of the target variable affects how our model generates returns. 

#### Evaluating Target Variable Proportion / Scoring Measures

Before we begin building a model, we must note the goal: Generate the highest amount of returns for a hypothetical loan portfolio. After running a series of proportions using a RandomForestClassifier with the default parameters, we produce the following:

<img src="./images/proportion_eval.png"/>


From the chart, we can see that optimizing for a higher accuracy actually is causing our returns to decrease and that returns are rising with a higher precision. This makes sense because it is more important that we reduce our False Positives (Predicting a loan will payoff when in reality it charged-off) even at the expense of converting some of our True Positives (Predicting correctly a loan will payoff) to False Negatives (Predicting a loan will charge-off when in reality the loan will payoff). Therefore, we will choose to focus on optimizing our precision / returns moving forward. 

Also from the charts, it appears that having a greated proportion of charged-off loans in our set improves our model returns. This may be because the model will experience more charge-offs and thus predict negatively more often, thereby increasing our precision and returns. Moving forward, we will choose the proportion that seemed to do best: 0.5, that is to say we will have twice as many Charged-Off loans as Fully Paid loans in our training set.

## Model Evaluation

#### Logistic Regression

Results:
-  Best Parameters:
  -  C: 8
  -  solver: liblinear
  -  penalty: l2
-  Precision: 93.0%
-  Accuracy: 42.9%
-  Returns: 6.6%

#### RandomForestClassifier

Results:
-  max_depth: 10 
-  max_features: sqrt 
-  

#### GradientBoostingClassifier

Results:

#### XGradientBoostingClassifier

Hyperparameters:
-  
-  

Results:

#### Neural Net

