import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

filename = 'german.data'
credit_df = pd.read_csv( "german.data", delim_whitespace = True, header = None)
# columns = ['checkin_acc', 'duration', 'credit_history', 'purpose', 'amount',
#            'saving_acc', 'present_emp_since', 'inst_rate', 'personal_status',
#            'other_debtors', 'residing_since', 'property', 'age',
#            'inst_plans', 'housing', 'num_credits',
#            'job', 'dependents', 'telephone', 'foreign_worker', 'status']
#
# credit_df.columns = columns
# credit_df.target_names = ['Good', 'Bad']
# X_features = list( credit_df.columns )
# credit_df_complete = pd.get_dummies( credit_df[X_features], drop_first = True )
# print(credit_df.info())
# print(X_features)
# credit_df_complete = pd.get_dummies( credit_df[X_features], drop_first = True )
# print(credit_df_complete.head())
#
# import seaborn as sns
# import matplotlib
# import matplotlib.pyplot as plt
#
# sns.set_style('darkgrid')
# matplotlib.rcParams['font.size'] = 14
# matplotlib.rcParams['figure.figsize'] = (9, 5)
# matplotlib.rcParams['figure.facecolor'] = '#00000000'
# print(credit_df_complete.isna().any().any())
# credit_df_complete.status.unique()
#
#
# from sklearn.preprocessing import LabelEncoder
#
# le = LabelEncoder()
# le.fit(credit_df_complete.status)
# credit_df_complete = le.transform(credit_df_complete.status)
# print(credit_df_complete.status.head(5))

credit_df.info()
credit_df.columns=['account_bal','duration','payment_status','purpose',
                   'credit_amount','savings_bond_value','employed_since',
                   'intallment_rate','sex_marital','guarantor','residence_since',
                   'most_valuable_asset','age','concurrent_credits','type_of_housing',
                   'number_of_existcr','job','number_of_dependents','telephone',
                   'foreign','target']

credit_df= credit_df.replace(['A11','A12','A13','A14', 'A171','A172','A173','A174','A121','A122','A123','A124'],
                  ['neg_bal','positive_bal','positive_bal','no_acc','unskilled','unskilled','skilled','highly_skilled',
                   'none','car','life_insurance','real_estate'])


# import libraries for visualizations

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

credit_df.isna().any().any()

credit_df.target.unique()

from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()
le.fit(credit_df.target)
credit_df.target=le.transform(credit_df.target)
credit_df.target.head(5)

good_bad_per=round(((credit_df.target.value_counts()/credit_df.target.count())*100))
good_bad_per
print(plt.pie(good_bad_per,labels=['Good loans', 'Bad loans'], autopct='%1.0f%%', startangle=90))
plt.title('Percentage of good and bad loans')








