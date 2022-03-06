import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

filename = 'german.data'
credit_df = pd.read_csv( "german.data", delim_whitespace = True, header = None)

#credit_df.info()
credit_df.columns=['account_bal','duration','payment_status','purpose',
                   'credit_amount','savings_bond_value','employed_since',
                   'installment_rate','sex_marital','guarantor','residence_since',
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
#print(good_bad_per)
#plt.pie(good_bad_per,labels=['Good loans', 'Bad loans'], autopct='%1.0f%%', startangle=90)
#plt.title('Percentage of good and bad loans')
credit_df['credit_amount']=np.log(credit_df['credit_amount'])

#print(credit_df[['credit_amount','duration','age']].describe())

# #fig, axes = plt.subplots(1,3, figsize=(16,8))
# plt.suptitle('Histogram of continuous variables')
# axes[0].hist(credit_df['duration'])
# axes[0].set_xlabel('No. of observations')
# axes[0].set_ylabel('Years')
# axes[0].set_title('Histogram of loan duration');
#
# axes[1].hist(credit_df['credit_amount'])
# axes[1].set_xlabel('No. of observations')
# axes[1].set_ylabel('Credit amount (dollars)')
# axes[1].set_title('Histogram of Credit amount');
#
# axes[2].hist(credit_df['age'])
# axes[2].set_xlabel('No. of observations')
# axes[2].set_ylabel('Age')
# axes[2].set_title('Histogram of Age');
#
# #plt.show()
#
# # box-plots of continues variables
#
# fig, ax = plt.subplots(1,3,figsize=(20,5))
# plt.suptitle('BOX PLOTS')
# sns.boxplot(credit_df['credit_amount'], ax=ax[0]);
# sns.boxplot(credit_df['duration'], ax=ax[1], color='salmon');
# sns.boxplot(credit_df['age'], ax=ax[2], color='darkviolet');
#plt.show()

#sns.scatterplot(y=credit_df.credit_amount,x=credit_df.duration, hue=credit_df.target, s=100)
#plt.show()

#credit_df.groupby('job')['target'].value_counts().unstack(level=1).plot.barh(stacked=True, figsize=(10, 6))

#sns.lineplot(data=credit_df, x='duration', y='credit_amount', hue='target', palette='deep');

#credit_df.groupby('most_valuable_asset')['target'].value_counts().unstack(level=1).plot.barh(stacked=True, figsize=(10, 6))

#sns.scatterplot(y=credit_df.credit_amount, x=credit_df.most_valuable_asset, hue=credit_df.target, s=100);
#plt.show()

# Number of unique classes in each object column
print(credit_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0))

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in credit_df:
    if credit_df[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(credit_df[col].unique())) <= 2:
            # Train on the training data
            le.fit(credit_df[col])
            # Transform both training and testing data
            credit_df[col] = le.transform(credit_df[col])

            # Keep track of how many columns were label encoded
            le_count += 1

#print('%d columns were label encoded.' % le_count)

credit_df = pd.get_dummies(credit_df)

#print('Encoded Features shape: ', credit_df.shape)

# Find correlations with the target and sort
correlations = credit_df.corr()['target'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

# Extract the significantly correlated variables
#corr_data = credit_df[['duration','credit_amount',
                   'installment_rate','residence_since',
                   'age','number_of_existcr','number_of_dependents','telephone',
                   'foreign','target']]
corr_data_corrs = corr_data.corr()


# Heatmap of correlations
sns.heatmap(corr_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
plt.show()

