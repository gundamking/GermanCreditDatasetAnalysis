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
#corr_data = credit_df[['duration','credit_amount','installment_rate','residence_since','age','number_of_existcr','number_of_dependents','telephone','foreign','target']]
#corr_data_corrs = corr_data.corr()


# Heatmap of correlations
#sns.heatmap(corr_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
#plt.title('Correlation Heatmap');
#plt.show()


# Extract the significantly correlated variables
corr_data = credit_df[['target', 'account_bal_neg_bal','duration','account_bal_no_acc']]
corr_data_corrs = corr_data.corr()


# Heatmap of correlations
#sns.heatmap(corr_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
#plt.title('Correlation Heatmap');
#plt.show()


# Make a new dataframe for polynomial features
poly_features = credit_df[['duration', 'account_bal_neg_bal', 'account_bal_no_acc']]
poly_target = credit_df['target']

from sklearn.preprocessing import PolynomialFeatures

# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree=2)
# Train the polynomial features
poly_transformer.fit(poly_features)

# Transform the features
poly_features = poly_transformer.transform(poly_features)
print('Polynomial Features shape: ', poly_features.shape)

print(poly_transformer.get_feature_names(input_features = ['duration','account_bal_neg_bal','account_bal_no_acc']))

# Create a dataframe for polynomial features
poly_features = pd.DataFrame(
    poly_features, columns = poly_transformer.get_feature_names(
        ['duration','account_bal_neg_bal','account_bal_no_acc']))

# Add in the target
poly_features['target'] = poly_target

# Find the correlations with the target
poly_corrs = poly_features.corr()['target'].sort_values()

# Display the correlations
print(poly_corrs)

print(list(poly_features))

#deleting duplicate columns in poly_features

for i in list(poly_features.columns):
  for j in list(credit_df.columns):
    if (i==j):
      poly_features.drop(labels=i, axis=1, inplace=True)

poly_features.drop(labels='1', axis=1, inplace=True)
print(list(poly_features))


from sklearn. model_selection import train_test_split
x, y = credit_df.drop('target', axis=1), credit_df['target']
print(x.shape, y.shape)

x_train, x_test, y_train,y_test= train_test_split(x,y, test_size=.2, random_state=42)
print(x_train.shape, x_test.shape)
# Let's normalize the features to prevent undue influence in the model.

from sklearn.preprocessing import MinMaxScaler

# scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# fit on features dataset
scaler.fit(x_train)
scaler.fit(x_test)
x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)


y.value_counts(normalize=True)

# import packages, functions, and classes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate

# prepare models
models = []
models.append(('DT', DecisionTreeClassifier(random_state=42)))
models.append(('LR', LogisticRegression(random_state=42)))
models.append(('RF', RandomForestClassifier(random_state=42)))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC(gamma='auto', random_state=42)))

models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))


# evaluate each model in turn
results_recall = []
results_roc_auc = []
names = []
# recall= tp/ (tp+fn). Best value=1, worst value=0
scoring = ['recall', 'roc_auc']

for name, model in models:
    # split dataset into k folds. use one fold for validation and remaining k-1 folds for training
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # Evaluate a score by cross-validation. Returns array of scores of the model for each run of the cross validation.
    # cv_results = cross_val_score(model, x_train, y_train, cv=skf, scoring=scoring)
    cv_results = cross_validate(model, x_train, y_train, cv=skf, scoring=scoring)
    results_recall.append(cv_results['test_recall'])
    results_roc_auc.append(cv_results['test_roc_auc'])
    names.append(name)
    msg = "%s- recall:%f roc_auc:%f" % (name, cv_results['test_recall'].mean(), cv_results['test_roc_auc'].mean())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(11, 6))
fig.suptitle('Recall scoring Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_recall, showmeans=True)
ax.set_xticklabels(names)
#plt.show();

fig = plt.figure(figsize=(11, 6))
fig.suptitle('AUC scoring Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_roc_auc, showmeans=True)
ax.set_xticklabels(names)
#plt.show();

from sklearn.tree import DecisionTreeClassifier

# initialize a tree
tree= DecisionTreeClassifier(random_state=42)

# fit model
tree.fit(x_train, y_train)

# predict
pred_test= tree.predict(x_test)

pred_test.shape, y_test.shape

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_test, pred_test)

from sklearn.model_selection import StratifiedKFold, cross_val_score

skf= StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

val_scores= cross_val_score(estimator=tree, X=x_train, y=y_train, cv=skf)

val_scores

val_scores.mean()

# to display progress of a loop

# fit model for different values of max_depth without using GridSearchCV

cv_accuracies_by_depth, test_accuracies_by_depth = [], []
max_depth_values = np.arange(2, 11)

from tqdm import notebook
# for each value of max_depth
for curr_max_depth in notebook.tqdm(max_depth_values):
    tree = DecisionTreeClassifier(random_state=42, max_depth=curr_max_depth)

    # perform cross-validation
    val_scores = cross_val_score(estimator=tree, X=x_train, y=y_train, cv=skf)
    cv_accuracies_by_depth.append(val_scores.mean())

    # assess the model with a test test
    tree.fit(x_train, y_train)
    curr_pred = tree.predict(x_test)
    test_accuracies_by_depth.append(accuracy_score(curr_pred, y_test))


# validation curve
plt.plot(max_depth_values, cv_accuracies_by_depth, label='cv')
plt.plot(max_depth_values, test_accuracies_by_depth, label='test')
plt.legend()
plt.xlabel('max depth')
plt.ylabel('accuracies')
plt.title('Decision-Tree validation curve for max_depth');

# fit tree
tree= DecisionTreeClassifier( random_state=42, max_depth=3).fit(x_train, y_train)

# generate graph locally

from io import StringIO
import pydotplus
from ipywidgets import Image
from sklearn.tree import export_graphviz
dot_data= StringIO()

export_graphviz(decision_tree= tree, out_file=dot_data,
                filled= True, feature_names=x.columns)
graph= pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(value=graph.create_png())

##Logistic Regression

tuned_models_test=[]
tuned_models_train=[]

# Create the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001, random_state=42)

# Train on the training data
log_reg.fit(x_train, y_train)

# Evaluate on test dataset
recall_test= recall_score(y_test,log_reg.predict(x_test))
roc_test=roc_auc_score(y_test,log_reg.predict_proba(x_test)[:, 1])
print('LR',' recall_test:', round(recall_test,2),' auc_roc_test:', round(roc_test,2))
tuned_models_test.append(('LR',' recall_test:', round(recall_test,2),' auc_roc_test:', round(roc_test,2)))

# Evaluate on train dataset
roc_train= cross_val_score(log_reg, x_train, y_train, cv=skf, scoring='roc_auc').mean()
recall_train= cross_val_score(log_reg, x_train, y_train, cv=skf, scoring='recall').mean()
print('LR',' recall_train:', round(recall_train,2),' auc_roc_train:', round(roc_train,2))
tuned_models_train.append(('LR',' recall_train:', round(recall_train,2),' auc_roc_train:', round(roc_train,2)))
print(classification_report(y_test, log_reg.predict(x_test)))

##Random Forest

# create model with default parameters- baseline
rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)

# Train it on the training set
cv_result_baseline= cross_val_score(rf_baseline, x_test, y_test, cv=skf)

# Evalute the results (cross-val)
print("CV accuracy score: {:.2f}%".format(cv_result_baseline.mean() * 100))

# train model
rf_baseline.fit(x_train, y_train)

pred_test_rf= rf_baseline.predict(x_test)
print("Test accuracy score: {:.2f}%".format((accuracy_score(pred_test_rf, y_test) * 100)))


# Create lists to save the values of accuracy on training and test sets
train_acc = []
test_acc = []
temp_train_acc
trees_grid = [5, 10, 15, 20, 30, 50, 75, 100]

train_acc, test_acc


print("Best CV accuracy is {:.2f}% with {} trees".format(max(test_acc)*100,
                                                        trees_grid[np.argmax(test_acc)]))


plt.plot(trees_grid, train_acc, label='train')
plt.plot(trees_grid, test_acc, label='test')
plt.legend()
plt.xlabel('No. of trees (n_estimators)')
plt.ylabel('Accuracies')
plt.title('Random-Forest: accuracy vs n_estimators');


# Create lists to save the values of accuracy on training and test sets
train_acc = []
test_acc = []
temp_train_acc
max_depth_grid = [3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

for max_depth in max_depth_grid:
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=max_depth).fit(x_train, y_train)
    temp_train_acc=cross_val_score(rf, x_test, y_test, cv=skf)
    train_acc.append(temp_train_acc.mean())
    test_acc.append(accuracy_score(rf.predict(x_test), y_test))

print("Best CV accuracy is {:.2f}% with {} max_depth".format(max(test_acc)*100,
                                                        max_depth_grid[np.argmax(test_acc)]))


plt.plot(max_depth_grid, train_acc, label='train')
plt.plot(max_depth_grid, test_acc, label='test')
plt.legend()
plt.xlabel('max_depth')
plt.ylabel('Accuracies')
plt.title('Random-Forest: accuracy vs max_depth');

# create model with default parameters- baseline
rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)

# Train it on the training set
cv_result_baseline= cross_val_score(rf_baseline, x_test, y_test, cv=skf)

# Evalute the results (cross-val)
print("CV accuracy score: {:.2f}%".format(cv_result_baseline.mean() * 100))


# train model
rf_baseline.fit(x_train, y_train)

pred_test_rf= rf_baseline.predict(x_test)
print("Test accuracy score: {:.2f}%".format((accuracy_score(pred_test_rf, y_test) * 100)))

# Create lists to save the values of accuracy on training and test sets
train_acc = []
test_acc = []
temp_train_acc = []
min_samples_leaf_grid = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

for min_sample in min_samples_leaf_grid:
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, min_samples_leaf=min_sample).fit(x_train, y_train)
    temp_train_acc=cross_val_score(rf, x_test, y_test, cv=skf)
    train_acc.append(temp_train_acc.mean())
    test_acc.append(accuracy_score(rf.predict(x_test), y_test))

print("Best CV accuracy is {:.2f}% with {} min_sample_leaf".format(max(test_acc)*100,
                                                        min_samples_leaf_grid[np.argmax(test_acc)]))

plt.plot(min_samples_leaf_grid, train_acc, label='train')
plt.plot(min_samples_leaf_grid, test_acc, label='test')
plt.legend()
plt.xlabel('min_sample_leaf')
plt.ylabel('Accuracies')
plt.title('Random-Forest: accuracy vs min_sample_leaf');

# Initialize the set of parameters for exhaustive search and fit
parameters = {'max_features': [7, 10, 16, 18],
              'min_samples_leaf': [1, 3, 5, 7],
              'max_depth': [15, 20, 24, 27]}
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
gcv = GridSearchCV(rf, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(x_train, y_train)

gcv.best_params_, gcv.best_score_