import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex
from collections import Counter

sns.set_style('whitegrid')

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score

df = pd.read_csv("german.data", header= None)
print(df.shape)
target = df.values[:,-1]
counter = Counter(target)
for k,v in counter.items():
    per = v / len(target)*100
    print('Class %d, Count %d, Percentage=% .3f%%' % (k, v, per))


















