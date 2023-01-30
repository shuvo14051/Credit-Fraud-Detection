
# # Goal
# * Credit Fraud Detection
# * Imbalance classification problem
# * Oversampling with SMOTE
# * Machine Learning approach
# * Deep Learning approach

# ## Import necessary libraries

import time
import collections
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read the dataset

df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.head()

# ## Exploratory Data Aanalysis (EDA)

print(df.shape)
"""
80/20 can be a good train test ratio.
Total number of samples = 284807. It's a lot of data.
So 20% will be 56k which is a good amount of data for the test set.
"""
print("10% of total data:", int(df.shape[0]*.2)) 

"""
Check null values.
There is no null values in this dataset.
We are good to go for further steps.
"""
print(sum(df.isnull().sum()))

df.info()

"""
See the number of each class.
99.83% data poitns are in one class
00.17% data points are in another class.
We have a clear class imbalance here.
"""
print(df['Class'].value_counts(normalize=True))
print(df.columns)

## Count plot of the target coulumn.
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution\n(0>>Not Fraud & 1>>Fraud)")
plt.show()

fig, ax = plt.subplots(1, 2,figsize=(16,4))
sns.distplot(df['Amount'], ax=ax[0])
sns.distplot(df['Time'], ax=ax[1])
ax[0].set_title("Distribution of Transaction Amount")
ax[1].set_title("Distribution of Transaction Time")
plt.show()


# ## Train test split

X = df.drop('Class', axis=1)
y = df['Class']

sns.countplot(y=y, data=df, palette="mako_r")
plt.title("Result before oversampling")
plt.ylabel('Class')
plt.xlabel('Total')
plt.show()

from imblearn.over_sampling import SMOTE
X, y = SMOTE().fit_resample(X, y)

sns.countplot(y=y, data=df)
plt.title("Result after oversampling")
plt.ylabel('Drug Type')
plt.xlabel('Total')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify=y,random_state=42)


# ## Scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# # 1. Machine Learning Approach
# ## Modeling & performance evaluation

def model(clf):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def performance(y_true, y_pred):
    print("======================================================")
    print(classification_report(y_test,y_pred))
    print("======================================================")
    print("Overall accuracy = ",accuracy_score(y_test,y_pred))


# ## SVM model
# <figure>
# <img src = "https://raw.githubusercontent.com/shuvo14051/Share-images/master/SVM_imbalance.jpg" height=400 width=400>
#   <figcaption>Result of SVM without balancing the dataset</figcaption>
# </figure>

y_pred = model(SVC())
performance(y_test,y_pred)


# ## Logistic Regression
# <figure>
# <img src = "https://raw.githubusercontent.com/shuvo14051/Share-images/master/LogiticRegression_imbalance.jpg" height=400 width=400>
#   <figcaption>Result of LogisticRegression without balancing the dataset</figcaption>
# </figure>

y_pred = model(LogisticRegression())
performance(y_test,y_pred)


# ## RandomForest classifier
# <figure>
# <img src = "https://raw.githubusercontent.com/shuvo14051/Share-images/master/RandomForest_imbalance.jpg" height=400 width=400>
#   <figcaption>Result of RandomForest without balancing the dataset</figcaption>
# </figure>
# 

y_pred = model(RandomForestClassifier())
performance(y_test,y_pred)


# ## KNN
# <figure>
# <img src = "https://raw.githubusercontent.com/shuvo14051/Share-images/master/KNN_imbalance.jpg" height=400 width=400>
#   <figcaption>Result of KNN without balancing the dataset</figcaption>
# </figure>

y_pred = model(KNeighborsClassifier())
performance(y_test,y_pred)


# # 2. Deep Learning Approach

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

n_cols = X_train.shape[1]
model = Sequential()
model.add(Dense(50, activation = 'relu', input_shape=(n_cols,)))
model.add(BatchNormalization())
model.add(Dropout(.2))

model.add(Dense(25, activation = 'relu',))
model.add(BatchNormalization())
model.add(Dropout(.2))

model.add(Dense(11, activation = 'relu',))
model.add(BatchNormalization())
model.add(Dropout(.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2, mode='min')

result = model.fit(X_train, y_train, 
          epochs=50, 
          batch_size=128, 
          validation_data=(X_test,y_test),
          )

loss_df = pd.DataFrame(model.history.history)
plt.plot(loss_df['loss'], label='Training loss')
plt.plot(loss_df['val_loss'], label='Validation loss')
plt.legend()

plt.plot(loss_df['accuracy'], label='Training accuracy')
plt.plot(loss_df['val_accuracy'], label='Validation accuracy')
plt.legend()

y_pred = model.predict(X_test)
y_pred = y_pred.round()
performance(y_test,y_pred)
