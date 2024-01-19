import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings('ignore')
# to load dataset
df = pd.read_csv("creditcard.csv")
df.head()

X = df[['Time', 'Amount']]  # input
Y = df['Class']  # output

Y.value_counts()
X.isnull().sum()

print(df.dtypes)
X.head()

Y.head()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Apply standardScaler method on X_train and X_test
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

Y_train.value_counts()

# create the object of RandomOverSampler class
ros = RandomOverSampler(random_state=1)

X_train1, Y_train1 = ros.fit_resample(X_train, Y_train)
# fit_resample() inbuilt method h RandomOverSampler class ka

Y_train1.value_counts()

print(X_train1.shape, Y_train1.shape)

Y_test.value_counts()

X_test1, Y_test1 = ros.fit_resample(X_test, Y_test)

Y_test1.value_counts()


def create_model(model):
    model.fit(X_train1, Y_train1)
    Y_pred = model.predict(X_test1)
    print(confusion_matrix(Y_test1, Y_pred))
    print(classification_report(Y_test1, Y_pred))
    return model


lr = LogisticRegression()

lr = create_model(lr)

dt = DecisionTreeClassifier(random_state=1)

# create object of DecisionTreeClassifier class
dt = DecisionTreeClassifier(random_state=1)  # bydefault use method gini index

# call function
dt = create_model(dt)

# In[31]:


dict = {'Input Columns': X.columns, 'IG': dt.feature_importances_}
# IG means information Gain
# convert dictionary into DataFrame
df1 = pd.DataFrame(dict)
# sorting dataframe df1 according to IG column in descending order
# use inbuilt method sort_values() of pandas library
df1.sort_values('IG', ascending=False)  # by default ascending=True
# means ascending order


dt1 = DecisionTreeClassifier(random_state=1, criterion='entropy')
dt1 = create_model(dt1)

for i in range(10, 101):  # start i=10  stop=101-1=100 and step=+1
    # train on minimum 10 decision tree and max 100
    # create object of RandomForestClassifier  class
    rfc = RandomForestClassifier(random_state=1, n_estimators=i)
    # n_estimators inbuilt parameter of RandomForestClassifier  class
    # how many no. of decisiontree for train the model
    print("No. of decision Tree : ", i)
    # call function
    rfc = create_model(rfc)

for i in range(45, 101):
    dt1 = DecisionTreeClassifier(random_state=1, min_samples_leaf=i)
    print("min samples leaf: ", i)
    print("Confusion_matrix and Classification report")
    dt1 = create_model(dt1)
    print(dt1)

rfc = RandomForestClassifier(random_state=1, n_estimators=65)
# n_estimators inbuilt parameter of RandomForestClassifier  class
# how many no. of decisiontree for train the model
# call function
rfc.fit(X_train1, Y_train1)
rfc = create_model(rfc)

print(rfc.feature_importances_)

ada = AdaBoostClassifier(random_state=1, n_estimators=3)

# call function
# ada = create_model(ada)
# print(ada.feature_importances_)  # to show information gain of each features

for i in range(1, 17):  # start i=1  stop=17-1
    # create object of AdaBoostClassifier class and passing the parameter
    # n_estimators (means how many decision stump)
    # decision stump depends on input(information gain)
    ada = AdaBoostClassifier(random_state=1, n_estimators=i)
    print("No. of decision stump means no. of inputs ", i)
    # call function
    ada = create_model(ada)
# create object of AdaBoostClassifier class and passing the parameter
# n_estimators (means how many decision stump)
# decision stump depends on input(information gain)
ada = AdaBoostClassifier(random_state=1, n_estimators=3)
ada.fit(X_train1, Y_train1)
# call function
ada = create_model(ada)
print(ada.feature_importances_)
