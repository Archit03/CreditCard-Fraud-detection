import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm  # Added for progress bars
from sklearn.svm import LinearSVC, SVC

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("creditcard.csv")

# Select features and target variable
X = df[['Time', 'Amount']]
Y = df['Class']

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Apply standardScaler method on X_train and X_test
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Oversample the minority class
ros = RandomOverSampler(random_state=1)
X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)
X_test_resampled, Y_test_resampled = ros.fit_resample(X_test, Y_test)


def evaluate_model(model, X_train, Y_train, X_test, Y_test, model_name):
    tqdm.write(f"\n{model_name}:")  # Display model name using tqdm
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    tqdm.write("Confusion Matrix:")
    tqdm.write(str(confusion_matrix(Y_test, Y_pred)))
    tqdm.write("Classification Report:")
    tqdm.write(classification_report(Y_test, Y_pred))


# Logistic Regression
lr = LogisticRegression()
evaluate_model(lr, X_train_resampled, Y_train_resampled, X_test_resampled, Y_test_resampled, "Logistic Regression")

# Decision Tree
dt = DecisionTreeClassifier(random_state=1)
evaluate_model(dt, X_train_resampled, Y_train_resampled, X_test_resampled, Y_test_resampled, "Decision Tree")

# Random Forest
rfc = RandomForestClassifier(random_state=1, n_estimators=65)
evaluate_model(rfc, X_train_resampled, Y_train_resampled, X_test_resampled, Y_test_resampled, "Random Forest")

# AdaBoost
ada = AdaBoostClassifier(random_state=1, n_estimators=3)
evaluate_model(ada, X_train_resampled, Y_train_resampled, X_test_resampled, Y_test_resampled, "AdaBoost")

# Gradient Boosting
gbc = GradientBoostingClassifier(random_state=1, n_estimators=3)
evaluate_model(gbc, X_train_resampled, Y_train_resampled, X_test_resampled, Y_test_resampled, "Gradient Boosting")

# Linear Support Vector Machine
svc = LinearSVC(random_state=1)
evaluate_model(svc, X_train_resampled, Y_train_resampled, X_test_resampled, Y_test_resampled, "Linear Support Vector Machine")

"""
# Support Vector Machine with Polynomial kernel
p_svc = SVC(random_state=1, kernel='poly')
evaluate_model(p_svc, X_train_resampled, Y_train_resampled, X_test_resampled, Y_test_resampled, "Support Vector Machine with Polynomial kernel")
"""