import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("creditcard.csv")

# Select features and target variable
X = df[['Time', 'Amount']]  # Select features
Y = df['Class']  # Select target variables

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Apply standardScaler method on X_train and X_test
ss = StandardScaler()  # Create an object of StandardScaler
X_train = ss.fit_transform(X_train)  # It sets the mean to 0 and standard deviation to 1
X_test = ss.transform(X_test)  # It transforms the data by scaling it to unit variance.

# Oversample the minority class
ros = RandomOverSampler(random_state=1)  # Create an object of RandomOverSampler
X_train_resampled, Y_train_resampled = ros.fit_resample(X_train,
                                                        Y_train)  # It resamples the data by oversampling the minority class.
X_test_resampled, Y_test_resampled = ros.fit_resample(X_test,
                                                      Y_test)  # It resamples the data by oversampling the minority class.


@st.cache_data
def evaluate_model(model, X_train, Y_train, X_test, Y_test, model_name):
    model.fit(X_train, Y_train)  # fit the model
    Y_pred = model.predict(X_test)  # predict the target variable
    cm = confusion_matrix(Y_test, Y_pred)  # confusion matrix
    cr = classification_report(Y_test, Y_pred)  # classification report
    return cm, cr


# Model Evaluation
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=1),
    "Random Forest": RandomForestClassifier(random_state=1, n_estimators=65),
    "AdaBoost": AdaBoostClassifier(random_state=1, n_estimators=3),
    "Gradient Boosting": GradientBoostingClassifier(random_state=1, n_estimators=3),
    "Linear Support Vector Machine": LinearSVC(random_state=1)
}

# Streamlit UI
st.title("Model Evaluation")
selected_model = st.selectbox("Select Model", list(models.keys()))

if st.button("Evaluate"):
    cm, cr = evaluate_model(models[selected_model], X_train_resampled, Y_train_resampled, X_test_resampled,
                            Y_test_resampled, selected_model)
    st.write(f"Confusion Matrix for {selected_model}:")
    st.write(cm)
    st.write(f"Classification Report for {selected_model}:")
    st.write(cr)

