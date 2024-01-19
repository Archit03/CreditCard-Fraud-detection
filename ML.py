import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import warnings

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

# Create a RandomForestClassifier
rfc = RandomForestClassifier(random_state=1, n_estimators=65)

# Train the model
rfc.fit(X_train_resampled, Y_train_resampled)

# Evaluate the model
Y_pred = rfc.predict(X_test_resampled)
conf_matrix = confusion_matrix(Y_test_resampled, Y_pred)
classification_rep = classification_report(Y_test_resampled, Y_pred)

# Feature Importance
feature_importance = rfc.feature_importances_
features = X.columns

# Streamlit app
st.title("Credit Card Fraud Detection")

# Display confusion matrix
st.subheader("Confusion Matrix:")
st.write(conf_matrix)

# Display classification report
st.subheader("Classification Report:")
st.write(classification_rep)

# Plot confusion matrix heatmap
st.subheader("Confusion Matrix Heatmap:")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
st.pyplot()

# Plot feature importance
st.subheader("Feature Importance:")
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
st.pyplot()
