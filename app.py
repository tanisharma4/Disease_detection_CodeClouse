import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("C:\\Users\\KIIT\\Desktop\\Data Science\\udemy\\Project 1\\heart-disease (1).csv")

df = load_data()

# Main title
st.title("Heart Disease Prediction App")

# Display the first few rows of the DataFrame
if st.checkbox("Show DataFrame"):
    st.write(df.head())

# Display basic statistics
st.subheader("Data Statistics")
st.write(df.describe())

# Sidebar
st.sidebar.title("Model Settings")

# Model selection
model_name = st.sidebar.selectbox("Select Model", ["Logistic Regression", "KNN", "Random Forest"])

# Data splitting
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Model fitting and evaluation
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=2000)
elif model_name == "KNN":
    model = KNeighborsClassifier()
elif model_name == "Random Forest":
    model = RandomForestClassifier()

model.fit(X_train_scaled, y_train)
accuracy = model.score(X_test_scaled, y_test)

# Make predictions
y_preds = model.predict(X_test_scaled)

st.write(f"Model Accuracy: {accuracy}")

# Visualizations
st.subheader("Data Visualizations")

# Displaying the target distribution
st.write("Target Distribution")
target_counts = df['target'].value_counts()
st.bar_chart(target_counts)

# Age vs. Max Heart Rate
st.write("Age vs. Max Heart Rate")
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df.age[df.target==1], df.thalach[df.target==1], color='red')
ax.scatter(df.age[df.target==0], df.thalach[df.target==0], color='green')
plt.title("Age vs heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", 'No Disease'])
st.pyplot(fig)

# Heart Disease Frequency Per Chest Pain Type
st.write("Heart Disease Frequency Per Chest Pain Type")
cp_target_cross = pd.crosstab(df.cp, df.target)
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='cp', y='target')
plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No Disease","Disease"])
fig_cp = plt.gcf()  # Get the current figure
st.pyplot(fig_cp)

# Correlation Matrix
st.write("Correlation Matrix")
co_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(co_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="magma", cbar_kws={"shrink": 0.75})
plt.title("Correlation Matrix")
fig_corr = plt.gcf()  # Get the current figure
st.pyplot(fig_corr)

# ROC Curve and AUC
st.write("Receiver Operating Characteristic (ROC) Curve")
y_score = model.predict_proba(X_test_scaled)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
st.pyplot(fig)

# Confusion Matrix
st.write("Confusion Matrix")
y_preds = model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_preds)
sns.set(font_scale=1.5)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, cbar=False)
plt.xlabel("True label")
plt.ylabel("Predicted label")
st.pyplot(fig)

# Classification Report
st.write("Classification Report")
cr = classification_report(y_test, y_preds)
st.write(cr)

# Cross-validated Metrics
st.subheader("Cross-validated Metrics")
cv_acc = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
cv_precision = cross_val_score(model, X, y, cv=5, scoring="precision").mean()
cv_recall = cross_val_score(model, X, y, cv=5, scoring="recall").mean()
cv_f1 = cross_val_score(model, X, y, cv=5, scoring="f1").mean()

# Cross-validated Metrics
cv_metrics = pd.DataFrame({
    "Accuracy": cv_acc,
    "Precision": cv_precision,
    "Recall": cv_recall,
    "F1": cv_f1
}, index=[0])

st.write(cv_metrics)

# Feature Importance (for Logistic Regression only)
if model_name == "Logistic Regression":
    st.subheader("Feature Importance")
    feat_dict = dict(zip(df.columns, list(model.coef_[0])))
    feature_df = pd.DataFrame(feat_dict, index=[0])
    st.bar_chart(feature_df.T)

# Additional features
# Add any additional features you want to include in your app
