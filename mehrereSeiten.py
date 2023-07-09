import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the data
df = pd.read_csv("C:/Users/ahmad/PycharmProjects/Bank_Marketing/Bank Marketing.csv")

# Preprocessing
df.drop(columns="id", axis=1, inplace=True)
le = LabelEncoder()
for column in df.select_dtypes(include='object'):
    df[column] = le.fit_transform(df[column])
x = df.drop(columns="Class", axis=1)
y = df[["Class"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=0)

# Model Training and Evaluation
def train_model(model, x_train, y_train, x_test, y_test):
    if isinstance(model, (GaussianNB, DecisionTreeClassifier, SVC, AdaBoostClassifier, RandomForestClassifier)):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    else:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=50, batch_size=20, validation_data=(x_test, y_test))
        y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

classifiers = {
    'Artificial Neural Network': Sequential([
        Dense(16, activation="relu"),
        Dense(36, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ]),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=1),
    'Random Forest': RandomForestClassifier()
}

# Page for Display DataFrame and Display DataFrame Information and Display Missing Values and Display Descriptive Statistics
def page_data():
    st.title("Data Exploration")
    st.header("Display DataFrame")
    st.dataframe(df.astype(str))

    st.header("Data Information")
    st.text(f"Shape: {df.shape}")
    st.text("Columns:")
    st.write(df.columns)

    st.header("Missing Values")
    st.text("Number of missing values per column:")
    st.write(df.isnull().sum())

    st.header("Descriptive Statistics")
    st.write(df.describe())

# Page for Data Distribution and Correlation Matrix Heatmap
def page_distribution():
    st.title("Data Distribution")
    for col in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        st.pyplot()

    st.title("Correlation Matrix")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True)
    st.pyplot()

# Page for Model Training and Evaluation
def page_model():
    st.title("Model Training and Evaluation")
    for name, classifier in classifiers.items():
        accuracy = train_model(classifier, x_train, y_train, x_test, y_test)
        st.subheader(name)
        st.text(f"Accuracy: {accuracy}")

# Streamlit App
def main():
    st.title("Bank Marketing Analysis")
    st.subheader("Code Demo with Streamlit")

    # Create a sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Data Exploration", "Data Distribution", "Model Training"])

    # Render the selected page
    if page == "Data Exploration":
        page_data()
    elif page == "Data Distribution":
        page_distribution()
    elif page == "Model Training":
        page_model()

if __name__ == '__main__':
    main()
