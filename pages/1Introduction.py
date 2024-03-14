#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# Define the Streamlit app
def app():

    if "le_list" not in st.session_state:
        st.session_state.le_list = []

    st.subheader('The task: Classify E-banking usage as very low, low, moderate, high or very high.')
    text = """CBM Student E-Banking Usage Dataset
    \nThis dataset investigates the factors that affect e-banking usage and spending habits 
    among students at CBM.
    Features:
    sex (categorical): Student's gender (e.g., Male, Female)
    year_level (categorical): Student's year in the program
    course (categorical): Student's course of study
    family_income (numerical): Student's reported family income level
    Target Variable:
    usagelevel (categorical): Level of e-banking usage by the student
    Sampling Method:
    Stratified random sampling: This ensures the sample population reflects the 
    proportions of students based on year level and/or course within CBM."""
    with st.expander("About the Dataset. CLick to expand."):
        st.write(text)

    #replace with your dataset
    df = pd.read_csv('e-banking3.csv', header=0)
    df = df.drop('Usage', axis=1)
    
    # Shuffle the DataFrame (returns a copy)
    df = df.sample(frac=1)
    st.write('Browse the dataset')
    st.write(df)

    # Create a progress bar object
    st.progress_bar = st.progress(0, text="Generating data graphs please wait...")

    with st.expander("CLick to view graphs"):
        plot_feature(df, "usagelevel", "Usage Level", "Distribution of E-banking Usage Level")
        plot_feature(df, "Sex", "Sex", "Distribution of Sex")
        plot_feature(df, "Year Level", "Year Level", "Distribution of Year Level")
        plot_feature(df, "Course", "Course", "Distribution of Course")
        plot_feature(df, "Income", "Income", "Distribution of Family Income")

        countplot(df, "Sex", "Differences in E-Benking Usage According to Sex")
        countplot(df, "Year Level", "Differences in E-Banking Usage According to Year Level")
        countplot(df, "Course", "Differences in E-Banking Usage According to Course")
        countplot(df, "Income", "Differences in E-Banking Usage According to Family Income")

    with st.expander("CLick to view unique values"):
        # Get column names and unique values
        columns = df.columns
        unique_values = {col: df[col].unique() for col in columns}    
        
        # Display unique values for each column
        st.write("\n**Unique Values:**")
        for col, values in unique_values.items():
            st.write(f"- {col}: {', '.join(map(str, values))}")


    # encode the data to numeric
    le = LabelEncoder()
    #Get the list of column names
    column_names = df.columns.tolist()

    le_list = []  # Create an empty array to store LabelEncoders
    # Loop through each column name
    for cn in column_names:
        le = LabelEncoder()  # Create a new LabelEncoder for each column
        le.fit(df[cn])  # Fit the encoder to the specific column
        le_list.append(le)  # Append the encoder to the list
        df[cn] = le.transform(df[cn])  # Transform the column using the fitted encoder

    # save the label encoder to the session state
    st.session_state["le_list"] = le_list
    st.session_state['df'] = df    

    st.write('The Dataset after encoding features to numbers')
    st.write(df)

    st.write('Descriptive Statistics')
    st.write(df.describe().T)

    X = df.drop('usagelevel', axis=1)
    y = df['usagelevel']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    #save the values to the session state    
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test    
    
    for i in range(100):
        # Update progress bar value
        st.progress_bar.progress(i + 1)
        # Simulate some time-consuming task (e.g., sleep)
        time.sleep(0.01)        
    st.success("Data visualization completed!")

def plot_feature(df, feature, feature_label, title):
    # Show the distribution of usagelevel
    #df_counts = df["usagelevel"].value_counts().reset_index(name="count")
    df_counts = df[feature].value_counts().reset_index(name="count")
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(8, 3))  # Adjust figure size as needed
    p = sns.barplot(
        y=feature,
        x="count",
        data=df_counts,
        palette="bright",  # Adjust palette as desired (see seaborn color palettes)
        hue = feature
    )
    # Customize plot elements
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel(feature_label, fontsize=12)
    ax.bar_label(ax.containers[0])  # Add frequency counts to the bars
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.tight_layout()
    st.pyplot(fig)

def countplot(df, feature, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    # Create the countplot with clear title and legend
    p = sns.countplot(x=feature, data = df, hue='usagelevel',  palette='bright')
    ax.set_title(title, fontsize=14)

    # Display the plot
    plt.tight_layout()  # Prevent overlapping elements
    st.pyplot(fig)

#run the app
if __name__ == "__main__":
    app()
