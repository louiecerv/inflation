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

    st.subheader('Introduction')
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

    df = pd.read_csv('e-banking3.csv', header=0)
    #df = df.drop('Usage', axis = 1)

    # Shuffle the DataFrame
    df = df.sample(frac=1)
    st.write('Browse the dataset')
    st.write(df)

    st.write('Frequency counts')
    #st.write(df.describe(include='all'))

    # Get the group frequency count of each column
    group_freq_count = df.groupby(['usagelevel', 'Sex', 'Year Level', 'Course', 'Income']).size().unstack().fillna(0)


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
        if cn != "Usage":
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
    st.write('The e-banking usage means and std when grouped according to Usage Level:')
    mean_std(df, "usagelevel")
    st.write('The e-banking usage means and std when grouped according to Sex:')
    mean_std(df, "Sex")
    st.write('The e-banking usage means and std when grouped according to Year Level:')
    mean_std(df, "Year Level")
    st.write('The e-banking usage means and std when grouped according to Course:')
    mean_std(df, "Course")
    st.write('The e-banking usage means and std when grouped according to Income:')
    mean_std(df, "Income")
   
def mean_std(df, column_name):
    grouped_data = df.groupby(column_name)

    # Calculate mean and standard deviation of usagelevels for each gender group
    results = grouped_data['Usage'].agg(['mean', 'std'])
    # Print the results
    st.write(results)

#run the app
if __name__ == "__main__":
    app()
