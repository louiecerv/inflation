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
from scipy.stats import chi2_contingency
import scipy.stats as stats
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

    st.subheader('Frequency counts')

    display_freqs(df, "Sex")
    display_freqs(df, "Year Level")
    display_freqs(df, "Course")
    display_freqs(df, "Income")

    # Create a figure and axes object
    fig, ax = plt.subplots()

    # Create the countplot using the axes object
    p = sns.countplot(x="Usage", data=df, palette="Set1", ax=ax)

    # Rotate x-axis labels for better readability
    plt.setp(p.get_xticklabels(), rotation=90)
    plt.title('Frequency distribution of E-banking Usage')
    st.pyplot(fig)
   
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
    plot_usage_by(df, "Sex")
    mean_std(df, "Sex")
    st.write('The e-banking usage means and std when grouped according to Year Level:')
    plot_usage_by(df, "Year Level")
    mean_std(df, "Year Level")
    st.write('The e-banking usage means and std when grouped according to Course:')
    plot_usage_by(df, "Course")
    mean_std(df, "Course")
    st.write('The e-banking usage means and std when grouped according to Income:')
    plot_usage_by(df, "Income")
    mean_std(df, "Income")

    st.subheader('Inferential Statistics')
    text = """Chi-square test: This is a statistical method used to determine if there is 
    a significant association between two categorical variables. For example, if we want 
    to know if there is a significant association between sex and levels of usage of 
    online payment, you can use a chi-square test to determine if there is a significant
    difference in the distribution of responses between males and females."""
    st.write(text)

    chi_square("Sex")

    
def mean_std(df, column_name):
    grouped_data = df.groupby(column_name)

    # Calculate mean and standard deviation of usage for each group
    results = grouped_data['Usage'].agg(['mean', 'std'])
    # Print the results
    st.write(results)

def display_freqs(df, column):
    # Get the frequency count of each class in the column
    col_counts = df[column].value_counts()

    # Print the frequency table
    st.write(col_counts)
    
    # Create the figure and axes objects    
    fig, ax = plt.subplots()  # Create a figure and a single axes
    # Create a bar chart of the frequency using seaborn
    sns.barplot(x=col_counts.index, y=col_counts.values)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title('Frequency of ' + column)
    st.pyplot(fig)

def plot_usage_by(df, column):
    # Create the figure and axes object
    fig, ax = plt.subplots(figsize=(6, 3))

    # Create the countplot directly on the provided axes
    sns.countplot(x=column, data=df, hue='usagelevel', palette='bright', ax=ax)

    # Set the title and adjust layout
    ax.set_title("Usage Levels Grouped by " + column, fontsize=14)
    plt.tight_layout()  # Prevent overlapping elements

    # Display the plot
    st.pyplot(fig)

def chi_square(df, column):
# Generate a contingency table
    cont_table = pd.crosstab(df['Sex'], df['usagelevel'])
    # Display the contingency table
    st.write(cont_table)    
    # perform a chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(cont_table)
    # print the results
    st.write("Chi-square statistic: ", chi2_stat)
    st.write("p-value: ", p_value)
    st.write("Degrees of freedom: ", dof)
    st.write("Expected frequencies: \n", expected)

#run the app
if __name__ == "__main__":
    app()
