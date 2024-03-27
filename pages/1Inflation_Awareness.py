#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import scipy.stats as stats
import time

# Define the Streamlit app
def app():

    if "le_list" not in st.session_state:
        st.session_state.le_list = []

    st.subheader('Statistical Analysis on the Factors that Could Affect the Inflation Awareness')
    text = """CBM Student E-Banking Usage Dataset
    \nThis dataset investigates the factors that affect inflation awareness 
    among job order workers in a state university.
    \nFeatures:
    sex (categorical): (Male, Female)
    age (categorical): (18-35 years old, 36-50 years old, above 50 years old)
    Civil Status (categorical): (Single, Married)
    Educational Attainment(categorical): (Elementary Graduate, High School Graduate, College Graduate, Masteral Graduate)
    Socio-economic status (categorical): (low income, lower-middle income)
    Awareness (ordinal): Awareness as measured by a Likert scale
    \nSampling Method:
    Stratified random sampling"""
    with st.expander("About the Dataset. CLick to expand."):
        st.write(text)

    df = pd.read_csv('inflation-final.csv', header=0)
    df = df.drop('Perception', axis = 1)
    df = df.drop('perceptionlevel', axis = 1)

    st.write('Browse the dataset')
    st.write(df)

    df1 = df.copy()

    st.subheader('Frequency counts')

    display_freqs(df, "Sex")
    display_freqs(df, "Age")
    display_freqs(df, "Civil Status")
    display_freqs(df, "Educ")
    display_freqs(df, "SEStatus")

    # Create a figure and axes object
    fig, ax = plt.subplots()

    # Create the countplot using the axes object
    p = sns.countplot(x="Awareness", hue = "Awareness", data=df, palette="Set1", ax=ax, legend=False)

    # Rotate x-axis labels for better readability
    plt.setp(p.get_xticklabels(), rotation=90)
    plt.title('Frequency distribution of Inflation Awareness')
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
        if cn != "Awareness":
            le = LabelEncoder()  # Create a new LabelEncoder for each column
            le.fit(df[cn])  # Fit the encoder to the specific column
            le_list.append(le)  # Append the encoder to the list
            df[cn] = le.transform(df[cn])  # Transform the column using the fitted encoder

    # save the label encoder to the session state
    st.session_state["le_list"] = le_list
    st.session_state['df'] = df    

    st.write('The Dataset after encoding the features to numbers')
    st.write(df)

    st.write('Descriptive Statistics')
    st.write(df.describe().T)
    st.write('The inflation awareness means and std when grouped according to Sex:')
    plot_usage_by(df, "Sex")
    mean_std(df, "Sex")
    st.write('The inflation awareness means and std when grouped according to Age:')
    plot_usage_by(df, "Age")
    mean_std(df, "Age")
    st.write('The inflation awareness means and std when grouped according to Civil Status:')
    plot_usage_by(df, "Civil Status")
    mean_std(df, "Civil Status")
    st.write('The inflation awareness means and std when grouped according to Educational Attainment:')
    plot_usage_by(df, "Educ")
    mean_std(df, "Educ")
    st.write('The inflation awareness means and std when grouped according to Socio-Economic Status:')
    plot_usage_by(df, "SEStatus")
    mean_std(df, "SEStatus")

    st.subheader('Inferential Statistics')
    st.subheader('ANOVA Test')
    text = """ANOVA, or Analysis of Variance, is a statistical technique used to compare the 
    means of three or more groups. It does this by breaking down the total variance in the 
    data into two components:
    \nVariance between groups: This represents the differences between the average 
    values of the groups.
    \nVariance within groups: This represents the variation around the mean within each group.
    ANOVA then calculates a statistic called the F-statistic. This statistic compares the variance
    between groups to the variance within groups. If the F-statistic is large, it suggests that 
    the differences between the group means are statistically significant. In other words, 
    it's likely that there's a true difference between the groups, not just random chance.
    \nIn essence, ANOVA helps you assess whether observed differences in group means are 
    meaningful or not.
    \nThe F-statistic measures the difference between the means of the groups relative to the 
    variability within the groups.
    \nThe p-value indicates the probability of observing such a difference by chance, assuming no 
    real effect of sex on usage.
    \nA statistically significant result (p-value < 0.05) suggests that the variable has a significant 
    effect on e-banking usage."""
    st.write(text)

def mean_std(df, column_name):
    grouped_data = df.groupby(column_name)

    # Calculate mean and standard deviation of usage for each group
    results = grouped_data['Awareness'].agg(['mean', 'std'])
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
    sns.countplot(x=column, data=df, hue='awarenesslevel', palette='bright', ax=ax)

    # Set the title and adjust layout
    ax.set_title("Awareness Level Grouped by " + column, fontsize=14)
    plt.tight_layout()  # Prevent overlapping elements

    # Display the plot
    st.pyplot(fig)    

#run the app
if __name__ == "__main__":
    app()
