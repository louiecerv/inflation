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

    st.subheader('Statistical Analysis on the Factors that Could Possible Affect the Spending Habits')
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

    df = pd.read_csv('influence.csv', header=0)

    # Shuffle the DataFrame
    df = df.sample(frac=1)
    st.write('Browse the dataset')
    st.write(df)

    df1 = df.copy()

    st.subheader('Frequency counts')

    display_freqs(df, "Sex")
    display_freqs(df, "Year Level")
    display_freqs(df, "Course")
    display_freqs(df, "Income")

    # Create a figure and axes object
    fig, ax = plt.subplots()

    # Create the countplot using the axes object
    p = sns.countplot(x="Influence", hue = "Influence", data=df, palette="Set1", ax=ax, legend=False)

    # Rotate x-axis labels for better readability
    plt.setp(p.get_xticklabels(), rotation=90)
    plt.title('Frequency distribution of Influences on Spending Habits')
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
        if cn != "Influence":
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
    st.write('The spending habit influence means and std when grouped according to Influence Level:')
    mean_std(df, "influencelevel")
    st.write('The spending influence means and std when grouped according to Sex:')
    plot_usage_by(df, "Sex")
    mean_std(df, "Sex")
    st.write('The spending influence means and std when grouped according to Year Level:')
    plot_usage_by(df, "Year Level")
    mean_std(df, "Year Level")
    st.write('The spending influence means and std when grouped according to Course:')
    plot_usage_by(df, "Course")
    mean_std(df, "Course")
    st.write('The spending influence means and std when grouped according to Income:')
    plot_usage_by(df, "Income")
    mean_std(df, "Income")

    st.subheader('Inferential Statistics')
    text = """insert interpretation """
    st.write(text)
    st.subheader('Chi-square Test of Sex and Spending Influence Level')
    chi_square(df, "Sex")
    text = """Based on the chi-square test results, there is no statistically significant 
    association between sex (male/female) and spending habits influence level.
    The chi-square statistic (2.41) is relatively low, indicating weak evidence for a relationship.
    The p-value (0.66) is much greater than the typical significance level (0.05). 
    A p-value this high suggests we cannot reject the null hypothesis, which states that
    there's no association between sex and spending habits influence. Degrees of freedom (4) refer to 
    the number of comparisons made after considering sample size. In simpler terms, the data 
    doesn't provide enough evidence to conclude that men and women are influenced differently 
    when it comes to spending habits."""
    st.write(text)

    st.subheader('Chi-square Test of Course and Spending Influence Level')
    chi_square(df, "Course")
    text = """Based on the chi-square test statistic of 23.79 and a p-value of 0.02, we can reject 
    the null hypothesis. This means that there is a statistically significant difference in 
    spending habits level grouped by course. In other words, the spending habits level is not 
    independent of the course taken. There's a connection between the two. For instance, 
    students enrolled in a particular course might tend to spend more on certain things 
    compared to students in other courses"""
    st.write(text)

    st.subheader('Chi-square Test of Year Level and Spending Influence Level')
    chi_square(df, "Year Level")
    text = """insert interpretation """
    st.write(text)

    st.subheader('Chi-square Test of Income and Spending Influence Level')
    chi_square(df, "Income")
    text = """insert interpretation """
    st.write(text)

    st.subheader('ANOVA Test')

    text = """insert interpretation """
    st.write(text)

    g1 = df1.loc[(df1['Sex'] =='Male'), 'Influence']
    g2 = df1.loc[(df1['Sex'] =='Female'), 'Influence']
    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2)
    # Print the results
    st.subheader('ANOVA Test of Sex and Spending Habit Influence')
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))
    text = """Insert Interpretation"""
    st.write(text)

    g1 = df1.loc[(df1['Year Level'] =='First Year'), 'Influence']
    g2 = df1.loc[(df1['Year Level'] =='Second Year'), 'Influence']
    g3 = df1.loc[(df1['Year Level'] =='Third Year'), 'Influence']
    g4 = df1.loc[(df1['Year Level'] =='Fourth Year'), 'Influence']
    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2, g3, g4)
    # Print the results
    st.subheader('ANOVA Test of Year Level and Spending Habit Influence')
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))
    text = """Insert Interpretation"""
    st.write(text)

    g1 = df1.loc[(df1['Course'] =='BSTM'), 'Influence']
    g2 = df1.loc[(df1['Course'] =='BSCM'), 'Influence']
    g3 = df1.loc[(df1['Course'] =='BSBA'), 'Influence']
    g4 = df1.loc[(df1['Course'] =='BSHM'), 'Influence']

    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2, g3, g4)
    # Print the results
    st.subheader('ANOVA Test of Course and Spending Habit Influence')
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))
    text = """Insert Interpretation"""
    st.write(text)

    g1 = df1.loc[(df1['Income'] =='Php 20 000 and Below'), 'Influence']
    g2 = df1.loc[(df1['Income'] =='Php 20 001 to Php 60 000'), 'Influence']
    g3 = df1.loc[(df1['Income'] =='Above Php 60 000'), 'Influence']

    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2, g3)
    st.subheader('ANOVA Test of Income and Spending Habit Influence')
    # Print the results
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))
    text = """Insert Interpretation"""
    st.write(text)

def mean_std(df, column_name):
    grouped_data = df.groupby(column_name)

    # Calculate mean and standard deviation of usage for each group
    results = grouped_data['Influence'].agg(['mean', 'std'])
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
    sns.countplot(x=column, data=df, hue='influencelevel', palette='bright', ax=ax)

    # Set the title and adjust layout
    ax.set_title("Spending Influence Levels Grouped by " + column, fontsize=14)
    plt.tight_layout()  # Prevent overlapping elements

    # Display the plot
    st.pyplot(fig)

def chi_square(df, column):
# Generate a contingency table
    cont_table = pd.crosstab(df[column], df['influencelevel'])
    # Display the contingency table
    st.write(cont_table)    
    # perform a chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(cont_table)
    # print the results
    st.write("Chi-square statistic: ", f"{chi2_stat:.2f}")
    st.write("p-value: ", f"{p_value:.2f}")
    st.write("Degrees of freedom: ", dof)
    st.write("Expected frequencies: \n", expected)


#run the app
if __name__ == "__main__":
    app()
