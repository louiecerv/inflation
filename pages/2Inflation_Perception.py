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

    st.subheader('Statistical Analysis on the Factors that Could Affect the Inflation Perception')
    text = """Describe the inflation perception in the paper..."""
    with st.expander("About the Dataset. CLick to expand."):
        st.write(text)

    df = pd.read_csv('inflation-final.csv', header=0)
    df = df.drop('Awareness', axis = 1)
    df = df.drop('awarenesslevel', axis = 1)

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
    p = sns.countplot(x="Perception", hue = "Perception", data=df, palette="Set1", ax=ax, legend=False)

    # Rotate x-axis labels for better readability
    plt.setp(p.get_xticklabels(), rotation=90)
    plt.title('Frequency distribution of Inflation Perception')
    st.pyplot(fig)
   
    with st.expander("CLick to view unique values"):
        # Get column names and unique values
        columns = df.columns
        unique_values = {col: df[col].unique() for col in columns}    
        
        # Display unique values for each column
        st.write("\n**Unique Values:**")
        for col, values in unique_values.items():
            st.write(f"- {col}: {', '.join(map(str, values))}")

    st.write('Descriptive Statistics')
    st.write(df.describe().T)
    st.write('The inflation perception means and std of all respondents as an entire group')
    plot_usage_by(df, "perceptionlevel")
    st.write('The inflation perception means and std when grouped according to Sex:')
    plot_usage_by(df, "Sex")
    mean_std(df, "Sex")
    st.write('The inflation perception means and std when grouped according to Age:')
    plot_usage_by(df, "Age")
    mean_std(df, "Age")
    st.write('The inflation perception means and std when grouped according to Civil Status:')
    plot_usage_by(df, "Civil Status")
    mean_std(df, "Civil Status")
    st.write('The inflation perception means and std when grouped according to Educational Attainment:')
    plot_usage_by(df, "Educ")
    mean_std(df, "Educ")
    st.write('The inflation perception means and std when grouped according to Socio-Economic Status:')
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

    st.subheader('T-test on the Level of Awareness Grouped by Sex')
    male_awareness = df1[df1['Sex'] == 'Male']['Perception']
    female_awareness = df1[df1['Sex'] == 'Female']['Perception']
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(male_awareness, female_awareness)
    # Print the results
    st.write("t-statistic:", t_statistic)
    st.write("p-value:", p_value)

    st.subheader('T-test on the Level of Perception Grouped by Civil Status')
    single_awareness = df1[df1['Civil Status'] == 'Single']['Perception']
    married_awareness = df1[df1['Civil Status'] == 'Married']['Perception']
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(single_awareness, married_awareness)
    # Print the results
    st.write("t-statistic:", t_statistic)
    st.write("p-value:", p_value)

    st.subheader('T-test on the Level of Perception Grouped by Socio-economic Status')
    lower_awareness = df1[df1['SEStatus'] == 'Low Income Class (Between ?9,100 to ?18,200)']['Perception']
    mid_awareness = df1[df1['SEStatus'] == 'Lower Middle Income Class (Between ?18,201 to ?36,400)']['Perception']
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(lower_awareness, mid_awareness)
    # Print the results
    st.write("t-statistic:", t_statistic)
    st.write("p-value:", p_value)    

    st.subheader('ANOVA test on the Level of Perception Grouped by Age')
    g1 = df1.loc[(df1['Age'] =='18-35 years old'), 'Perception']
    g2 = df1.loc[(df1['Age'] =='36-50 years old'), 'Perception']
    g3 = df1.loc[(df1['Age'] =='above 50 years old'), 'Perception']
    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2, g3)
    # Print the results
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))    

    st.subheader('ANOVA test on the Level of Perception Grouped by Educational Attainment')
    g1 = df1.loc[(df1['Educ'] =='Elementary Graduate'), 'Perception']
    g2 = df1.loc[(df1['Educ'] =='High School Graduate'), 'Perception']
    g3 = df1.loc[(df1['Educ'] =='College Graduate'), 'Perception']
    g4 = df1.loc[(df1['Educ'] =="Master's Graduate"), 'Perception']
    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2, g3, g4)
    # Print the results
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))    

def mean_std(df, column_name):
    grouped_data = df.groupby(column_name)

    # Calculate mean and standard deviation for each group
    results = grouped_data['Perception'].agg(['mean', 'std'])
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
    p = sns.barplot(x=col_counts.index, y=col_counts.values)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    # Rotate x-axis labels for better readability
    plt.setp(p.get_xticklabels(), rotation=90)    
    plt.title('Frequency of ' + column)
    plt.tight_layout()  # Prevent overlapping elements    
    st.pyplot(fig)

def plot_usage_by(df, column):
    # Create the figure and axes object
    fig, ax = plt.subplots(figsize=(6, 3))

    # Create the countplot directly on the provided axes
    p = sns.countplot(x=column, data=df, hue='perceptionlevel', palette='bright', ax=ax)

    # Set the title and adjust layout
    ax.set_title("Perception Level Grouped by " + column, fontsize=14)
    # Rotate x-axis labels for better readability
    plt.setp(p.get_xticklabels(), rotation=90)
    plt.tight_layout()  # Prevent overlapping elements

    # Display the plot
    st.pyplot(fig)    

#run the app
if __name__ == "__main__":
    app()
