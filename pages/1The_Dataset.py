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

    df1 = df.copy()

    st.subheader('Frequency counts')

    display_freqs(df, "Sex")
    display_freqs(df, "Year Level")
    display_freqs(df, "Course")
    display_freqs(df, "Income")

    # Create a figure and axes object
    fig, ax = plt.subplots()

    # Create the countplot using the axes object
    p = sns.countplot(x="Usage", hue = "Usage", data=df, palette="Set1", ax=ax, legend=False)

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
    st.subheader('Chi-square Test of Sex and Usage Level')
    chi_square(df, "Sex")
    st.subheader('Chi-square Test of Course and Usage Level')
    chi_square(df, "Course")
    st.subheader('Chi-square Test of Year Level and Usage Level')
    chi_square(df, "Year Level")
    st.subheader('Chi-square Test of Income and Usage Level')
    chi_square(df, "Income")

    st.subheader('ANOVA Test')

    text = """ANOVA, or Analysis of Variance, is a statistical technique used to compare the 
    means of two or more groups. It does this by breaking down the total variance in the 
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

    g1 = df1.loc[(df1['Sex'] =='Male'), 'Usage']
    g2 = df1.loc[(df1['Sex'] =='Female'), 'Usage']
    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2)
    # Print the results
    st.subheader('ANOVA Test of Sex and E-Banking Usage')
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))
    text = """The F-statistic of 1.72 and p-value of 0.1912 in
    the ANOVA test suggests there is not statistically significant evidence to reject 
    the null hypothesis. In other words, at a significance level of 0.05, we 
    cannot conclude that the average usage level differs between the two sexes 
    based on this data."""
    st.write(text)
    
    g1 = df1.loc[(df1['Year Level'] =='First Year'), 'Usage']
    g2 = df1.loc[(df1['Year Level'] =='Second Year'), 'Usage']
    g3 = df1.loc[(df1['Year Level'] =='Third Year'), 'Usage']
    g4 = df1.loc[(df1['Year Level'] =='Fourth Year'), 'Usage']
    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2, g3, g4)
    # Print the results
    st.subheader('ANOVA Test of Year Level and E-Banking Usage')
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))
    text = """An F-statistic of 0.60 and a p-value of 0.6132 suggests that there is no statistically 
    significant difference in usage level between year levels.  Since both the F-statistic and p-value 
    point in the same direction, we can confidently conclude that there's not enough evidence 
    to say that usage level differs statistically between year levels in this study."""
    st.write(text)

    g1 = df1.loc[(df1['Course'] =='BSTM'), 'Usage']
    g2 = df1.loc[(df1['Course'] =='BSCM'), 'Usage']
    g3 = df1.loc[(df1['Course'] =='BSBA'), 'Usage']
    g4 = df1.loc[(df1['Course'] =='BSHM'), 'Usage']

    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2, g3, g4)
    # Print the results
    st.subheader('ANOVA Test of Course and E-Banking Usage')
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))
    text = """Based on the F-statistic of 3.43 and a p-value of 0.0176, we can reject the null 
    hypothesis. This means there is a statistically significant difference in usage level 
    between courses.  In simpler terms, the results suggest that the average usage level 
    is not the same across different courses. There's evidence to conclude that some 
    courses have higher or lower usage levels compared to others."""
    st.write(text)

    g1 = df1.loc[(df1['Income'] =='Php 20 000 and Below'), 'Usage']
    g2 = df1.loc[(df1['Income'] =='Php 20 001 to Php 60 000'), 'Usage']
    g3 = df1.loc[(df1['Income'] =='Above Php 60 000'), 'Usage']

    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2, g3)
    st.subheader('ANOVA Test of Income and E-Banking Usage')
    # Print the results
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))
    text = """F-statistic: 5.26 - This value indicates the ratio of the variance between 
    income brackets to the variance within income brackets. A higher F-statistic suggests 
    a greater difference in usage levels between the income brackets.
    \np-value: 0.0057 - This value represents the probability of observing an F-statistic
     this extreme, assuming there's no actual difference between the income brackets 
     (null hypothesis). A small p-value (typically less than 0.05) rejects the null hypothesis.
     In this case, with an F-statistic of 5.26 and a p-value of 0.0057, we can reject the null
     hypothesis. This means there's a statistically significant difference in usage level 
     between the income brackets. The F-statistic itself doesn't tell you the direction of 
     the difference (i.e., which income bracket uses more), but it suggests that income 
     brackets play a role in usage levels."""
    st.write(text)

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
    cont_table = pd.crosstab(df[column], df['usagelevel'])
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
