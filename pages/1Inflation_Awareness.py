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
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Define the Streamlit app
def app():

    st.subheader('Statistical Analysis on the Factors that Could Affect the Inflation Awareness')
    text = """Inflation Awareness Among Job Order Workers Dataset
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
    with st.expander("About the Dataset. Click here to expand."):
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
    plt.title('Frequency distribution of Inflation Awareness of the Respondents taken as one Group')
    st.pyplot(fig)
   
    with st.expander("Click to view unique values"):
        # Get column names and unique values
        columns = df.columns
        unique_values = {col: df[col].unique() for col in columns}    
        
        # Display unique values for each column
        st.write("\n**Unique Values:**")
        for col, values in unique_values.items():
            st.write(f"- {col}: {', '.join(map(str, values))}")

    st.write('Descriptive Statistics')
    st.write(df.describe().T)
    st.write('The inflation awareness means and std of all respondents as an entire group')
    plot_usage_by(df, "awarenesslevel")
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


    st.subheader('Independent Sample T-Test')
    text = """The independent samples t-test, also called the two-sample t-test, 
    is a statistical tool used to compare the means of two independent groups. 
    It helps you assess if there's a significant difference in the average values 
    between two unrelated groups.
    \nPurpose: It determines if there's a statistically significant difference between
    the means of two independent groups on a continuous variable.
    \nOutputs: The test provides a t-statistic and a p-value. The p-value tells you 
    how likely it is to observe such a difference by chance, assuming the null hypothesis (no difference between means) is true. A low p-value (typically below 0.05) indicates a statistically significant difference.
    By interpreting the p-value, you can conclude whether the observed difference in 
    means between the two groups is likely due to random chance or reflects a true 
    difference in the populations they represent."""
    st.write(text)

    st.subheader('T-test on the Level of Awareness Grouped by Sex')
    male_awareness = df1[df1['Sex'] == 'Male']['Awareness']
    female_awareness = df1[df1['Sex'] == 'Female']['Awareness']
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(male_awareness, female_awareness)
    # Print the results
    st.write("t-statistic: {:.2f}".format(t_statistic))
    st.write("p-value: {:.4f}".format(p_value))  

    text = """The results of the independent samples t-test do not show statistically 
    significant evidence of a difference in awareness levels between males and females. 
    \nT-statistic: -0.5621. In t-tests, this statistic doesn't directly tell about 
    significance but indicates the direction and strength of the effect. 
    A negative value means the average awareness level for females was higher than 
    for males in this case. However, the magnitude is very small, close to zero.
    \nP-value: 0.5751. This is the key value for statistical significance. A common 
    threshold for significance is 0.05. Since the p-value here (0.5751) is much 
    greater than 0.05, we fail to reject the null hypothesis. The null hypothesis 
    states that there is no difference between the groups (in this case, males and females
    regarding awareness level).Based on this t-test, there's not enough evidence to 
    say that there's a significant difference in awareness levels between males and females. 
    It's possible that a larger sample size or a different study design might reveal
    a difference, but this particular test doesn't provide conclusive evidence."""
    st.write(text)

    st.subheader('T-test on the Level of Awareness Grouped by Civil Status')
    single_awareness = df1[df1['Civil Status'] == 'Single']['Awareness']
    married_awareness = df1[df1['Civil Status'] == 'Married']['Awareness']
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(single_awareness, married_awareness)
    # Print the results
    st.write("t-statistic: {:.2f}".format(t_statistic))
    st.write("p-value: {:.4f}".format(p_value))  

    text = """The results of the independent samples t-test do not show a 
    statistically significant difference in the level of awareness between 
    single and married workers. 
    \nt-statistic: -0.5622. This statistic doesn't tell us about the direction
    of the difference (i.e., whether single or married people have higher 
    awareness) but rather the strength of the evidence against the null 
    hypothesis (which is that there is no difference). 
    In this case, the absolute value is very small, indicating weak evidence 
    against the null hypothesis.
    \np-value: 0.5751. This is the probability of observing a t-statistic as 
    extreme or more extreme than the one calculated, assuming the null hypothesis 
    is true. A common significance level used in hypothesis testing is 0.05. 
    Since the p-value (0.5751) is greater than 0.05, we fail to reject the 
    null hypothesis. The data doesn't provide enough evidence to conclude that 
    there's a significant difference in awareness levels between single and 
    married people. It's possible that there is a small difference, but the 
    sample size or variability in the data might be too high to detect it 
    with this test."""
    st.write(text)

    st.subheader('T-test on the Level of Awareness Grouped by Socio-economic Status')
    lower_awareness = df1[df1['SEStatus'] == 'Low Income Class (Between ?9,100 to ?18,200)']['Awareness']
    mid_awareness = df1[df1['SEStatus'] == 'Lower Middle Income Class (Between ?18,201 to ?36,400)']['Awareness']
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(lower_awareness, mid_awareness)
    # Print the results
    st.write("t-statistic: {:.2f}".format(t_statistic))
    st.write("p-value: {:.4f}".format(p_value))    

    text = """The results of the independent sample t-test indicate that there is not a 
    statistically significant difference (p-value > 0.05) in the level of awareness 
    between the low and middle socioeconomic status groups. In other words, we 
    cannot reject the null hypothesis that the mean level of awareness is the same 
    for both groups.
    \nIt is important to note that the t-statistic is negative, but we cannot interpret
    the direction of the difference from the t-statistic alone. The t-statistic only 
    tells us about the magnitude and direction of the difference relative to the null 
    hypothesis of no difference."""
    st.write(text)

    st.subheader('ANOVA test on the Level of Awareness Grouped by Age')
    g1 = df1.loc[(df1['Age'] =='18-35 years old'), 'Awareness']
    g2 = df1.loc[(df1['Age'] =='36-50 years old'), 'Awareness']
    g3 = df1.loc[(df1['Age'] =='above 50 years old'), 'Awareness']
    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2, g3)
    # Print the results
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))  

    text = """The reported F-statistic (0.73) and p-value (0.4826) suggest that there is no 
    statistically significant difference in the mean level of awareness between the 
    age groups that were compared.
    \nF-statistic (0.73): This value is relatively low, indicating that the 
    variances of the awareness levels in the age groups are similar. 
    In an ANOVA test, a high F-statistic would suggest unequal variances.
    \np-value (0.4826): This value is greater than a common significance 
    level (0.05). Since the p-value is not less than the significance level, 
    we fail to reject the null hypothesis. The null hypothesis, in this case, is 
    that the means of the awareness level are equal between the age groups."""
    st.write(text)

    # Fit the ANOVA model
    model = ols('Awareness ~ C(Age)', data=df).fit()
    # Perform ANOVA    
    anova_table = sm.stats.anova_lm(model, typ=2)
    # Display ANOVA table

    # Print the ANOVA table
    st.write("ANOVA Table - Source of variation: Age")
    st.write(anova_table)

    st.subheader('ANOVA test on the Level of Awareness Grouped by Educational Attainment')
    g1 = df1.loc[(df1['Educ'] =='Elementary Graduate'), 'Awareness']
    g2 = df1.loc[(df1['Educ'] =='High School Graduate'), 'Awareness']
    g3 = df1.loc[(df1['Educ'] =='College Graduate'), 'Awareness']
    g4 = df1.loc[(df1['Educ'] =="Master's Graduate"), 'Awareness']
    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2, g3, g4)
    # Print the results
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))

    text = """The results of the ANOVA test suggest that there is not a 
    statistically significant difference in the level of awareness between the four 
    education attainment groups.
    \nF-statistic (1.62): The F-statistic is used to compare the variances between 
    groups (in this case, education attainment groups) to the variance within the 
    groups.
    \np-value (0.1889): This is the key value for interpreting the results. A p-value 
    is the probability of observing a test statistic (like the F-statistic) as extreme 
    as the one we calculated, assuming there is no real difference between the groups 
    (the null hypothesis). In general, a smaller p-value indicates stronger evidence 
    against the null hypothesis. Since the p-value (0.1889) is greater than the commonly
    used significance level of 0.05, we fail to reject the null hypothesis. 
    In other words, the data doesn't provide enough evidence to conclude that there 
    is a statistically significant difference in the level of awareness among the 
    four education attainment groups."""
    st.write(text)

    # Fit the ANOVA model
    model = ols('Awareness ~ C(Educ)', data=df).fit()
    # Perform ANOVA    
    anova_table = sm.stats.anova_lm(model, typ=2)
    # Display ANOVA table

    # Print the ANOVA table
    st.write("ANOVA Table - Source of variation: Educational Attainment")
    st.write(anova_table)    

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

    # Create a container
    container = st.container()

    # Create the figure and axes objects    
    fig, ax = plt.subplots()  # Create a figure and a single axes
    # Create a bar chart of the frequency using seaborn
    p = sns.barplot(x=col_counts.index, y=col_counts.values)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    # Rotate x-axis labels for better readability
    plt.setp(p.get_xticklabels(), rotation=90)    
    plt.title('Frequencies of ' + column)
    plt.tight_layout()  # Prevent overlapping elements    

    # Display the plot within the container
    container.pyplot(fig)

    # Add CSS to container using st.write (adjust width as needed)
    st.write("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stContainer {
        width: 200px; 
        margin: auto;
        }
    </style>
    """, unsafe_allow_html=True)


def plot_usage_by(df, column):

    # Create a container
    container = st.container()

    # Create the figure and axes object
    fig, ax = plt.subplots()

    # Create the countplot directly on the provided axes
    p = sns.countplot(x=column, data=df, hue='awarenesslevel', palette='bright', ax=ax)

    # Set the title and adjust layout
    ax.set_title("Awareness Level Grouped by " + column, fontsize=14)
    # Rotate x-axis labels for better readability
    plt.setp(p.get_xticklabels(), rotation=90)

    # Display the plot within the container
    container.pyplot(fig)

    # Add CSS to the container using st.write (adjust width as needed)
    st.write("""
    <style>
        .plot-container {
        width: 400px;  /* Adjust width here */
        margin: auto;
        }
    </style>
    """, unsafe_allow_html=True)


#run the app
if __name__ == "__main__":
    app()
