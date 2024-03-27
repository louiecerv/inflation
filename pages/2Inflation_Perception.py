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
    text = """This part of the data app sought to answer this research question:
    Are there significant differences in the level of perception of inflation 
    among job order workers in a state university when classified according to age, 
    sex, civil status, educational attainment, and socioeconomic status?"""
    with st.expander("About the research question. CLick to expand."):
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
 
    st.subheader('T-test on the Level of Perception Grouped by Sex')
    male_awareness = df1[df1['Sex'] == 'Male']['Perception']
    female_awareness = df1[df1['Sex'] == 'Female']['Perception']
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(male_awareness, female_awareness)
    # Print the results
    st.write("t-statistic: {:.2f}".format(t_statistic))
    st.write("p-value: {:.4f}".format(p_value))  

    text = """The results of the independent sample t-test indicate that 
    there is no statistically significant difference in the levels of 
    perception of inflation between males and females. This is because 
    the p-value (0.2276) is greater than the significance level set at 0.05. 
    The data does not provide enough evidence to conclude that males and 
    females have different perceptions of inflation. It is possible that 
    there is a difference, but this study could not detect it with 
    enough certainty."""
    st.write(text)

    st.subheader('T-test on the Level of Perception Grouped by Civil Status')
    single_awareness = df1[df1['Civil Status'] == 'Single']['Perception']
    married_awareness = df1[df1['Civil Status'] == 'Married']['Perception']
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(single_awareness, married_awareness)
    # Print the results

    st.write("t-statistic: {:.2f}".format(t_statistic))
    st.write("p-value: {:.4f}".format(p_value))  

    text = """The results of the independent samples t-test indicate that there 
    is no statistically significant difference between the levels of perception of 
    inflation for single and married people. 
    \nt-statistic (0.5268): This value doesn't tell us much by itself in this 
    context. It's a test statistic used to compare the means of two groups, 
    but without a critical value or p-value, we can't determine significance.
    \np-value (0.5993): This is the key finding. A p-value greater than 0.05 
    suggests that we fail to reject the null hypothesis. In this case, the null 
    hypothesis is that the average perception of inflation is the same for 
    single and married people."""
    st.write(text)

    st.subheader('T-test on the Level of Perception Grouped by Socio-economic Status')
    lower_awareness = df1[df1['SEStatus'] == 'Low Income Class (Between ?9,100 to ?18,200)']['Perception']
    mid_awareness = df1[df1['SEStatus'] == 'Lower Middle Income Class (Between ?18,201 to ?36,400)']['Perception']
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(lower_awareness, mid_awareness)
    # Print the results
    st.write("t-statistic: {:.2f}".format(t_statistic))
    st.write("p-value: {:.4f}".format(p_value))  

    text = """The results of the independent sample t-test indicate that there is no 
    statistically significant difference in the levels of perception of inflation 
    between low-income and lower-middle-income individuals. 
    \nT-statistic: 0.5692 - This value tells the direction and strength of the 
    difference between the two groups' average perception of inflation. In this case, 
    the t-statistic is very close to 0, indicating a negligible difference between 
    the means.
    \nP-value: 0.5704 - The p-value represents the probability of observing a 
    t-statistic as extreme or more extreme than the one obtained, assuming the null 
    hypothesis (no difference between the groups) is true. A high p-value (greater 
    than 0.05) suggests that the observed difference is likely due to chance.
    Since the p-value (0.5704) is greater than the commonly used significance level 
    of alpha = 0.05, we fail to reject the null hypothesis. In other words, the 
    t-test doesn't provide enough evidence to conclude that there's a statistically 
    significant difference in how low-income and lower-middle-income people perceive 
    inflation. The t-test suggests that there's no clear distinction between how 
    low-income and lower-middle-income individuals view inflation. The small 
    difference observed between the two groups could be due to random chance."""
    st.write(text)

    st.subheader('ANOVA test on the Level of Perception Grouped by Age')
    g1 = df1.loc[(df1['Age'] =='18-35 years old'), 'Perception']
    g2 = df1.loc[(df1['Age'] =='36-50 years old'), 'Perception']
    g3 = df1.loc[(df1['Age'] =='above 50 years old'), 'Perception']
    # Perform one-way ANOVA test
    F_statistic, p_value = stats.f_oneway(g1, g2, g3)
    # Print the results
    st.write("F-statistic: {:.2f}".format(F_statistic))
    st.write("p-value: {:.4f}".format(p_value))   

    text = """The results of the ANOVA test indicate that there is not 
    statistically significant evidence to reject the null hypothesis. 
    In simpler terms, this means we don't have enough evidence to conclude 
    that there are differences in the level of perception of inflation between 
    the three age groups.
    \nF-statistic (0.39): This statistic compares the variance between the 
    groups (variance explained by the independent variable, age) to the 
    variance within the groups (unexplained variance). A low F-statistic, 
    like the one here (0.39), suggests that the variance between the groups 
    is relatively small compared to the variance within the groups.
    \np-value (0.6765): This value represents the probability of observing a 
    test statistic (F-statistic in this case) as extreme or more extreme than
    what was obtained, assuming the null hypothesis is true. A high p-value 
    (greater than 0.05 indicates that the observed F-statistic is not statistically 
    unusual and could have easily occurred by chance, even if there were no
    real differences between the groups. Based on this ANOVA test, we cannot 
    say that there's a statistically significant difference in the level of 
    perception of inflation between the three age groups. It's possible that there
    might be some real differences, but the sample data we have is not strong 
    enough to detect them definitively. """
    st.write(text)

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
    text = """The results of the ANOVA test indicate that there is no significant
    difference in the level of perception of inflation between the four education
    attainment groups.
    \nF-statistic (0.39): This statistic measures the ratio of variances between 
    groups compared to the variance within groups. A low F-statistic, like 0.39 
    in this case, suggests that the variances between the groups are similar to 
    the variances within the groups.
    \np-value (0.7626): This value represents the probability of observing an 
    F-statistic as extreme or more extreme than the one obtained, assuming the 
    null hypothesis (no difference between groups) is true. A high p-value 
    (greater than 0.05, which is a commonly used significance level) suggests that 
    the observed F-statistic is likely due to chance, and we fail to reject the 
    null hypothesis. Based on this ANOVA test, we can't conclude that there's a 
    statistically significant difference in how job order workers from different 
    education attainment groups perceive inflation."""
    st.write(text) 

def mean_std(df, column_name):
    grouped_data = df.groupby(column_name)

    # Calculate mean and standard deviation for each group
    results = grouped_data['Perception'].agg(['mean', 'std'])
    # Print the results
    st.write(results)

def display_freqs(df, column):
     # Calculate counts for each sex
    scounts = df[column].value_counts()
    # Print the frequency table
    st.write(scounts)
    custom_colours = ['#ff7675', '#74b9ff']
    # Define labels and sizes for the pie chart
    sizes = [scounts[0], scounts[1]]

    # Create a figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Create the pie chart
    wedges, texts, autotexts = ax1.pie(sizes, labels=df[column], autopct='%1.0f%%',
                                    startangle=140, colors=custom_colours,
                                    textprops={'fontsize': 10}, explode=[0, 0.05])
    ax1.set_title('Distribution of ' + column)

    # Create the bar chart using seaborn
    ax2 = sns.barplot(x=df[column].unique(), y=df[column].value_counts(), ax=ax2, palette='viridis')
    ax2.set_xlabel(column)
    ax2.set_ylabel('Frequency')
    ax2.set_title(column + ' Count')

    # Tight layout to prevent overlapping elements
    plt.tight_layout()
    st.pyplot(fig)

def plot_usage_by(df, column):

    # Create a container
    container = st.container()

    # Create the figure and axes object
    fig, ax = plt.subplots()

    # Create the countplot directly on the provided axes
    p = sns.countplot(x=column, data=df, hue='perceptionlevel', palette='bright', ax=ax)

    # Set the title and adjust layout
    ax.set_title("Perception Level Grouped by " + column, fontsize=14)
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
