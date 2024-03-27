#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr
import scipy.stats as stats

# Define the Streamlit app
def app():

    st.subheader('Statistical Analysis on the Relationship of Inflation Awareness and Inflation Perception')
    text = """This part of the data app sought to answer this research question:
    Is there a significant correlation between job order workers' level of awareness of inflation 
    and their perception of its impact on their financial well-being?"""
    with st.expander("About the research question. CLick to expand."):
        st.write(text)

    df = pd.read_csv('inflation-final.csv', header=0)


    st.subheader("Spearman's rank correlation coefficient")
    text = """Spearman's rank correlation coefficient, denoted by the Greek letter rho (ρ), is a statistical 
    test used to measure the strength and direction of a monotonic relationship between two sets of data. 
    Unlike Pearson's correlation coefficient, which assumes a linear relationship, 
    Spearman's rank only cares about the order or rank of the data points.
    Monotonic Relationship: It assesses how well the relationship between two variables can be 
    described by a trend that consistently increases or decreases (either always going 
    up or always going down).
    Strength: The coefficient produces a value between 1 and -1. A value of 1 indicates a 
    perfect positive monotonic relationship (as one variable increases, the other always increases). 
    A value of -1 indicates a perfect negative monotonic relationship (as one variable increases, 
    the other always decreases). A value of 0 means no monotonic relationship.
    \nDirection: The positive or negative value reflects the direction of the monotonic trend."""
    st.write(text)

    st.subheader('Spearman Rank Order Correlation Coefficient of Awareness and Perception Levels')

    # Assuming your dataframe is called 'df'
    awareness_levels = df['awarenesslevel']
    perception_levels = df['perceptionlevel']

    # Calculate Spearman's rank correlation coefficient
    spearman_coeff, p_value = spearmanr(awareness_levels, perception_levels)

    # Print the results
    st.write("Spearman Rank Correlation Coefficient: {:.2f}".format(spearman_coeff))
    st.write("p-value: {:.4f}".format(p_value))
    text = """he Spearman rank correlation coefficient of 0.52 indicates a moderate 
    positive correlation between awareness of inflation and perception of 
    inflation among job order workers. This means that as awareness scores increase, 
    perception scores also tend to increase, but the relationship is not very strong.
    \nThe p-value of 0.0000 suggests that this correlation is statistically 
    significant.  In other words, it is very unlikely that this observed 
    correlation is due to chance alone.
    \nStrength: A coefficient of 0.52 falls within the range generally considered a 
    "moderate" correlation. There's a noticeable, but not extremely strong, 
    relationship between the two variables.
    \nDirection: The positive sign indicates that higher awareness of inflation 
    is associated with higher perception of its impact on financial well-being. 
    In other words, as awareness scores go up, perception scores also tend to go up.
    Significance: The p-value of 0.0000 is less than the typical significance level of 0.05. This means we can reject the null hypothesis that there's no correlation between awareness and perception. There's strong evidence to suggest a real association between these two factors in the data.
    \nCorrelation doesn't imply causation. Just because awareness and perception 
    are correlated doesn't necessarily mean that one causes the other. 
    There might be other factors influencing both variables.
    The strength of the correlation (moderate) suggests a trend, but there's also 
    a significant amount of variation. Not everyone with higher awareness will 
    have a higher perception of inflation's impact. The Spearman rank coefficient 
    suggests a meaningful positive association between awareness of inflation 
    and perception of its impact on financial well-being among job order workers at 
    the state university. However, it's important to consider the limitations of 
    correlation and explore the data further to understand the underlying relationship."""

#run the app
if __name__ == "__main__":
    app()
