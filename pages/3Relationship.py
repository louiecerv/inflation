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
    st.write("p-value: {:.8f}".format(p_value))

#run the app
if __name__ == "__main__":
    app()
