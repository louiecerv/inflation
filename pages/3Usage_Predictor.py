#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Define the Streamlit app
def app():

    if "gender" not in st.session_state:
        st.session_state.gender = ''
    if "yearlevel" not in st.session_state:
        st.session_state.yearlevel = ''
    if "course" not in st.session_state:
        st.session_state.course = ''
    if "income" not in st.session_state:
        st.session_state.income = ''
    if "user_inputs" not in st.session_state:
        st.session_state.user_inputs = ''

    st.write("""The trained classification model will predict the E-Banking Usage Level 
    from the information provided.""")
    
    st.subheader('User Information')
    update_selections()

    if st.button("Start"):
        st.write('Unencoded user inputs:')
        user_inputs = np.array(st.session_state['user_inputs'])
        st.write(user_inputs)

        st.write('Encoded user inputs:')
        le_list = st.session_state["le_list"]
        encoded = []
        i = 0
        for value in user_inputs[0]:
            result = le_list[i].transform([value])
            encoded.append(result)
            i = i + 1
        encoded = pd.DataFrame(encoded)
        st.write(encoded.transpose())
        predicted =  st.session_state["clf"].predict(encoded.transpose())
        level = ''
        if predicted==0:
            level = 'Very Low'
        elif predicted==1:
            level = 'Low'
        elif predicted==2:
            level = 'Moderate'
        elif predicted==3:
            level = 'High'
        else:        
            level = 'Very High'

        result = 'The predicted Usage Level: ' + level
        st.subheader(result)

        text = """The potential uses of E-Banking Usage level prediction:
        \n1. Targeted Marketing and Promotions:
        Banks can leverage the model to identify customer segments with
        high e-banking usage propensity. Based on these segments, banks 
        can design targeted marketing campaigns promoting features and 
        benefits of their e-banking platforms.This can lead to increased 
        adoption and usage of e-banking services.
        \n2. Fraud Detection and Risk Management:
        The model can be used to identify patterns in e-banking behavior 
        that deviate from a user's usual activity. This can be helpful in 
        flagging potentially fraudulent transactions and mitigating financial risks.
        \n3. Personalized User Experience:
        By understanding a user's e-banking habits, the model can help 
        personalize their online banking experience. For instance, 
        the platform can prioritize features most relevant to the user's 
        needs and suggest functionalities they might find helpful based 
        on their predicted usage patterns.
        \n4. Customer Segmentation and Retention:
        The model can be used to segment customers based on their 
        e-banking usage levels. Banks can then tailor their communication and service
        offerings to cater to the specific needs of each segment.
        This can improve customer satisfaction and retention.
        \n5. Product Development and Innovation: Insights from the model 
        can inform the development of new e-banking features and functionalities. 
        Banks can prioritize features that cater to the needs of high 
        e-banking usage segments, potentially leading to a more user-friendly 
        and adopted platform."""
        st.write(text)
    
def update_selections():
    gender = st.selectbox('Gender:', ['Male', 'Female'])
    st.session_state['gender'] = gender
    yearlevel =  st.selectbox('Year Level: ', ['First Year', 'Second Year', 'Third Year', 'Fourth Year'])
    st.session_state['yearlevel'] = yearlevel
    course =  st.selectbox('Course: ', ['BSTM', 'BSCM', 'BSBA', 'BSHM'])
    st.session_state['course'] = course
    income =  st.selectbox('Family Income: ', ['Php 20 000 and Below', 'Above Php 60 000', 'Php 20 001 to Php 60 000'])
    st.session_state['income'] = income
    
    st.session_state.user_inputs = [[gender, yearlevel, course, income]]
#run the app
if __name__ == "__main__":
    app()
