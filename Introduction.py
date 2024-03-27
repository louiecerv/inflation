#Input the relevant libraries
import streamlit as st

# Define the Streamlit app
def app():

    text = """Awareness and Perception on Inflation Among Job Order Workers in a State University"""
    st.subheader(text)

    text = """An udergraduate thesis at the College of Business and Management:"""
    st.write(text)  
    st.markdown("Balajadia, Arianne V., Comorro, Kyle Avefern B., Daria, Niña Paula C., Erazo, Josephine Aurea E., and Granada, Shaneen Joy A. *Awareness and Perception on Inflation Among Job Order Workers in a State University* Unpublished undergraduate thesis. (2024)")

    st.image('inflation.png', caption="Inflation Awareness and Perception")

    text = """This dataset focuses on understanding how job order workers at a state university 
    perceive and are aware of inflation. It includes information on various factors 
    that might influence their perception and awareness.
    \nFeatures:
    Age, Sex, Socio-ec-onomic status, Civil Status and Educational Attainment
    \nAwareness of Inflation (measured using a Likert Scale questionnaire) - 
    Scores range from "Strongly Disagree" (low awareness) to "Strongly Agree" (high awareness)
    \nPerception of Inflation (measured using a Likert Scale questionnaire) - 
    Scores range from "Strongly Disagree" (low awareness) to "Strongly Agree" (high awareness)
    \nAnalysis:
    The study employs statistical tests to investigate potential differences in awareness and 
    perception across the various groups defined by the demographic features.
    \nT-test: This test would likely be used to compare awareness and perception scores 
    between two groups, such as males vs. females or single vs. married workers.
    \nANOVA (Analysis of Variance): This test would be used to compare awareness and 
    perception scores across three or more groups, such as the different socio-economic status
    categories or educational attainment levels."""
    
    with st.expander("Click to view Data App Description"):
        st.write(text)


#run the app
if __name__ == "__main__":
    app()
