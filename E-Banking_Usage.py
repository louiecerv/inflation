#Input the relevant libraries
import streamlit as st

# Define the Streamlit app
def app():
    
    text = """E-banking Usage level and Influence on Spending Habits Among College of Business and Management Students"""
    st.subheader(text)

    text = """An udergraduate thesis at the College of Business and Management:"""
    st.write(text)  
    st.markdown("Baldevarona, A. M. G., Banggud, R. B., Gronifillo, G. C., Honrado, S. R. B., & Jamelarin,  R. J. E. L. *E-banking Usage Level and Influences on Spending Habits Among College of Business and Management students.* Unpublished undergraduate thesis. (2024)")

    st.image('e-banking.jpg', caption="The E-banking Usage")

    text = """\nThis dataset investigated the factors that affect the e-banking usage and 
    the influences in the spending habits among College of Business Management (CBM) students at 
    West Visayas State University (WVSU).
    Features:
    family_income (categorical): This feature represents the student's family 
    income level. It is divided into categories based on a pre-defined income range.
    Sex (binary): This feature indicates the student's sex, coded as "male" or "female."
    course (categorical): This feature specifies the student's academic program within CBM. 
    Label: e_banking_usage (binary): This variable indicates the student's level of 
    e-banking usage. It is coded as categories of 'very high', 'high', 'moderate', 'low' and 'very low'."""
    with st.expander("Click to view Data App Description"):
        st.write(text)


#run the app
if __name__ == "__main__":
    app()
