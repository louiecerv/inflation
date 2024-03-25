#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Define the Streamlit app
def app():

    st.sidebar.subheader('Select the classifier')

    # Create the selection of classifier
    options = ['Extra Trees Classifier', 'SVM', 'Decision Tree', 'Gradient Boosting', 'MLP Classifier']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option=='Extra Trees Classifier':        
        st.session_state.clf = ExtraTreesClassifier(n_estimators=200)
        st.session_state['selected_model'] = 0
    elif selected_option=='SVM':        
        st.session_state.clf = SVC(C=100.0)
        st.session_state['selected_model'] = 1
    elif selected_option == 'Decision Tree':
        st.session_state.clf = DecisionTreeClassifier()
        st.session_state['selected_model'] = 2
    elif selected_option == 'Gradient Boosting':
        st.session_state.clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3)
        st.session_state['selected_model'] = 3
    elif selected_option == 'MLP Classifier':
        st.session_state.clf = MLPClassifier(hidden_layer_sizes=(100,50), solver='adam', activation='relu', max_iter=200, random_state=42)
        st.session_state['selected_model'] = 4

    classifier = ''
    if st.session_state['selected_model'] == 0:     
        classifier = 'Extra Trees Classifier'
        report = """Extra Trees Classifier is a machine learning method for classification tasks. 
        It belongs to a family of algorithms called ensemble methods, which combine multiple
        models (in this case, decision trees) to get a better prediction than any single model.
        It builds multiple decision trees, but with randomness injected at two key points: 
        selecting features and choosing split points within those features. This randomness 
        helps reduce overfitting by making the trees more independent."""
        
    elif st.session_state['selected_model'] == 1:   
        classifier = "Support Vector Machine Classifier"
        report = """A support vector machine (SVM) is a machine learning algorithm that excels at 
        classifying data points.  SVMs find te line (or hyperplane in higher dimensions) 
        that maximizes the margin between the two classes. A wider margin translates to a 
        better separation and potentially better classification on unseen data. SVMs are 
        particularly useful for: High dimensional data: They can handle complex data 
        with many features. Smaller datasets: They can perform well even with limited 
        training data. Non-linear classification: By using kernel functions, 
        SVMs can handle data that isn't easily separable with a straight line."""
        
    elif st.session_state['selected_model'] == 2:  
        classifier = "Decision Tree Classifier"
        report = """A decision tree classifier is a machine learning algorithm that works 
        by building a tree-like model to classify data. It asks a series of questions 
        about the data's features at each branch of the tree, and based on the answers, 
        it directs the data down a specific path until it reaches a leaf node that 
        contains the predicted classification. 
        Easy to understand and interpret: The decision-making process is clear and 
        transparent, unlike some other machine learning models. Good for handling 
        high-dimensional data: They can work well with data that has many features. 
        Fast training time: Compared to some other algorithms, decision trees can be 
        trained relatively quickly."""

    elif st.session_state['selected_model'] == 3:   #Gradient Boosting
        classifier = "Gradient Boosting Classifier"
        report = """Gradient boosting classifier is a machine learning technique that 
        combines multiple weak models, like decision trees, into a single strong 
        learner. Here's the gist of how it works:
        \nTrain a weak learner on the data. Analyze the errors from that model's 
        predictions. Train a second weak learner to specifically address those errors. 
        Combine the predictions from both weak learners. By combining these weak 
        learners, the gradient boosting classifier becomes much better at predicting 
        the target variable than any of the individual weak models could be on their own."""
    else:
        classifier = "MLP Classifier"
        report = """The performance of an MLP classifier on a classification task for 
        online banking usage can vary depending on several factors, including:
        \nData characteristics: The quality and quantity of data used to train the model 
        is crucial. Factors like the number of features, their relevance, and the balance 
        between different classes all affect performance.
        \nMLP architecture: The number of hidden layers, neurons per layer, and activation functions 
        can significantly impact the model's ability to learn complex patterns in the data. 
        Tuning these hyperparameters is essential for optimal performance.
        \nTraining process: The chosen optimization algorithm, learning rate, and number of 
        training epochs all influence how well the MLP generalizes to unseen data"""
                 
    st.subheader('Performance of the ' + classifier)
    # Create the expander with a descriptive label
    with st.expander("Click to unfold and view the classifier performance details."):
        st.write(classifier)
        st.write(report)

    st.write("Click the button to start the test.")    

    if st.button("Begin Test"):
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        clf = st.session_state.clf

        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        # save the clf to the session state
        st.session_state['clf'] = clf

        st.subheader('Confusion Matrix')
        st.write('Confusion Matrix:')
        cm = confusion_matrix(y_test, y_test_pred)
        st.text(cm)
        text = """
        Diagonal elements: Ideally, these elements (very low predicted as very low, low 
        predicted as low, and so on) will have the highest values. This indicates a high 
        number of correct classifications for each class.
        \nOff-diagonal elements: These elements represent misclassifications. High values in
        these cells indicate the model frequently confuses between those specific classes.
        \nBy analyzing the confusion matrix, we can assess the performance of the 
        model for each class and identify areas for improvement, such as if the 
        model is consistently misclassifying "very low" data points as "low"."""
        with st.expander("Key elements in a Confusion Matrix"):
            st.write(text)

        st.subheader('Performance Metrics')
        st.text(classification_report(y_test, y_test_pred))

        text = """The classification_report function in scikit-learn provides a detailed breakdown
        of the performance of the classification model for a multi-class problem, 
        in this case, with five classes: "very low", "low", "moderate", "high", and "very high". 
        \nClass labels:  This section lists all the five classes present in the data.
        \nPrecision: This metric shows, for each class, the ratio of correctly predicted 
        positive cases (belonging to that class) to the total number of cases predicted 
        as positive by the model. In simpler terms, it tells you how often the model was 
        actually correct when it predicted a specific class (e.g., "very low").
        \nRecall: This metric, also known as True Positive Rate (TPR), shows, for each 
        class, the ratio of correctly predicted positive cases (belonging to that 
        class) to all actual positive cases in the data for that class. In simpler terms, 
        it tells how good the model was at identifying all the actual cases belonging 
        to a specific class (e.g., how many truly "very low" data points were correctly 
        classified).
        \nF1-score: This is a harmonic mean of precision and recall, combining both 
        metrics into a single score. A high F1-score indicates a good balance between 
        precision and recall.
        \nSupport: This column shows the total number of true instances for each class in 
        the test data. It shows the class distribution and identify potential
        issues due to class imbalance."""
        with st.expander("About the Classification Report"):
            st.write(text)


#run the app
if __name__ == "__main__":
    app()
