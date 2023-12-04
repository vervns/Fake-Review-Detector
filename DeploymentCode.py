
import streamlit as st
import pickle
from textblob import TextBlob
from nltk.corpus import stopwords
import string  

page = st.sidebar.selectbox("Select a Page", ("Explore", "Predict"))
if page == "Predict":

    def textP(review):
        nopunc = [char for char in review if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    with open('model1_NaiveBayes.pkl', 'rb') as model_file:
        model1 = pickle.load(model_file)

    with open('model2_RandomForest.pkl', 'rb') as model_file:
        model2 = pickle.load(model_file)

    with open('model3_DecisionTree.pkl', 'rb') as model_file:
        model3 = pickle.load(model_file)

    with open('model4_SVM.pkl', 'rb') as model_file:
        model4 = pickle.load(model_file)

    with open('model5_LogisticRegression.pkl', 'rb') as model_file:
        model5 = pickle.load(model_file)

    st.title("Fake Review Detector")

    selected_model = st.selectbox("Select Model:", ["Naive Bayes", "Random Forest", "Decision Tree", "SVM", "Logistic Regression"])

    model_description = {
        "Naive Bayes": "A probabilistic classification algorithm, with accuracy of 85.46%",
        "Random Forest": "An ensemble learning method using decision trees, with accuracy of 84.88%",
        "Decision Tree": "A tree-like model for classification and regression, with accuracy of 74.14%",
        "SVM": "A support vector machine classifier, with accuracy of 89.1%",
        "Logistic Regression": "A linear classification algorithm, with accuracy of 86.5%"
    }

    if selected_model:
        st.markdown(f"**Description**: {model_description[selected_model]}")

    user_input = st.text_input("Enter review:", )

    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                if selected_model == "Naive Bayes":
                    prediction = model1.predict([user_input])[0]
                elif selected_model == "Random Forest":
                    prediction = model2.predict([user_input])[0]
                elif selected_model == "Decision Tree":
                    prediction = model3.predict([user_input])[0]
                elif selected_model == "SVM":
                    prediction = model4.predict([user_input])[0]
                elif selected_model == "Logistic Regression":
                    prediction = model5.predict([user_input])[0]


            st.subheader("Prediction:")
            if prediction == 'CG':
                with st.container():
                    st.markdown('<div style="background-color: red; color: white; padding: 10px; border-radius: 5px;">Review is Fake</div>', unsafe_allow_html=True)
            elif prediction == 'OR':
                with st.container():
                    st.markdown('<div style="background-color: green; color: white; padding: 10px; border-radius: 5px;">Review is Real</div>', unsafe_allow_html=True)

elif page == "Explore":

    st.title("Detecting Fake Reviews on E-Commerce Platforms using NLP Techniques")
    st.markdown("This system uses Natural Language Processing (NLP) methods to create a dependable system that can spot false reviews on e-commerce websites. This will improve internet commerce's transparency and level of confidence. Because customers know the reviews they're reading are authentic, more sales and satisfied customers may result. Customers' improved reputations and growing trust in businesses might be advantageous for businesses.")

    st.markdown("**Data Source:** [Data used for this project](https://osf.io/tyue9/)")
    st.markdown("___")

    st.subheader("Data Visualization")

    selected_image = st.selectbox("Choose an Image to Display", ["Category Counts", "Rating Counts", "Text Length"])

    if selected_image == "Category Counts":
        st.image('Category Counts.png', use_column_width=True, caption="Category Counts")
    elif selected_image == "Rating Counts":
        st.image('Rating Counts.png', use_column_width=True, caption="Rating Counts")
    elif selected_image == "Text Length":
        st.image('Text Length.png', use_column_width=True, caption="Text Length")
    


# # streamlit run FYP_DeploymentCode.py

# Fake review: These done fit well and look great.  I love the smoothness of the edges and the extra
# Real review: Bought 2 and sent one back because it didn't work