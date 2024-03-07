import pandas as pd #For reading and cleansing the data
from sklearn.model_selection import train_test_split # split data into train and test data set
from sklearn.feature_extraction.text import CountVectorizer # convert text to numerical format
from sklearn.naive_bayes import MultinomialNB # classifying the dataset
import streamlit as st #For creating the web app

# Read and Clean Data

data = pd.read_csv("/Users/mehedihasan/Downloads/spam.csv")

data.drop_duplicates(inplace=True)

data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])


# Categorize data using scikit (input set vs output set)

mess = data['Message']
cat = data["Category"]

(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess,cat, test_size=0.2)

# transform text into numerical data 

cv = CountVectorizer(stop_words='english')

features = cv.fit_transform(mess_train)

# Creating the Model

#classifying data (whether it's spam or not)

model = MultinomialNB()
model.fit(features, cat_train)

#Test Model

features_test = cv.transform(mess_test)

# Predict data

def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]  

st.header('Spam Detection')

input_mess = st.text_input('Paste Email or Message Here')

if st.button('Validate'):
    output = predict(input_mess)
    st.write("Prediction: ", output)