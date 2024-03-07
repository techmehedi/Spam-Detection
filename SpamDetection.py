import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

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
print(model.score(features_test, cat_test))