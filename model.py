import numpy as np
import pandas as pd
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Load train and test data
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
y_train=train_data.target

#A quick look at our data
train_data[train_data["target"] == 0]["text"].values[0:100]

#Breaking train data into two parts using train_test_split
from sklearn.model_selection import train_test_split
train_train_data,val_train_data,train_y_train,val_y_train=train_test_split(train_data,y_train,random_state=1)

#Scikit-learn's CountVectorizer is used to count the words in each tweet and turn them into data so that our machine learning model can process
count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train_train_data["text"])

print(train_vectors[0].todense().shape)
print(train_vectors[0].todense())

#In val_vectors ,using only .transform() so that the tokens in the train vectors are the only ones mapped to the val vectors
val_vectors = count_vectorizer.transform(val_train_data["text"])

print(val_vectors[0].todense().shape)
print(val_vectors[0].todense())

#Building a model
#we're assuming here is a linear connection. So let's build a linear model and see!
# Our vectors are really big, so we want to push our model's weights
# toward 0 without completely discounting different words - ridge regression
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_y_train, cv=3, scoring="f1")
print(scores)

clf.fit(train_vectors, train_y_train)
predicted_val=clf.predict(val_vectors)
print(predicted_val)

#Calculating absolute error of predicted value and the target value in train dataset
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(predicted_val, val_y_train)
print(val_mae)

#CountVectorizer is used to count the words in each tweet in test dataset
test_vectors = count_vectorizer.transform(test_data["text"])

print(test_vectors[0].todense().shape)
print(test_vectors[0].todense())

#Making the prediction
val_predict=clf.predict(test_vectors)
print(val_predict)

#Exporting predicted values into the submission file
submission = pd.read_csv('submission.csv')
submission["target"] = val_predict
submission.head()