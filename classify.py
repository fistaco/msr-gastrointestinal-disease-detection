import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

from helpers import load_feature_df_and_relevant_data_from_pickles

# Load our pre-processed data from our pickle file
(feature_df, labels, df_index_to_feature_path, df_index_to_img_path) = \
    load_feature_df_and_relevant_data_from_pickles()

# Split in train and test data
x_train, x_test, y_train, y_test = \
    train_test_split(feature_df, labels, test_size=0.25)

# Bernoulli Naive Bayes classification
bernoulli_clf = BernoulliNB()
bernoulli_clf.fit(x_train, y_train)

y_pred = bernoulli_clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)

# Multi-class logistic regression classification
logreg_clf = LogisticRegression(multi_class="ovr")
logreg_clf.fit(x_train, y_train)

y_pred = logreg_clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
