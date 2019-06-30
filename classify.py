import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from helpers import *

# Load our pre-processed data from our pickle file
(feature_df, labels, df_index_to_feature_path, df_index_to_img_path) = \
    load_feature_df_and_relevant_data_from_pickles()

# Convert labels to numeric values for classification
labels = np.array([convert_label_str_to_num(label) for label in labels])

# Split in train and test data
x_train, x_test, y_train, y_test = \
    train_test_split(feature_df, labels, test_size=0.25)

# Define candidate classifiers to test
clfs = {
    "SGD": SGDClassifier(random_state=7),
    "LogReg": LogisticRegression(multi_class="ovr", solver="lbfgs"),
    "LinearSVC": LinearSVC(random_state=7),
    "RandForest": RandomForestClassifier(random_state=7)
}

# Train and evaluate each classifier
for (id, clf) in clfs.items():
    print(f"Training {id} classifier!")
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1 = f1_score(y_test, y_pred, average="micro")
    print(f"{id}: accuracy={acc}, precision={prec}, recall={recall}, f1={f1}")
