import pickle

import pandas as pd


def load_feature_df_and_relevant_data_from_pickles():
    feature_df = pd.read_pickle("./feature_df.pickle")

    data_pkl_filename = "labels_and_index_conversions.pickle"
    data_pickle_file = open(data_pkl_filename, "rb")
    (labels, index_to_feature_path, index_to_img_path) = \
        pickle.load(data_pickle_file)
    data_pickle_file.close()

    return (feature_df, labels, index_to_feature_path, index_to_img_path)
