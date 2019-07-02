import pickle

import pandas as pd

# Define dictionaries to map labels to numeric values and the other way around
label_strings = [
    'blurry-nothing', 'colon-clear', 'dyed-lifted-polyps',
    'dyed-resection-margins', 'esophagitis', 'instruments', 'normal-cecum',
    'normal-pylorus', 'normal-z-line', 'out-of-patient', 'polyps',
    'retroflex-rectum', 'retroflex-stomach', 'stool-inclusions',
    'stool-plenty', 'ulcerative-colitis'
]
label_str_to_num = dict(zip(label_strings, range(16)))
label_num_to_str = dict(zip(range(16), label_strings))


def load_feature_df_and_relevant_data_from_pickles():
    feature_df = pd.read_pickle("./feature_df.pickle")

    data_pkl_filename = "labels_and_index_conversions.pickle"
    data_pickle_file = open(data_pkl_filename, "rb")
    (labels, index_to_feature_path, index_to_img_path) = \
        pickle.load(data_pickle_file)
    data_pickle_file.close()

    return (feature_df, labels, index_to_feature_path, index_to_img_path)


def convert_label_str_to_num(label_str):
    return label_str_to_num[label_str]


def convert_label_num_to_str(label_num):
    return label_num_to_str[label_num]
