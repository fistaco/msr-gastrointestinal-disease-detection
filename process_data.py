import numpy as np
import pandas as pd
import os
import pickle

# Build a DataFrame containing all images' features
classes = [
    "blurry-nothing", "dyed-lifted-polyps", "esophagitis","normal-cecum",
    "normal-z-line", "polyps", "retroflex-stomach", "stool-plenty",
    "colon-clear", "dyed-resection-margins", "instruments", "normal-pylorus",
    "out-of-patient", "retroflex-rectum", "stool-inclusions",
    "ulcerative-colitis"
]

# Construct a DataFrame (df) of 5293 images using the class direcotries' images
images_amnt = 5293
given_features = ["JCD", "Tamura", "ColorLayout", "EdgeHistogram",
                  "AutoColorCorrelogram", "PHOG"]
# Store the amount of columns for each given feature for easier processing
feature_column_amnts = {
    "JCD": 168,
    "Tamura": 18,
    "ColorLayout": 33,
    "EdgeHistogram": 80,
    "AutoColorCorrelogram": 256,
    "PHOG": 630
}
features_amnt = sum(feature_column_amnts.values())

# Create the required column names for the DataFrame
columns = []
for col in given_features:
    for i in range(feature_column_amnts[col]):
        columns.append(f"{col}_{i}")

df = pd.DataFrame(
    np.zeros((images_amnt, features_amnt), dtype=float),
    columns=columns
)
# Store labels separately for each df index
labels = np.full(images_amnt, "", dtype="U22")
# df["label"] = pd.Series(0*images_amnt, dtype=int)

# For each df index, remember the image and features filepaths for easy access
df_index_to_feature_filepath = {}
df_index_to_img_filepath = {}

# Iterate over all images' feature files in each class directory and add them
# to the df.
i = 0  # Keep track of the current df index
for dir_str in classes:
    # Define full paths so we can store them later
    full_feature_dir = f"./data/Medico_2018_development_set_features/{dir_str}"
    full_img_dir = f"./data/Medico_2018_development_set/{dir_str}"

    # Work with the feature dir for extraction
    directory = os.fsencode(full_feature_dir)

    for file in os.listdir(directory):
        print(f"Extracting features for image {i}/{images_amnt}...")

        # Save the relevant filepaths for this df index
        filename = os.fsdecode(file)
        feature_filepath = f"{full_feature_dir}/{filename}"
        img_filepath = f"{full_img_dir}/{filename}"
        df_index_to_feature_filepath[i] = feature_filepath
        df_index_to_img_filepath[i] = img_filepath

        # Save the features for this df record
        feature_file = open(feature_filepath, "r")

        # Each line is a feature name followed by comma-separated values
        lines = feature_file.readlines()
        for (index, line) in enumerate(lines):
            # Strip the newline character and the feature name + colon
            feature_name = given_features[index]
            lines[index] = lines[index].strip(f"\n:{feature_name}")

            # Add the comma-separated values to this feature's columns
            feature_vals = [float(f) for f in lines[index].split(",")]
            for (j, val) in enumerate(feature_vals):
                col = f"{feature_name}_{j}"
                df.iloc[i][col] = val

            # Store the label in the df
            labels[i] = dir_str

        feature_file.close()

        i += 1
print(f"Done extracting features!")

# Store the df, labels, and index conversions as pickle files to save time
df.to_pickle("./feature_df.pickle")
data_pkl_filename = "labels_and_index_conversions.pickle"
data_pickle_file = open(data_pkl_filename, "wb")
pickle.dump(
    (labels, df_index_to_feature_filepath, df_index_to_img_filepath),
    data_pickle_file
)
data_pickle_file.close()


def load_feature_df_and_relevant_data_from_pickles():
    feature_df = pd.read_pickle("./feature_df.pickle")

    data_pkl_filename = "labels_and_index_conversions.pickle"
    data_pickle_file = open(data_pkl_filename, "rb")
    (labels, index_to_feature_path, index_to_img_path) = \
        pickle.load(data_pickle_file)
    data_pickle_file.close()

    return (feature_df, labels, index_to_feature_path, index_to_img_path)
