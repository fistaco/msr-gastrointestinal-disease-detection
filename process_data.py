import numpy as np
import pandas as pd
import os

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
columns = [f"{col}_{i}"
           for i in range(feature_column_amnts[col])
           for col in given_features]
columns.append("label")  # Save the class label
columns.append("filepath")  # Save the image filepath for each row

df = pd.DataFrame(
    np.zeros((images_amnt, features_amnt), dtype=float),
    columns=columns
)
df_index_to_filepath = {}

# Iterate over all images' feature files in each class directory and add them
# to the df.
i = 0  # Keep track of the current df index
directories = [f"./data/Medico_2018_development_set_features/{dirname}"
               for dirname
               in classes]
for dir_str in directories:
    directory = os.fsencode(dir_str)

    for file in os.listdir(directory):
        # Save the filepath for this df index
        filename = os.fsdecode(file)
        filepath = f"{dir_str}/{filename}"
        df_index_to_filepath[i] = filepath

        # Save the features for this df record
        feature_file = open(filepath, "r")

        # Each line is a feature name followed by comma-separated values
        lines = feature_file.readlines()
        for (index, line) in enumerate(lines):
            # Strip the newline character and the feature name + colon
            feature_name = given_features[index]
            lines[index] = lines[index].strip(f"\n:{feature_name}")

            # Now add the comma-separated values to this feature's columns
            feature_vals = [float(f) for f in lines[index].split(",")]
            for (j, val) in enumerate(feature_vals):
                col = f"{feature_name}_{j}"
                df.iloc[i][col] = val

        feature_file.close()

        i += 1
