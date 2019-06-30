import os
import pickle

import cv2
import numpy as np
import pandas as pd

# Define class strings to iterate over all image directories
classes = [
    "blurry-nothing", "dyed-lifted-polyps", "esophagitis", "normal-cecum",
    "normal-z-line", "polyps", "retroflex-stomach", "stool-plenty",
    "colon-clear", "dyed-resection-margins", "instruments", "normal-pylorus",
    "out-of-patient", "retroflex-rectum", "stool-inclusions",
    "ulcerative-colitis"
]


def compute_custom_features(df, index_to_img_path):
    feature_computation_funcs = [
        compute_and_append_histogram_features
    ]

    # Iterate over each img in the df to find its filepath
    total_imgs = len(df)
    for (i, row) in df.iterrows():
        print(f"Computing features for image {i}/{total_imgs}")

        # Load image with openCV2
        img_path = index_to_img_path[i]
        img = cv2.imread(img_path)  # Img is loaded as a 3D pixel BGR matrix

        # Compute all features for this image and add them to the df
        for compute_and_add_feature in feature_computation_funcs:
            df = compute_and_add_feature(img, i, df)

    print(f"Done computing custom features!")

    return df


def compute_and_append_histogram_features(img, img_index, df):
    # Compute HSV hist
    hsv_channels = [0, 1, 2]
    # Choose few "hue" bins, because the images are mostly just blue or pink.
    # Choose many "value" bins, because of the many important shades of pink.
    hsv_bins = [4, 8, 12]
    hsv_ranges = [[0, 180], [0, 256], [0, 256]]
    hsv_hist = compute_hist(
        img, cv2.COLOR_BGR2HSV, hsv_channels, hsv_bins, hsv_ranges)

    # Compute YCrCb hist
    ycrcb_channels = [0, 1, 2]
    # We need to distinguish many brightness levels, but only want to detect
    # if an image is very blue or not otherwise.
    ycrcb_bins = [12, 2, 8]  
    ycrcb_ranges = [[0, 256], [0, 256], [0, 256]]
    ycrcb_hist = compute_hist(
        img, cv2.COLOR_BGR2YCrCb, ycrcb_channels, ycrcb_bins, ycrcb_ranges)

    # Append both histograms' values to the df as features
    df = df.append_hist_features(df, img_index, hsv_hist, "HSV")
    df = df.append_hist_features(df, img_index, ycrcb_hist, "YCrCb")

    return df


def compute_hist(bgr_img, color_space_func, channels, bins, ranges):
    # Convert the image to the desired color space
    img = cv2.cvtColor(bgr_img, color_space_func)

    # Return histogram as a 1D array so we can easily use it as features
    histogram = np.zeros(np.sum(bins))

    # Iterate over channels and append each channel hist to the full histogram
    for i in range(len(channels)):
        channel_hist = \
            cv2.calcHist([img], [channels[i]], None, [bins[i]], ranges[i])

        start = int(np.sum(bins[0:channels[i]]))
        end = start + bins[i]
        histogram[start:end] = channel_hist.flatten()

    return histogram


def append_hist_features(df, img_index, hist, color_space):
    for (i, bin_val) in enumerate(hist):
        column = f"{color_space}_{i}"
        df.loc[img_index, column] = bin_val

    return df
