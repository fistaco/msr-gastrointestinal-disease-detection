import os
import pickle

import cv2
import numpy as np
import pandas as pd

from helpers import *

# Define class strings to iterate over all image directories
classes = [
    "blurry-nothing", "dyed-lifted-polyps", "esophagitis", "normal-cecum",
    "normal-z-line", "polyps", "retroflex-stomach", "stool-plenty",
    "colon-clear", "dyed-resection-margins", "instruments", "normal-pylorus",
    "out-of-patient", "retroflex-rectum", "stool-inclusions",
    "ulcerative-colitis"
]


def compute_custom_features(df, index_to_img_path_dict):
    feature_computation_funcs = [
        # compute_and_append_histogram_features,
        compute_and_append_local_binary_patterns_features
    ]

    # Iterate over each img in the df to find its filepath
    total_imgs = len(df)
    for (i, row) in df.iterrows():
        print(f"Computing features for image {i}/{total_imgs}")

        # Load image with openCV2
        img_path = index_to_img_path_dict[i]
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
    df = append_hist_features(df, img_index, hsv_hist, "HSV")
    df = append_hist_features(df, img_index, ycrcb_hist, "YCrCb")

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


def compute_and_append_local_binary_patterns_features(bgr_img, img_index, df):
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # Define limits for our iteration over 16x16 pixel blocks
    block_size = 256  # Amount of pixels in a block in both dimensions
    (x_bound, y_bound) = (img.shape[0], img.shape[1])

    # The full LBP histogram is a concatenation of all cells' LBP histograms
    amnt_cells = (x_bound//block_size + 1) * (y_bound//block_size + 1)
    full_lbp_hist = np.zeros(amnt_cells * 256)
    full_lbp_idx = 0  # Track where we're adding cell histograms

    # Iterate over each pixel block to obtain a full LBP feature vector
    print("Computing LBP histograms for all cells...")
    for i in range(0, x_bound, block_size):
        for j in range(0, y_bound, block_size):
            cell = img[i:(i + block_size), j:(j + block_size)]

            # Store occurrence counts of each possible 8-bit pattern
            cell_lbp_hist = np.zeros(256)

            # Compute a LBP for each pixel in the cell
            for x in range(cell.shape[0]):
                for y in range(cell.shape[1]):
                    lbp = compute_local_binary_pattern(cell, x, y)

                    cell_lbp_hist[lbp] += 1

            # Append the computed cell lbp histogram to the full lbp hist
            full_lbp_hist[full_lbp_idx:(full_lbp_idx + 256)] = cell_lbp_hist
            full_lbp_idx += 256

    # Set the features in the df
    print(f"Appending features to the df...")
    for (i, hist_val) in enumerate(full_lbp_hist):
        column = f"LBP_{i}"
        print(i)
        df.loc[img_index, column] = hist_val

    return df


def compute_local_binary_pattern(cell, x, y, radius=3):
    centre_val = cell[x, y]

    # Define the 8 neighbour indices
    r = radius
    nb_locations = [(x + i, y + j) for i in [-r, 0, r] for j in [-r, 0, r]]
    nb_locations.remove((x, y))

    # Define cell boundaries based on its shape
    x_bound = cell.shape[0]
    y_bound = cell.shape[1]

    # Define a "comparison byte" based on the centre value being larger or
    # smaller than each of its 8 neighbours.
    comp_byte = 0
    for (i, nb_loc) in enumerate(nb_locations):
        shift = 8 - i - 1  # Start from the left side of the byte

        # Treat a value as smaller than ours if it's out of bounds
        nb_out_of_bounds = \
            nb_loc[0] >= x_bound or nb_loc[1] >= y_bound or \
            nb_loc[0] < 0 or nb_loc[1] < 0
        if nb_out_of_bounds:
            comp_byte &= (1 << shift)
            continue

        # Write 0 if the centre value is larger, write 1 otherwise. Use bitwise
        # operators to improve performance. Recall that each bit is initialised
        # at 0, so we only have to check if centre_val <= nv_val.
        nb_val = cell[nb_loc[0], nb_loc[1]]
        if centre_val <= nb_val:
            comp_byte &= (1 << shift)

    return comp_byte


if __name__ == "__main__":
    (df, labels, df_index_to_feature_path, df_index_to_img_path) = \
        load_feature_df_and_relevant_data_from_pickles()

    df = compute_custom_features(df, df_index_to_img_path)
    df.to_pickle("./feature_df_plus_color_hist_and_lbp_features.pickle")
