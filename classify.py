import numpy as np
import pandas as pd
from process_data import load_feature_df_and_relevant_data_from_pickles

(feature_df, labels, df_index_to_feature_path, df_index_to_img_path) = \
    load_feature_df_and_relevant_data_from_pickles()
