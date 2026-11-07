import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MY_STUDENT_ID = 40278603

# Load raw data
data_raw = pd.read_csv("../data/covtype-10.csv")

# Select 5 types at random using my student id
unique_cover_types_df = pd.DataFrame(data_raw['Cover_Type'].unique())
selected_cover_types = unique_cover_types_df.sample(n=5, random_state=MY_STUDENT_ID)[0].tolist()

# Filter only selected types
data_selected_types_only = data_raw[data_raw['Cover_Type'].isin(selected_cover_types)]

# Selecting 2500 samples of each class at random
data_clean = data_selected_types_only.groupby("Cover_Type").sample(n=2500, random_state=MY_STUDENT_ID)

# Store clean data to csv
data_clean.to_csv("../data/clean_covtype.csv")

# My types are: [3, 5, 1, 6, 2]
# 1 Spruce / fir (Cue: Traditional pine)
# 2 Lodgepole Pine (Cue: Taller than traditional)
# 3 Ponderosa Pine (Cue: Wider than traditional)
# 5 Aspen (Cue: White wood, yellow leafs)
# 6 Douglas Fir (Cue: Similar to traditional.)