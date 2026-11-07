# Allowed: 
# - NumPy, but only for arrays and basic linear algebra
# - Matplotlib, for plots and figures
import sys
import os



# get the directory of the above folder
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_loader import clean_and_save_data


def run_linear_regression():
    cleaned_data = clean_and_save_data()

    print(f"Loaded {len(cleaned_data)} rows ready for Linear Regression!")

#----

run_linear_regression()