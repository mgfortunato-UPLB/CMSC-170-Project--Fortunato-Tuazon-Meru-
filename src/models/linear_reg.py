# Allowed: 
# - NumPy, but only for arrays and basic linear algebra
# - Matplotlib, for plots and figures

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def get_processed_directory():  # go to processed directory
    models_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(models_dir))
    return os.path.join(project_root, "data", "processed")


def userChoice(printSentence, options):

    # show all possible options
    print(f"\n--- {printSentence} ---")
    for index, opt in enumerate(options, start=1):
        print(f"[{index}] {opt}")

    # get user input 
    while True:
        choice = int(input("Enter choice number: "))
        if 1 <= choice <= len(options):     # if choice is valid
            return options[choice - 1]      # return choice-1 (0 indexing)
        else:
            print(f"Invalid input. Please enter a number between 1 and {len(options)}.")


def select_file():
    processed_dir = get_processed_directory()
    regions_dir = os.path.join(processed_dir, "regions")

# ------------------------------ GET LOCATION FROM USER -------------------------------#
    # select if national, regional, or province
    scope = userChoice(
        "Select Geographic Level to Predict", 
        ["National (Philippines)", "Regional", "Provincial"]
    )
    # if national, get the national directory from user
    if scope == "National (Philippines)":
        target_folder = os.path.join(processed_dir, "Philippines")
        location_label = "PHILIPPINES"

    # if region, ask the user for what region to pick
    elif scope == "Regional":
        regions_list = []
        # get all regions
        for d in os.listdir(regions_dir):
            if os.path.isdir(os.path.join(regions_dir, d)):
                regions_list.append(d)
        regions = sorted(regions_list)

        # ask user for their choice
        selected_region = userChoice("Select Region", regions)
        target_folder = os.path.join(regions_dir, selected_region)
        location_label = selected_region


    # For provincial
    else:
        regions_list = []
        # get all regions
        for d in os.listdir(regions_dir):
            if os.path.isdir(os.path.join(regions_dir, d)):
                regions_list.append(d)
        regions = sorted(regions_list)
        selected_region = userChoice("Select Region", regions)
        region_path = os.path.join(regions_dir, selected_region)

        # get all provinces in region selected
        province_list = []
        for k in os.listdir(region_path):
            if os.path.isdir(os.path.join(region_path, k)):
                province_list.append(k)

        provinces = sorted(province_list)
        selected_province = userChoice(f"Select Province in {selected_region}", provinces)
        target_folder = os.path.join(region_path, selected_province)
        location_label = f"{selected_province} ({selected_region})"

# ------ GET CROPTYPE FROM USER 
    crop_list = []
    for f in os.listdir(target_folder):
        if f.endswith(".csv"):
            crop_list.append(f[:-4])        # remove .csv
    crop_files = sorted(crop_list)

    # ask user what crop it wants to predict
    selected_crop = userChoice("Select Crop Type", crop_files)
    csv_path = os.path.join(target_folder, f"{selected_crop}.csv")

    periods = [
        "Annual",
        "Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4",
        "Semester 1", "Semester 2",
        "All Quarters (Continuous Time Series)"
    ]
    selected_period = userChoice("Select Time Period to Model", periods)

    return csv_path, location_label, selected_crop, selected_period

# ------------------------------ LOAD DATA -------------------------------#
def load_data(csv_path, selected_period):
    X = []
    Y = []

    quarter_offsets = {
        "Quarter 1": 0.0,
        "Quarter 2": 0.25,
        "Quarter 3": 0.50,
        "Quarter 4": 0.75
    }

    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)                             # first row as keys
        for row in reader:
            year_str = row["Year"].strip()
            period_str = row["Period"].strip()
            prod_str = row["Production_Amount"].strip()

            if prod_str == "":   # skip row without records
                continue

            year = int(year_str)
            production = float(prod_str)

            if selected_period == "All Quarters (Continuous Time Series)":

                # if want to see all quarters, gets all the quarters using their offsets
                if period_str in quarter_offsets:
                    X.append([year + quarter_offsets[period_str]])
                    Y.append(production)
            else:
                if period_str == selected_period:
                    X.append([float(year)])
                    Y.append(production)

    return X, Y



#--------------------------- LINEAR REG -------------------------------------------#

def linear_regression(X, Y):
    # compute for best weights
    # linear regression formula W^= (X^T * X)^-1 X^Ty (From lecture)

    # convert normal lists to numpy arrays
    X_array = np.array(X, dtype=float)
    Y_array = np.array(Y, dtype=float)

    X_rownum = X_array.shape[0]
    # add 1 before each X value to be able to get the intercept
    # create an np array with values of 1 with (X rows and 1 col)
    ones = np.ones((X_rownum, 1))
    # combine horizontally the 1 and the original X values (1, x1) (1,x2)...etc
    X_b = np.hstack((ones, X_array))

    # linear reg formula
    X_transposed = X_b.T
    X_transposed_dot_X = X_transposed.dot(X_b)      # (X^T * X)
    X_inverse_mat = np.linalg.inv(X_transposed_dot_X)   # (X^T * X)^-1

    inverse_dot_trans = X_inverse_mat.dot(X_transposed) # (X^T * X)^-1 (X^T)

    w_hat = inverse_dot_trans.dot(Y_array) # w^ = (X^T * X)^-1 * (X^T) * Y

    intercept = w_hat[0]
    slope = w_hat[1]

    return intercept, slope

    

