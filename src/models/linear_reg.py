# Allowed: 
# - NumPy, but only for arrays and basic linear algebra
# - Matplotlib, for plots and figures

import os
import csv

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

# ------------------------------ GET CROPTYPE FROM USER -------------------------------#
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

