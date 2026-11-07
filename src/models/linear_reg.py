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


def forecast_and_plot(X, Y, intercept, slope, location, crop, period):
    # ask the user what to forcast
    while True:
        forecast_years = int(input("\nHow many years into the future do you want to forcast? (e.g., 5): "))
        if forecast_years > 0:
            break
        else:
            print("Please enter a positive integer")

    if period == "All Quarters (Continuous Time Series)":
        step = 0.25
        num_points = forecast_years * 4         # 4 quarters per year
    else:
        step = 1.0
        num_points = forecast_years

    # find the latest time point in data
    last_x = X[0][0]
    for i in X:
        current_year = i[0]
        if current_year > last_x:
            last_x = current_year

    future_X = []
    future_Y = []


    print(f"\n--- FORECAST RESULTS ({location} - {crop}) ---")
    for i in range(1, num_points + 1):
        next_x = last_x + (i*step)
        pred_y = intercept + (slope * next_x)

        if pred_y < 0:
            pred_y = 0.0

        future_X.append([next_x])
        future_Y.append(pred_y)

        if period == "All Quarters (Continuous Time Series)":
            year_part = int(next_x)
            quarter_decimal = next_x - year_part
            q_num = int((quarter_decimal * 4) + 1)
            print(f"Predicted {year_part} Q{q_num}: {pred_y:,.2f} Metric Tons")
        else:
            print(f"Predicted Year {int(next_x)}: {pred_y:,.2f} Metric Tons")

    # Visualization using matplotlib
    # make 1d array for x and future x
    x_flat = []
    for i in X:
        x_flat.append(i[0])

    last_historical_year = x_flat[-1]

    future_xflat = []
    for i in future_X:
        future_xflat.append(i[0])

    # calculate the straight line trend for linear regression
    hist_trend_Y = []
    for i in x_flat:
        predicted_val = intercept + (slope * i)
        hist_trend_Y.append(predicted_val)

    last_historical_prediction = hist_trend_Y[-1]

    future_xflat.insert(0, last_historical_year)
    future_Y.insert(0, last_historical_prediction)
    
    # create window
    plt.figure(figsize=(10,6))

    # plot historical data
    plt.scatter(x_flat, Y, color='blue', label='Historical Data', alpha=0.6)

    # plot the linear line
    plt.plot(x_flat, hist_trend_Y, color='black', linestyle='--', label='Historical Trend')

    #plot future forecast
    plt.scatter(future_xflat, future_Y, color='red', label='Future Forecast')
    plt.plot(future_xflat, future_Y, color='red', linestyle='-')

    # labels
    plt.title(f"{crop} Production Forecast for {location} ({period})")
    plt.xlabel("Time (Years)")
    plt.ylabel("Production Amount (Metric Tons)")

    # removes scientific notation for y axis
    plt.ticklabel_format(style='plain', axis='y')

    # formatting
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # from https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.format_coord.html
    ax = plt.gca()
    ax.format_coord = lambda x, y: f"x={x:.2f}, y={y:,.2f}"
    plt.show()

# ------ MAIN -------
if __name__ == "__main__":
    try:
        csv_path, location, crop, period, = select_file()

        if os.path.exists(csv_path):
            X, Y = load_data(csv_path, period)

            if len(X) < 2:
                print("Error, not enough data points to calculate linear regression")
            else:
                intercept, slope = linear_regression(X, Y)
                print(f"\nModel Trained! Slope: {slope:.2f}, Intercept: {intercept:.2f}")

                forecast_and_plot(X, Y, intercept, slope, location, crop, period)
        else:
            print(f"Status: FAILED. The file {csv_path} does not exist.")
    except Exception as e:
        print(f"\nAn error occured: {e}")
    
