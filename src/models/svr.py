import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv

# svr.py lives in src/models/, add src/ to sys.path so we can import eval_metrics (same as linear_reg.py)
models_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(models_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from eval_metrics import mean_absolute_error

# National / region / province SVR  (reads the files made by optimized_dataloader.py)


# svr.py lives in src/models/, so the project root is 3 folders up from this file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# where the graph will be saved
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Palay and Corn are totals of these four, so using them too would count the same crops twice
CROPS = ["Irrigated Palay", "Rainfed Palay", "White Corn", "Yellow Corn"]
FIRST_YEAR = 1987

def train_linear_svr(X, y, epsilon=0.05, lam=1e-4, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    
    # Initialize weights and bias to zeros
    w = np.zeros(n_features)
    b = 0.0
    loss_history = []
    
    for t in range(epochs):
        # Predict current values
        predictions = np.matmul(X, w) + b
        
        # Calculate error (Actual - Predicted)
        errors = y - predictions
        
        # Find points outside the epsilon-tube (|error| > epsilon)
        outside_mask = np.abs(errors) > epsilon
        
        # Get direction (+1 or -1) for points outside, 0 for points inside
        error_signs = np.sign(errors) * outside_mask
        
        # Calculate slopes (gradients)
        grad_w = (lam * w) - (np.matmul(X.T, error_signs) / n_samples)
        grad_b = -np.mean(error_signs)
        
        # Update weights and bias
        w -= lr * grad_w
        b -= lr * grad_b

        # Track loss (Regularization Loss + Epsilon-Insensitive Loss)
        margin_loss = np.maximum(0, np.abs(errors) - epsilon).mean()
        reg_loss = 0.5 * lam * np.sum(w ** 2)
        loss_history.append(reg_loss + margin_loss)
        
    return {'weights': w, 'bias': b, 'loss_history': loss_history}

def predict_linear_svr(model, X):
    return np.matmul(X, model['weights']) + model['bias']


def read_file(path, place, crop, data):
    #Read one csv (Year, Period, Production_Amount) into data[(place, crop, year, quarter)] = tons.
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)                                         # skip header
        for row in reader:
            if row and row[1].startswith("Quarter") and row[2] != "":     # blank = missing, skip
                key = (place, crop, int(row[0]), int(row[1].split()[1]))
                value = float(row[2])
                # a few provinces appear in two region folders; keep the larger value (the other is a 0 placeholder)
                data[key] = max(value, data.get(key, value))


def load_national():
    data = {}
    for crop in CROPS:
        read_file(os.path.join(PROCESSED_DIR, "Philippines", crop + ".csv"), "Philippines", crop, data)
    return data


def load_regions():
    data = {}
    regions_dir = os.path.join(PROCESSED_DIR, "regions")
    for region in sorted(os.listdir(regions_dir)):
        for crop in CROPS:
            path = os.path.join(regions_dir, region, crop + ".csv")
            if os.path.exists(path):
                read_file(path, region, crop, data)
    return data


def load_provinces():
    data = {}
    regions_dir = os.path.join(PROCESSED_DIR, "regions")
    for region in sorted(os.listdir(regions_dir)):
        region_dir = os.path.join(regions_dir, region)
        for province in sorted(os.listdir(region_dir)):
            province_dir = os.path.join(region_dir, province)
            if not os.path.isdir(province_dir):              # skip the region's own crop files
                continue
            for crop in CROPS:
                path = os.path.join(province_dir, crop + ".csv")
                if os.path.exists(path):
                    read_file(path, province, crop, data)
    return data


def make_row(places, place, crop, year, q):
    #One row of features: place, crop, year and quarter only (no lag values, so nothing feeds back on itself)
    row = []

    # which place: one slot per place, 1.0 for the matching one and 0.0 for the rest
    for p in places:
        if place == p:
            row.append(1.0)
        else:
            row.append(0.0)

    # which crop: same idea, one slot per crop
    for c in CROPS:
        if crop == c:
            row.append(1.0)
        else:
            row.append(0.0)

    # the year as a plain number
    row.append(float(year))

    # which quarter: one slot each for Q1, Q2, Q3, Q4
    for k in (1, 2, 3, 4):
        if q == k:
            row.append(1.0)
        else:
            row.append(0.0)
    return row


def build_features(data):
    #Turn the data into a table X (inputs) and y (tons). Each row = one place, crop and quarter
    places = sorted({key[0] for key in data})
    X, y, years = [], [], []
    for (place, crop, year, q), tons in sorted(data.items(), key=lambda kv: (kv[0][2], kv[0][3])):
        X.append(make_row(places, place, crop, year, q))
        y.append(tons)
        years.append(year)
    return np.array(X), np.array(y), np.array(years)


def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))

def r2(actual, pred):
    return 1.0 - np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2)


def run_svr(data, last_train_year=2014, epsilon=0.05, lam=1e-4, lr=0.01, epochs=1000):
    #Train on years <= last_train_year, test on later years. Works for any level (national/region/province).
    X, y, years = build_features(data)
    train = years <= last_train_year
    test = ~train

    # work on a log scale and standardize, using the TRAINING rows only
    y_log = np.log1p(y)
    y_mean, y_std = y_log[train].mean(), y_log[train].std()
    X_mean, X_std = X[train].mean(axis=0), X[train].std(axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std
    y_scaled = (y_log - y_mean) / y_std

    model = train_linear_svr(X_scaled[train], y_scaled[train], epsilon, lam, lr, epochs)

    # predict, then go back to metric tons
    pred = np.expm1(predict_linear_svr(model, X_scaled[test]) * y_std + y_mean)
    pred = np.maximum(pred, 0)
    actual = y[test]

    return {
        "model": model,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "mae": mean_absolute_error(actual, pred),
        "rmse": rmse(actual, pred),
        "r2": r2(actual, pred),
        "actual": actual,
        "predicted": pred,
        # kept so we can predict new rows (fitted line + future forecast) later
        "places": sorted({key[0] for key in data}),
        "last_train_year": last_train_year,
        "X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std,
    }


def run_national_svr(last_train_year=2014, epsilon=0.05, lam=1e-4, lr=0.01, epochs=1000):
    return run_svr(load_national(), last_train_year=last_train_year, epsilon=epsilon,
                   lam=lam, lr=lr, epochs=epochs)


def run_region_svr(last_train_year=2014, epsilon=0.05, lam=1e-4, lr=0.01, epochs=1000):
    return run_svr(load_regions(), last_train_year=last_train_year, epsilon=epsilon,
                   lam=lam, lr=lr, epochs=epochs)


def run_province_svr(last_train_year=2014, epsilon=0.05, lam=1e-4, lr=0.01, epochs=1000):
    return run_svr(load_provinces(), last_train_year=last_train_year, epsilon=epsilon,
                   lam=lam, lr=lr, epochs=epochs)


ALL_QUARTERS = "All Quarters (Continuous Time Series)"
PERIODS = ["Annual", "Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4",
           "Semester 1", "Semester 2", ALL_QUARTERS]

# which quarters are added up for each period
PERIOD_QUARTERS = {
    "Annual": [1, 2, 3, 4],
    "Quarter 1": [1], "Quarter 2": [2], "Quarter 3": [3], "Quarter 4": [4],
    "Semester 1": [1, 2], "Semester 2": [3, 4],
}
QUARTER_OFFSETS = {1: 0.0, 2: 0.25, 3: 0.50, 4: 0.75}

# Palay and Corn are not trained on directly, they are the sum of their parts
COMPOSITE_CROPS = {
    "Palay": ["Irrigated Palay", "Rainfed Palay"],
    "Corn": ["White Corn", "Yellow Corn"],
}


def userChoice(printSentence, options):
    print(f"\n--- {printSentence} ---")
    for index, opt in enumerate(options, start=1):
        print(f"[{index}] {opt}")
    while True:
        try:
            choice = int(input("Enter choice number: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        if 1 <= choice <= len(options):
            return options[choice - 1]
        print(f"Invalid input. Please enter a number between 1 and {len(options)}.")


def predict_tons(res, rows):
    #Predict metric tons for feature rows, using the scaling from training
    X = (np.array(rows) - res["X_mean"]) / res["X_std"]
    scaled = predict_linear_svr(res["model"], X)
    return np.maximum(np.expm1(scaled * res["y_std"] + res["y_mean"]), 0)


def get_series(data, place, crop):
    #{(year, quarter): tons} for one place and one trained crop
    series = {}
    for (p, c, year, q), tons in data.items():
        if p == place and c == crop:
            series[(year, q)] = tons
    return series


def add_up(parts):
    #add several {(year, q): tons} together, only for quarters every part has
    common = set(parts[0])
    for part in parts[1:]:
        common &= set(part)
    return {k: sum(part[k] for part in parts) for k in common}


def fitted_quarters(res, series, place, crop):
    #What the model predicts for every past quarter
    keys = sorted(series)
    if not keys:
        return {}
    rows = [make_row(res["places"], place, crop, year, q) for (year, q) in keys]
    return dict(zip(keys, predict_tons(res, rows)))


def forecast_quarters(res, place, crop, start, end):
    #Predict every quarter after `start` up to `end`. Each quarter is predicted straight from
    #place, crop, year and quarter, so one prediction never depends on another.
    keys = []
    year, q = start
    while (year, q) < end:
        q += 1
        if q == 5:
            year, q = year + 1, 1
        keys.append((year, q))
    if not keys:
        return {}
    rows = [make_row(res["places"], place, crop, y_, q_) for (y_, q_) in keys]
    return dict(zip(keys, predict_tons(res, rows)))


def to_points(series, period):
    #Turn {(year, q): tons} into {x: tons} for the chosen period (x is the year, or year + 0.25 steps)
    if period == ALL_QUARTERS:
        return {year + QUARTER_OFFSETS[q]: v for (year, q), v in series.items()}
    wanted = PERIOD_QUARTERS[period]
    by_year = {}
    for (year, q), v in series.items():
        if q in wanted:
            by_year.setdefault(year, {})[q] = v
    # only years that have every quarter of the period
    return {float(y): sum(d.values()) for y, d in by_year.items() if len(d) == len(wanted)}


def split_year(data, place, crop, period):
    #Same rule as linear_reg.py: first 70% of the points train, last 30% test.
    #Returns the first test year, so the SVR can train on years before it.
    parts = COMPOSITE_CROPS.get(crop, [crop])
    hist = to_points(add_up([get_series(data, place, c) for c in parts]), period)
    xs = sorted(hist)
    split_index = int(len(xs) * 0.7)
    return int(xs[split_index])


def unit_label(period):
    if period == ALL_QUARTERS:
        return "Quarter"
    if period.startswith("Semester"):
        return "Semester"
    return "Year"


def x_label(x, period):
    if period == ALL_QUARTERS:
        year = int(x)
        return f"{year} Q{int(round((x - year) * 4)) + 1}"
    return f"Year {int(x)}"


def evaluate_svr(res, data, location, place, crop, period):
    #Same printout as evaluate_model() in linear_reg.py (MAE from eval_metrics.py)
    parts = COMPOSITE_CROPS.get(crop, [crop])
    actual = add_up([get_series(data, place, c) for c in parts])
    fitted = add_up([fitted_quarters(res, get_series(data, place, c), place, c) for c in parts])
    hist = to_points(actual, period)
    fit = to_points(fitted, period)

    test_start = res["last_train_year"] + 1
    unit = unit_label(period)
    shared = sorted(x for x in hist if x in fit)
    full_mae = mean_absolute_error([hist[x] for x in shared], [fit[x] for x in shared])
    test_x = [x for x in shared if x >= test_start]

    print(f"\n--- EVALUATION METRICS (SVR): {location} - {crop} - ({period}) ---")
    print(f"Full Dataset MAE (In-Sample): \n\tThe model is off by {full_mae:,.2f} Metric Tons of {crop} per {unit} in {location} (n={len(shared)})")
    if test_x:
        test_mae = mean_absolute_error([hist[x] for x in test_x], [fit[x] for x in test_x])
        print(f"Test Split MAE ({int(test_x[0])} - {int(test_x[-1])}), 30% unseen data: \n\tThe model is off by {test_mae:,.2f} Metric Tons of {crop} per {unit} in {location} (n = {len(test_x)}) (70% train / 30% test)")


def forecast_and_plot_svr(res, data, location, place, crop, period, forecast_years):
    parts = COMPOSITE_CROPS.get(crop, [crop])

    # real data, past predictions of the model
    actual_parts = [get_series(data, place, c) for c in parts]
    actual = add_up(actual_parts)
    fitted = add_up([fitted_quarters(res, get_series(data, place, c), place, c) for c in parts])

    hist = to_points(actual, period)           # blue dots
    fit = to_points(fitted, period)            # dashed line
    if len(hist) < 2:
        print("Error, not enough data points for this period")
        return

    # --- how far to forecast
    start = max(actual)                        # last (year, quarter) with real data
    last_x = max(hist)
    if period == ALL_QUARTERS:
        end = (start[0] + forecast_years, start[1])
    else:
        end = (int(last_x) + forecast_years, 4)

    future_quarters = add_up([
        forecast_quarters(res, place, c, start, end) for c in parts
    ])
    full = dict(actual)
    full.update(future_quarters)               # real quarters + predicted quarters
    full_points = to_points(full, period)
    future = sorted((x, v) for x, v in full_points.items() if x > last_x)

    test_start = res["last_train_year"] + 1

    print(f"\n--- FORECAST RESULTS ({location} - {crop}) ---")
    for x, v in future:
        print(f"Predicted {x_label(x, period)}: {v:,.2f} Metric Tons")

    # --- plot (same elements as linear_reg.py)
    x_hist = sorted(hist)
    x_fit = sorted(fit)
    fut_x = [x for x, v in future]
    fut_y = [v for x, v in future]
    # start the red line from the model's last historical value so the lines connect
    fut_x.insert(0, last_x)
    fut_y.insert(0, fit.get(last_x, hist[last_x]))

    plt.figure(figsize=(10, 6))
    plt.scatter(x_hist, [hist[x] for x in x_hist], color="blue", label="Historical Data", alpha=0.6)
    plt.plot(x_fit, [fit[x] for x in x_fit], color="black", linestyle="--", label="Historical Trend (SVR fit)")
    plt.scatter(fut_x, fut_y, color="red", label="Future Forecast")
    plt.plot(fut_x, fut_y, color="red", linestyle="-")
    plt.axvline(test_start, color="gray", linestyle=":", label=f"Test split ({test_start}+)")

    plt.title(f"{crop} Production Forecast for {location} ({period}) - SVR")
    plt.xlabel("Time (Years)")
    plt.ylabel("Production Amount (Metric Tons)")
    plt.ticklabel_format(style="plain", axis="y")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # same hover text as the linear regression plot
    ax = plt.gca()
    ax.format_coord = lambda x, y: f"x={x:.2f}, y={y:,.2f}"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in f"SVR_{location}_{crop}_{period}")
    plt.savefig(os.path.join(RESULTS_DIR, safe + ".png"))
    plt.show()


def select_svr_options():
    regions_dir = os.path.join(PROCESSED_DIR, "regions")

    scope = userChoice("Select Geographic Level to Predict",
                       ["National (Philippines)", "Regional", "Provincial"])

    if scope == "National (Philippines)":
        data = load_national()
        place, location = "Philippines", "PHILIPPINES"

    else:
        regions = sorted(d for d in os.listdir(regions_dir) if os.path.isdir(os.path.join(regions_dir, d)))
        region = userChoice("Select Region", regions)
        if scope == "Regional":
            data = load_regions()
            place, location = region, region
        else:
            region_path = os.path.join(regions_dir, region)
            provinces = sorted(d for d in os.listdir(region_path) if os.path.isdir(os.path.join(region_path, d)))
            place = userChoice(f"Select Province in {region}", provinces)
            data = load_provinces()
            location = f"{place} ({region})"

    # only offer crops that have data for this place
    crops = []
    for c in ["Palay", "Corn"] + CROPS:
        if add_up([get_series(data, place, p) for p in COMPOSITE_CROPS.get(c, [c])]):
            crops.append(c)
    crop = userChoice("Select Crop Type", sorted(crops))
    period = userChoice("Select Time Period to Model", PERIODS)

    return data, place, location, crop, period, scope


# ------ MAIN -------
if __name__ == "__main__":
    try:
        data, place, location, crop, period, scope = select_svr_options()

        print("\nTraining SVR, please wait...")
        last_train_year = split_year(data, place, crop, period) - 1     # 70% train / 30% test, same as linear_reg.py
        res = run_svr(data, last_train_year=last_train_year)
        print(f"Model Trained on years up to {last_train_year}! train rows={res['n_train']}, test rows={res['n_test']}, "
              f"final training loss={res['model']['loss_history'][-1]:.4f}")

        # training loss (should go down and flatten out), saved for the report
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.plot(res["model"]["loss_history"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(scope + " SVR: training loss")
        plt.savefig(os.path.join(RESULTS_DIR, "SVR_" + scope.split()[0] + "_loss.png"))
        plt.close()

        eval_choice = userChoice(
            "Do you want to compute the Evaluation Metric (MAE) for this model? ",
            ["Yes, show MAE score", "No, proceed directly to forecasting"]
        )

        if eval_choice == "Yes, show MAE score":
            evaluate_svr(res, data, location, place, crop, period)

        while True:
            try:
                forecast_years = int(input("\nHow many years into the future do you want to forecast? (e.g., 5): "))
            except ValueError:
                print("Please enter a positive integer")
                continue
            if forecast_years > 0:
                break
            print("Please enter a positive integer")

        forecast_and_plot_svr(res, data, location, place, crop, period, forecast_years)
    except Exception as e:
        print(f"\nAn error occured: {e}")