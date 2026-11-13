import numpy as np
import matplotlib.pyplot as plt
import os
import csv

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


def build_features(data):
    #Turn the data into a table X (inputs) and y (tons). Each row = one place, crop and quarter
    places = sorted({key[0] for key in data})
    X, y, years = [], [], []
    for (place, crop, year, q), tons in sorted(data.items(), key=lambda kv: (kv[0][2], kv[0][3])):
        # last quarter, and the same quarter last year
        prev_year, prev_q = (year - 1, 4) if q == 1 else (year, q - 1)
        lag1 = data.get((place, crop, prev_year, prev_q))
        lag4 = data.get((place, crop, year - 1, q))
        if lag1 is None or lag4 is None:                      # skip if no history
            continue
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

        # last quarter's and last year's production, on a log scale
        row.append(np.log1p(lag1))
        row.append(np.log1p(lag4))
        X.append(row)
        y.append(tons)
        years.append(year)
    return np.array(X), np.array(y), np.array(years)


def mae(actual, pred):
    return np.mean(np.abs(actual - pred))

def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))

def r2(actual, pred):
    return 1.0 - np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual)) ** 2)


def run_svr(data, last_train_year=2018, epsilon=0.05, lam=1e-4, lr=0.01, epochs=1000):
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
        "mae": mae(actual, pred),
        "rmse": rmse(actual, pred),
        "r2": r2(actual, pred),
        "actual": actual,
        "predicted": pred,
    }


def run_national_svr(last_train_year=2018, epsilon=0.05, lam=1e-4, lr=0.01, epochs=1000):
    return run_svr(load_national(), last_train_year=last_train_year, epsilon=epsilon,
                   lam=lam, lr=lr, epochs=epochs)


def run_region_svr(last_train_year=2018, epsilon=0.05, lam=1e-4, lr=0.01, epochs=1000):
    return run_svr(load_regions(), last_train_year=last_train_year, epsilon=epsilon,
                   lam=lam, lr=lr, epochs=epochs)


def run_province_svr(last_train_year=2018, epsilon=0.05, lam=1e-4, lr=0.01, epochs=1000):
    return run_svr(load_provinces(), last_train_year=last_train_year, epsilon=epsilon,
                   lam=lam, lr=lr, epochs=epochs)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for name, fn in [("National", run_national_svr), ("Region", run_region_svr), ("Province", run_province_svr)]:
        r = fn()
        print(f"{name:9s} train={r['n_train']:6d} test={r['n_test']:5d} | "
              f"SVR   MAE={r['mae']:>9,.0f} RMSE={r['rmse']:>9,.0f} R2={r['r2']:.4f} | ")

        # actual vs predicted (points near the red line = good predictions)
        plt.figure(figsize=(6, 6))
        plt.scatter(r['actual'], r['predicted'], s=10)
        top = max(r['actual'].max(), r['predicted'].max())
        plt.plot([0, top], [0, top], "r--", label="perfect prediction")
        plt.xlabel("Actual production (tons)")
        plt.ylabel("Predicted production (tons)")
        plt.title(name + " SVR: actual vs predicted")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, name + "_actual_vs_predicted.png"))
        plt.close()

        # training loss (should go down and flatten out)
        plt.plot(r['model']['loss_history'])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(name + " SVR: training loss")
        plt.savefig(os.path.join(RESULTS_DIR, name + "_loss.png"))
        plt.close()