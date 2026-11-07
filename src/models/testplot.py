import csv
import matplotlib.pyplot as plt

with open("processed/Philippines/Corn.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader) # skip header

    x_values = []
    y_values = []

    for row in reader:
        if row[1].startswith("Quarter"):
            x_values.append(float(row[0]))
            y_values.append(float(row[2]))

plt.plot(x_values, y_values, "r.")
plt.show()