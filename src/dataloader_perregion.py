# 1. load data if clean_data.csv exists
# 2. if not, run cleaning, save to clean data.csv, return dataframe

import csv
import os
import re

def clean_and_save_data(raw_path=None, clean_path=None):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if raw_path is None:
        raw_path = os.path.join(project_root,"data","raw","raw_data.csv")
    if clean_path is None:
        clean_path = os.path.join(project_root,"data","processed","national.csv")


    cleaned_rows = []

    with open(raw_path, mode="r", encoding="utf-8") as rawfile:     # open raw_data.csv
        reader = csv.reader(rawfile)
        next(reader)                    # skip title row

        headers = next(reader)

        # from 1987 Q1 to 2026 annual
        time_headers = headers[2:]

        for row in reader:
            if not row:
                continue
            croptype = row[0].strip()           # first col is always croptype
            raw_geo = row[1]                    # 2nd is geolocation

            if raw_geo.startswith("...."):
                geo_level = "Province"
            elif raw_geo.startswith(".."):
                geo_level = "Region"
                region = raw_geo.lstrip(".")
                os.makedirs(os.path.dirname(clean_path), exist_ok=True)
                with open(clean_path, mode="w", newline="", encoding="utf-8") as cleanfile:
                    writer = csv.writer(cleanfile)
                    writer.writerow(["Year", "Period", "Geo_level", "Geolocation", "Croptype", "Production_Amount"])
                    writer.writerows(cleaned_rows)
                cleaned_rows = []
                clean_path = os.path.join(project_root,"data","processed","regions",region.strip("/")+".csv")
            else:
                geo_level = "National"

            # lstrip removes leading dots
            clean_geo = raw_geo.lstrip(".").strip()

            for col_name, val_str in zip(time_headers, row[2:]):
                val_str = val_str.strip()
                if val_str == ".." or val_str == "":    # for missing inputs
                    continue

                parts = col_name.split(" ", 1)
                year = int(parts[0])
                period = parts[1]
                production = float(val_str)

                cleaned_rows.append([year, period, geo_level, clean_geo, croptype, production])

    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    with open(clean_path, mode="w", newline="", encoding="utf-8") as cleanfile:
        writer = csv.writer(cleanfile)
        writer.writerow(["Year", "Period", "Geo_level", "Geolocation", "Croptype", "Production_Amount"])
        writer.writerows(cleaned_rows)

    print("Test test")
    return cleaned_rows



# ------------------------------

clean_and_save_data()