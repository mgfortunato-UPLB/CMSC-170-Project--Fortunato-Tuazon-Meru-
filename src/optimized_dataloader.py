import csv
import os
import re

def clean_name(raw_geo):
    # remove leading dots, whitespace, and footnotes from the raw_geo string
    name = raw_geo.lstrip(".").strip()
    name = re.sub(r"\s*[a-z]/$", "", name)           # remove footnote like ' a/' or 'd/'
    return name.strip()

def clean_and_save_data(raw_path=None, clean_path=None):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if raw_path is None:
        raw_path = os.path.join(project_root,"data","raw","raw_data.csv")
    if clean_path is None:
        clean_path = None

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

            for col_name, val_str in zip(time_headers, row[2:]):
                val_str = val_str.strip()

                parts = col_name.split(" ", 1)
                year = int(parts[0])
                period = parts[1]
                if val_str == ".." or val_str == "": 
                    production = ""         #Lorenz: changed to blank because 0 says that it produces nothing while in fact they have no records
                else:
                    production = float(val_str)
                cleaned_rows.append([year, period, production])

            #####segregating block of code###################################################################
            # imagine cleanpath as a pointer for files
            if raw_geo.startswith("...."):
                province =  clean_name(raw_geo)                      # was raw_geo.lstrip("."), changed to remove /a, /c, etc.
                clean_path = os.path.join(project_root,"data","processed","regions",region,province,croptype+".csv")

            elif raw_geo.startswith(".."):
                region = clean_name(raw_geo)                        # was raw_geo.lstrip(".").strip("d/")
                clean_path = os.path.join(project_root,"data","processed","regions",region,croptype+".csv")
                
            else:
                clean_path = os.path.join(project_root,"data","processed","Philippines",croptype+".csv")
            ##################################################################################################

            #####appender/writer block of code################################################################
            if os.path.exists(clean_path):
                with open(clean_path, mode="a", newline="", encoding="utf-8") as cleanfile:
                    writer = csv.writer(cleanfile)
                    writer.writerows(cleaned_rows)
            else:
                os.makedirs(os.path.dirname(clean_path), exist_ok=True)
                with open(clean_path, mode="w", newline="", encoding="utf-8") as cleanfile:
                    writer = csv.writer(cleanfile)
                    writer.writerow(["Year", "Period", "Production_Amount"])
                    writer.writerows(cleaned_rows)
            ##################################################################################################

            cleaned_rows = [] #back to zero, next row from raw_data.csv

    print("Test test")
    return cleaned_rows

# ------------------------------

#optimized_dataloader.clean_and_save_data() to call in main
if __name__ == "__main__":
    clean_and_save_data()