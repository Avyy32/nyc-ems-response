import pandas as pd

NUM = ["initial_severity_level_code", "hour", "dayofweek", "month", "is_weekend", "is_night",
       "is_rush_hour", "population", "median_hh_income", "median_age", "poverty_rate",
       "pct_black", "pct_hispanic", "pct_no_vehicle"]
BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

def clean_borough(s):
    return s.str.title().replace({"Richmond / Staten Island": "Staten Island"})

def build_X(df):
    X = df[NUM].astype(float).copy()
    b = clean_borough(df["borough"])
    for name in BOROUGHS:
        X["boro_" + name.replace(" ", "_")] = (b == name).astype(float)
    return X
