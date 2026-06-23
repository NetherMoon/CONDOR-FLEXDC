import pandas as pd
import numpy as np

df = pd.read_csv("all_data.csv")

print(df["workload_mix"][0])
print(df.head())