import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv(
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
)


def roast_category(value):
    if pd.isna(value):
        return -1

    roast_map = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }

    return roast_map.get(value, -1)

#exercise 1
features = ["100g_USD"]
df_model_1 = df[features + ["rating"]].dropna()
X = df_model_1[features]
y = df_model_1["rating"]
lm_0 = LinearRegression()
lm_0.fit(X, y)

with open("model_1.pickle", "wb") as f:
    pickle.dump(lm_0, f)

# Exercise 2
df["roast_cat"] = df["roast"].apply(roast_category)
features_dt = ["100g_USD", "roast_cat"]
df_model_2 = df[features_dt + ["rating"]].dropna()
X = df_model_2[features_dt]
y = df_model_2["rating"]
dt_0 = DecisionTreeRegressor()
dt_0.fit(X, y)

with open("model_2.pickle", "wb") as f:
    pickle.dump(dt_0, f)