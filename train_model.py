import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

data = {
    "sqft": [1000, 1500, 2000, 2500, 3000],
    "bhk": [2, 3, 3, 4, 4],
    "price": [50, 75, 100, 130, 160]
}

df = pd.DataFrame(data)

X = df[["sqft", "bhk"]]
y = df["price"]

model = RandomForestRegressor()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")
