from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Example accuracy (replace with your actual R2 score if printed earlier)
MODEL_ACCURACY = 91.4

@app.route("/")
def home():
    return render_template("index.html", prediction_text="", accuracy=MODEL_ACCURACY)

@app.route("/predict", methods=["POST"])
def predict():
    sqft = float(request.form["sqft"])
    bhk = float(request.form["bhk"])

    features = np.array([[sqft, bhk]])
    prediction = model.predict(features)[0]

    price_lakhs = round(prediction, 2)


    # Segment classification
    if price_lakhs > 80:
        segment = "Premium Property 🏢"
    elif price_lakhs > 50:
        segment = "Mid Range Property 🏠"
    else:
        segment = "Budget Friendly Property 🏡"

    return render_template("index.html",
                           prediction_text=f"₹ {price_lakhs} Lakhs",
                           segment=segment,
                           accuracy=MODEL_ACCURACY)

if __name__ == "__main__":
    app.run(debug=True)
