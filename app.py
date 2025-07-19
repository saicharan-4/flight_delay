from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/flight_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        airline = int(request.form["airline"])
        origin = int(request.form["origin"])
        destination = int(request.form["destination"])
        dep_time = int(request.form["dep_time"])

        features = np.array([[airline, origin, destination, dep_time]])
        prediction = model.predict(features)[0]

        result = "Flight will be DELAYED" if prediction == 1 else "Flight will be ON TIME"
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)

