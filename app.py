from flask import Flask, request, render_template
import joblib
import numpy as np

# 🔸 Create the Flask app first
app = Flask(__name__)

# 🔸 Load the ML model
model = joblib.load("best_model.pkl")

# 🔸 Home route
@app.route('/')
def home():
    return render_template("form.html")

# 🔸 Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [
            int(request.form['age']),
            int(request.form['workclass']),
            int(request.form['fnlwgt']),
            int(request.form['education_num']),
            int(request.form['marital_status']),
            int(request.form['occupation']),
            int(request.form['relationship']),
            int(request.form['race']),
            int(request.form['sex']),
            int(request.form['capital_gain']),
            int(request.form['capital_loss']),
            int(request.form['hours_per_week']),
            int(request.form['native_country'])
        ]
        pred = model.predict([np.array(input_data)])
        output = ">50K" if pred[0] == 1 else "<=50K"
        return render_template("result.html", prediction=output)
    except:
        return "Invalid input! Please go back and check the values."

# 🔸 Run the app
if __name__ == '__main__':
    app.run(debug=True)
