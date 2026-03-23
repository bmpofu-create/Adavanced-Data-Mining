from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('xgb_tuned.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check source of data
        if request.is_json:
            data = request.get_json()
            # Define feature order expected by model
            feature_names = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'hour', 'minute', 'month', 'day_of_year']
            features = [data[name] for name in feature_names]
            final_features = [np.array(features)]
            
            prediction = model.predict(final_features)
            output = prediction[0] # output is scalar from xgb predict
            
            return jsonify({'prediction': f"{output:.2f} kW"})
        else:
            # Get data from form (legacy)
            features = [float(x) for x in request.form.values()]
            final_features = [np.array(features)]
            
            prediction = model.predict(final_features)
            output = prediction[0] 
            
            return render_template('index.html', prediction_text=f'Predicted AC Power: {output:.2f} kW')

    except Exception as e:
        if request.is_json:
             return jsonify({'error': str(e)}), 400
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

