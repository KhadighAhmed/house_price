from flask import Flask, request, jsonify
import joblib
import numpy as np

# load the model from disk
model = joblib.load('model3.pkl')

app = Flask(__name__)


@app.route('/house_pred', methods=['POST'])
def prediction():
    try:
        house_details = request.get_json()
        h_type = house_details['type']
        h_size = house_details['size']
        h_bedrooms = house_details['bedrooms']
        h_bathrooms = house_details['bathrooms']
        h_floor = house_details['floor']
        h_fur = house_details['fur']
        h_rent = house_details['rent']
        h_region = house_details['region']
        h_city = house_details['city']

        price = model.predict(
            np.array([h_type, h_size, h_bedrooms, h_bathrooms, h_floor, h_fur, h_rent, h_region, h_city]))
        return jsonify(price)

    except Exception as e:
        # Handle exceptions and return an error message in JSON format
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)
