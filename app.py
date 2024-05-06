from flask import Flask, request, jsonify
import joblib
import numpy as np
from preproces import preproces

# load the model from disk
model = joblib.load('final_model.pkl')

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
        h_region = house_details['region']
        h_city = house_details['city']

        num_rooms = h_bathrooms+h_bedrooms

        # if h_city == "Cairo":
        #     h_city=1

        h_type, h_fur, h_region, h_city = preproces(h_type, h_fur, h_region, h_city)
        output = np.array([h_type, h_size, h_bedrooms, h_bathrooms, h_floor, h_fur, h_region, h_city, num_rooms])
        price = model.predict(output.reshape(1, -1))
        return jsonify(price[0])

    except Exception as e:
        # Handle exceptions and return an error message in JSON format
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)
