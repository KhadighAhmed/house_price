from flask import Flask, request, jsonify
import joblib
import numpy as np
from preproces import preproces

# load the model from disk
model = joblib.load('new_model.pkl')

app = Flask(__name__)


@app.route('/house_pred', methods=['POST'])
def prediction():
    try:
        data = request.get_json()
        # Extract and preprocess various user attributes
        age = data['age']
        gender = data['gender']
        education = data['education']
        employed = data['employed']
        income = data['income']
        gap_in_resume = data['gap_in_resume']
        pc = data['pc']
        internet_access = data['internet_access']
        live_with_parents = data['live_with_parents']
        read_out = data['read_out']
        disabled = data['disabled']
        mental_illness_b4 = data['mental_illness_b4']
        times_hosp = data['times_hosp']
        days_hosp = data['days_hosp']
        anxiety1 = data['anxiety1']
        anxiety2 = data['anxiety2']
        anxiety3 = data['anxiety3']
        depression1 = data['depression1']
        depression2 = data['depression2']
        lack_of_concentration1 = data['lack_of_concentration1']
        lack_of_concentration2 = data['lack_of_concentration2']
        obsessive_thinking1 = data['obsessive_thinking1']
        obsessive_thinking2 = data['obsessive_thinking2']
        mood_swing1 = data['mood_swing1']
        mood_swing2 = data['mood_swing2']
        panic_attacks = data['panic_attacks']
        compulsive_behavior1 = data['compulsive_behavior1']
        compulsive_behavior2 = data['compulsive_behavior2']
        tiredness1 = data['tiredness1']
        tiredness2 = data['tiredness2']

        # house_details = request.get_json()
        # h_type = house_details['type']
        # h_size = house_details['size']
        # h_bedrooms = house_details['bedrooms']
        # h_bathrooms = house_details['bathrooms']
        # h_floor = house_details['floor']
        # h_fur = house_details['fur']
        # h_region = house_details['region']
        # h_city = house_details['city']
        #
        # num_rooms = h_bathrooms+h_bedrooms

        # if h_city == "Cairo":
        #     h_city=1

        # h_type, h_fur, h_region, h_city = preproces(h_type, h_fur, h_region, h_city)
        # output = np.array([h_type, h_size, h_bedrooms, h_bathrooms, h_floor, h_fur, h_region, h_city, num_rooms])
        # price = model.predict(output.reshape(1, -1))
        result = np.array([employed, education, pc, mental_illness_b4, days_hosp, disabled, internet_access,
                           live_with_parents, gap_in_resume, income, read_out, times_hosp,
                           lack_of_concentration1, anxiety1, depression1, obsessive_thinking1, mood_swing1, panic_attacks,
                           compulsive_behavior1, tiredness1, age, gender]).reshape(1, -1)
        price = model.predict(result)
        return jsonify(int(price[0]))

    except Exception as e:
        # Handle exceptions and return an error message in JSON format
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)
