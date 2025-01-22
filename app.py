from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

with open("amenities.pkl", "rb") as f:
    new_amenities = pickle.load(f)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

final_amenities = []
for amenity in new_amenities:
    if amenity and amenity not in final_amenities:
        final_amenities.append(amenity)

app = Flask(__name__)

city_frequency_map = {
    'Kathmandu': 1483,
    'Lalitpur': 432,
    'Bhaktapur': 85,
    'Pokhara': 64,
    'Chitwan': 33,
    'Makwanpur': 12,
    'Nawalparasi': 12,
    'Dharan': 11,
    'Jhapa': 10,
    'Kavre': 7,
    'Kirtipur': 7,
    'Sunsari': 7,
    'Butwal': 6,
    'Biratnagar': 6,
    'Parsa': 4,
    'Bara': 4,
    'Dhading': 4,
    'Morang': 4,
    'Itahari': 3,
    'Kaski': 3,
    'Rupandehi': 3,
    'Birtamod': 1,
    'Nawalpur': 1,
    'Mahottari': 1,
    'Dang': 1,
    'Bardiya': 1,
    'Bhairahawa': 1,
    'Surkhet': 1,
    'Kapilvastu': 1,
    'Tanahu': 1,
    'Illam': 1,
    'Kailali': 1
}

amenities_frequency_map = {
    'Water Supply': 877, 
    'Drainage': 800, 
    'Balcony': 777, 
    'Water Tank': 726, 
    'TV Cable': 621, 
    'Wifi': 476, 
    'Solar Water': 406, 
    'Parking': 398, 
    'Modular Kitchen': 397, 
    'Garden': 393, 
    'Water Well': 352, 
    'Garage': 339, 
    'Frontyard': 274, 
    'Fencing': 263, 
    'Internet': 220, 
    'Backyard': 210,
    'Lawn': 189, 
    'Electricity Backup': 157, 
    'Store Room': 148, 
    'Washing Machine': 126, 
    'Air Condition': 119, 
    'Microwave': 111, 
    'Deck': 101, 
    'Kids Playground': 90, 
    'CCTV': 59, 
    'Gym': 51, 
    'Swimming Pool': 46, 
    'Intercom': 41, 
    'Security Staff': 37, 
    'Cafeteria': 26, 
    'Lift': 20, 
    'Maintenance': 20, 
    'Jacuzzi': 10
}

cities = list(city_frequency_map.keys())

@app.route("/")
def home():
    print("Home route accessed")
    return render_template("index.html", cities = cities, amenities = final_amenities)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Prediction route accessed")
        city = request.form.get("City")
        bedroom = int(request.form.get("Bedroom"))
        bathroom = int(request.form.get("Bathroom"))
        year = int(request.form.get("Year"))
        new_area = float(request.form.get("NewArea"))
        amenities = request.form.get("Amenities").split(",") 


        city_encoded = city_frequency_map.get(city, 0)

        amenities_score = sum(amenities_frequency_map.get(amenity.strip(), 0) for amenity in amenities)

        features = np.array([city_encoded, bedroom, bathroom, year, new_area, amenities_score]).reshape(1, -1)

        prediction = model.predict(features)

        return render_template("index.html", prediction = int(prediction[0]), cities = cities, amenities = final_amenities)

    except Exception as e:
        return render_template("index.html", error=f"Error occurred: {str(e)}")
    
if __name__ == "__main__":
    print("Running app...")
    app.run(debug=True, port=5001)
