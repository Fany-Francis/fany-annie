import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    battery = float(request.form['battery'])
    bluetooth = 1 if request.form['blue'] == 'Yes' else 0
    clock_speed = float(request.form['clock'])
    dual_sim = 1 if request.form['sim'] == 'Yes' else 0
    front_camera = float(request.form['frontcamera'])
    is_4g = 1 if request.form['4G'] == 'Yes' else 0
    internal_memory = int(request.form['memory'])
    mobile_depth = float(request.form['depth'])
    phone_weight = float(request.form['weight'])
    processor_cores = int(request.form['cores'])
    primary_camera = float(request.form['camera'])
    pixel_resolution_height = int(request.form['height'])
    pixel_resolution_width = int(request.form['width'])
    ram = int(request.form['ram'])
    screen_height = int(request.form['srheight'])
    screen_width = int(request.form['srwidth'])
    talk_time = float(request.form['time'])
    is_3g = 1 if request.form['3G'] == 'Yes' else 0
    touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
    wifi = 1 if request.form['wifi'] == 'Yes' else 0

    # Create an array from the input values
    arr = np.array([
        battery, bluetooth, clock_speed, dual_sim, front_camera, is_4g, internal_memory, mobile_depth, phone_weight,
        processor_cores, primary_camera, pixel_resolution_height, pixel_resolution_width, ram, screen_height,
        screen_width, talk_time, is_3g, touchscreen, wifi
    ])

    # Reshape the array for prediction
    prediction = model.predict(arr.reshape(1, -1))
    output = round(prediction[0], 2)

    return render_template('res.html', prediction_text='SMARTPHONES PRICE is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
