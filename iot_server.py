import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import AIS_Path_Utils as utils

from haversine import haversine, Unit
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from argparse import ArgumentParser

from flask import Flask
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)

LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = utils.get_boundaries()
print([LAT_MIN, LAT_MAX, LON_MIN, LON_MAX])

full_vessel_data = pd.read_csv("trimmed_M5_Z15_ULTRAtrim.csv")

def plot_prediction(prediction, actual, mmsi):
    print("Calling plot_prediction")
    predicted_norm_LON = []
    predicted_norm_LAT = []
    predicted_anom_LON = []
    predicted_anom_LAT = []
    actual_norm_LON = []
    actual_norm_LAT = []
    actual_anom_LAT = []
    actual_anom_LON = []

    path = []

    im = plt.imread("New Orleans.png")

    anom_count = 0; # If this goes over a certain threshold, the entire path is deemed anomalous

    print("Calculating errors")
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = utils.get_boundaries("New Orleans")
    print("Bounds: " + str([LAT_MIN, LAT_MAX, LON_MIN, LON_MAX]))
    # Calculate prediction errors
    for i in range(len(prediction)):
        #print("Reading predicted point:\t" + str([prediction[i,0], prediction[i,1]]))
        #print("Reading actual point:\t" + str([actual[i,0], actual[i,1]]))

        #re-scale LAT and LON
        pred_lat = prediction[i,0] * (LAT_MAX-LAT_MIN) + LAT_MIN
        pred_lon = prediction[i,1] * (LON_MAX-LON_MIN) + LON_MIN
        actual_lat = actual[i,0] * (LAT_MAX-LAT_MIN) + LAT_MIN
        actual_lon = actual[i,1] * (LON_MAX-LON_MIN) + LON_MIN

        #print("Predicting point:\t " + str([pred_lat, pred_lon]))
        #print("Actual point:\t" + str([actual_lat, actual_lon]))    #REPORTED ACTUAL LATITUDE IS CRAZY HIGH
        
        distance = haversine((pred_lat, pred_lon), (actual_lat, actual_lon), unit=Unit.NAUTICAL_MILES)

        LAT_error = abs(prediction[i,0] - actual[i,0])
        LON_error = abs(prediction[i,1] - actual[i,1])

        if (LAT_error > .025 and LON_error > .025): # adjust this as you need
            actual_anom_LON.append(actual_lon)
            actual_anom_LAT.append(actual_lat)
            predicted_anom_LON.append(pred_lon)
            predicted_anom_LAT.append(pred_lat)
            anom_count += 1
        else:
            actual_norm_LON.append(actual_lon)
            actual_norm_LAT.append(actual_lat)
            predicted_norm_LON.append(pred_lon)
            predicted_norm_LAT.append(pred_lat)
    
    print("Plotting and saving image")
    # Plot and save to image
    fig = Figure()
    res = fig.add_subplot(1, 1, 1)
    res.imshow(im, extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX])
    res.scatter(predicted_norm_LON, predicted_norm_LAT, marker='x', s=10, color='blue', label='Predicted Position')
    res.scatter(predicted_anom_LON, predicted_anom_LAT, marker='x', s=10, color='red',  label='Predicted Anomaly')
    res.scatter(actual_norm_LON, actual_norm_LAT, marker='o', s=10, color='green', label='Actual Position')
    res.scatter(actual_anom_LON, actual_anom_LAT, marker='o', s=10, color='red', label='Anomalous Position')
    
    print("Adding labels - CANCELED")
    fig.suptitle("Trajectory Prediction for " + str(mmsi))
    res.set_xlabel("Longitude")
    res.set_ylabel("Latitude")
    res.legend()

    return fig
    #fig.savefig(str(mmsi) + "_plot.png", bbox_inches="tight", dpi=200)
    #plt.clf() # Clear the figure for the next

@app.route('/') # Home directory
def home():
    return """
    <title>How to Predict AIS paths</title>
    <h1>How to View Available MMSI Values</h1> 
    <h3>Navigate to thetop.dog:5000/<i>vessel type</i></h3>
    <p>Acceptable vessel types are: </p>
    <ul>
        <li> /cargo </li>
        <li> /fishing </li>
        <li> /pleasurecraft </li>
        <li> /tanker </li> 
        <li> /tug </li>
    </ul>
    <h1>How to Use a Model to Predict the Path</h1>
    <h3>Simply append /### to any category above!</h3>
    <p> Where ### is ANY MMSI FROM ANY CATEGORY </p>
    <p> You can mix and match! </p>
    <p>Going to the individual pages WITHOUT numbers shows a list of available MMSI values</p>
    <h1>How to determine the type of a vessel given MMSI</h1>
    <p> Navigate to /type/###, and it will specify what type of vessel it is! </p>
    """

@app.route('/type/<int:mmsi>')
def display_type(mmsi):
    mmsi_groups = full_vessel_data.groupby("MMSI")
    if (mmsi in mmsi_groups.groups):
        chosen_vessel = full_vessel_data[full_vessel_data.MMSI == mmsi].reset_index()
        print(chosen_vessel)
        return "<p>The vessel with MMSI " + str(mmsi) + " is a " + str(utils.resolve_vessel_type(chosen_vessel.VesselType[0])) + " vessel</p>"


@app.route('/cargo')
def list_cargo_MMSI():
    list_of_mmsi = "<h1>Here is a list of accepted Cargo Vessel MMSIs</h1>"
    vessel_data = full_vessel_data[full_vessel_data.VesselType.isin(utils.cargo)]
    mmsi_groups = vessel_data.groupby("MMSI")
    for MMSI in mmsi_groups.groups:
        list_of_mmsi = list_of_mmsi + "<p>"+str(MMSI)+"</p>"
    return list_of_mmsi

@app.route('/cargo/<int:mmsi>')
def predict_cargo_path(mmsi):
    model = load_model("NO_Models/cargo_model.h5")
    vessel_data = full_vessel_data#pd.read_csv("trimmed_cargo.csv")
    mmsi_groups = vessel_data.groupby("MMSI")
    
    # If the chosen MMSI exists
    if (mmsi in mmsi_groups.groups):
        print("Valid MMSI input!")
        path = vessel_data[vessel_data.MMSI == mmsi].sort_values(by="BaseDateTime").reset_index(drop=True)

        print(path)

        # Gather input variables
        #input_feature = path.iloc[:,[1,2,3,4,5]].values # LAT, LON, SOG, COG, Heading
        input_feature = path.iloc[:,[2,3,4,5,6]].values # offset by one because I goofed the preprocessing here
        input_data = input_feature

        print("placeholder")
        lookback = 30 # Number of past points to use in prediction
        # Collect path for prediction, and real path to compare against
        X = []
        Y = []
        
        for i in range(len(path)-lookback-1):
            t = []
            for j in range(0, lookback):
                t.append(input_data[(i+j), :])
            X.append(t)
            Y.append(input_data[i + lookback, :])

        # Reshape into proper format (samples, timestep, # of features
        X, Y = np.array(X), np.array(Y)
        X = X.reshape(X.shape[0], lookback, 5)
        Y = Y.reshape(Y.shape[0], 5)
        
        prediction = model.predict(X)

        print(prediction)
    
        result = plot_prediction(prediction, Y, mmsi)
        
        output = io.BytesIO() # Not sure what's happening here
        FigureCanvas(result).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
    else:
        return "You chose an invalid MMSI!"

@app.route('/fishing')
def list_fishing_MMSI():
    list_of_mmsi = "<h1>Here is a list of accepted Fishing Vessel MMSIs</h1>"
    vessel_data = full_vessel_data[full_vessel_data.VesselType.isin(utils.fishing)]
    mmsi_groups = vessel_data.groupby("MMSI")
    for MMSI in mmsi_groups.groups:
        list_of_mmsi = list_of_mmsi + "<p>"+str(MMSI)+"</p>"
    return list_of_mmsi

@app.route('/fishing/<int:mmsi>')
def predict_fishing_path(mmsi):
    model = load_model("NO_Models/fishing_model.h5")
    vessel_data = full_vessel_data#pd.read_csv("trimmed_cargo.csv")
    mmsi_groups = vessel_data.groupby("MMSI")
    
    # If the chosen MMSI exists
    if (mmsi in mmsi_groups.groups):
        print("Valid MMSI input!")
        path = vessel_data[vessel_data.MMSI == mmsi].sort_values(by="BaseDateTime").reset_index(drop=True)

        print(path)

        # Gather input variables
        #input_feature = path.iloc[:,[1,2,3,4,5]].values # LAT, LON, SOG, COG, Heading
        input_feature = path.iloc[:,[2,3,4,5,6]].values # offset by one because I goofed the preprocessing here
        input_data = input_feature

        print("placeholder")
        lookback = 30 # Number of past points to use in prediction
        # Collect path for prediction, and real path to compare against
        X = []
        Y = []
        
        for i in range(len(path)-lookback-1):
            t = []
            for j in range(0, lookback):
                t.append(input_data[(i+j), :])
            X.append(t)
            Y.append(input_data[i + lookback, :])

        # Reshape into proper format (samples, timestep, # of features
        X, Y = np.array(X), np.array(Y)
        X = X.reshape(X.shape[0], lookback, 5)
        Y = Y.reshape(Y.shape[0], 5)
        
        prediction = model.predict(X)

        print(prediction)
    
        result = plot_prediction(prediction, Y, mmsi)
        
        output = io.BytesIO() # Not sure what's happening here
        FigureCanvas(result).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
    else:
        return "You chose an invalid MMSI!"

@app.route('/pleasurecraft')
def list_pleasurecraft_MMSI():
    list_of_mmsi = "<h1>Here is a list of accepted Pleasurecraft Vessel MMSIs</h1>"
    vessel_data = full_vessel_data[full_vessel_data.VesselType.isin(utils.pleasurecraft)]
    mmsi_groups = vessel_data.groupby("MMSI")
    for MMSI in mmsi_groups.groups:
        list_of_mmsi = list_of_mmsi + "<p>"+str(MMSI)+"</p>"
    return list_of_mmsi


@app.route('/pleasurecraft/<int:mmsi>')
def predict_pleasurecraft_path(mmsi):    
    model = load_model("NO_Models/pleasurecraft_model.h5")
    vessel_data = full_vessel_data#pd.read_csv("trimmed_cargo.csv")
    mmsi_groups = vessel_data.groupby("MMSI")
    
    # If the chosen MMSI exists
    if (mmsi in mmsi_groups.groups):
        print("Valid MMSI input!")
        path = vessel_data[vessel_data.MMSI == mmsi].sort_values(by="BaseDateTime").reset_index(drop=True)

        print(path)

        # Gather input variables
        #input_feature = path.iloc[:,[1,2,3,4,5]].values # LAT, LON, SOG, COG, Heading
        input_feature = path.iloc[:,[2,3,4,5,6]].values # offset by one because I goofed the preprocessing here
        input_data = input_feature

        print("placeholder")
        lookback = 30 # Number of past points to use in prediction
        # Collect path for prediction, and real path to compare against
        X = []
        Y = []
        
        for i in range(len(path)-lookback-1):
            t = []
            for j in range(0, lookback):
                t.append(input_data[(i+j), :])
            X.append(t)
            Y.append(input_data[i + lookback, :])

        # Reshape into proper format (samples, timestep, # of features
        X, Y = np.array(X), np.array(Y)
        X = X.reshape(X.shape[0], lookback, 5)
        Y = Y.reshape(Y.shape[0], 5)
        
        prediction = model.predict(X)

        print(prediction)
    
        result = plot_prediction(prediction, Y, mmsi)
        
        output = io.BytesIO() # Not sure what's happening here
        FigureCanvas(result).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
    else:
        return "You chose an invalid MMSI!"

@app.route('/tanker')
def list_tanker_MMSI():
    list_of_mmsi = "<h1>Here is a list of accepted Tanker Vessel MMSIs</h1>"
    vessel_data = full_vessel_data[full_vessel_data.VesselType.isin(utils.tanker)]
    mmsi_groups = vessel_data.groupby("MMSI")
    for MMSI in mmsi_groups.groups:
        list_of_mmsi = list_of_mmsi + "<p>"+str(MMSI)+"</p>"
    return list_of_mmsi

@app.route('/tanker/<int:mmsi>')
def predict_tanker_path(mmsi):
    model = load_model("NO_Models/tanker_model.h5")
    vessel_data = full_vessel_data#pd.read_csv("trimmed_cargo.csv")
    mmsi_groups = vessel_data.groupby("MMSI")
    
    # If the chosen MMSI exists
    if (mmsi in mmsi_groups.groups):
        print("Valid MMSI input!")
        path = vessel_data[vessel_data.MMSI == mmsi].sort_values(by="BaseDateTime").reset_index(drop=True)

        print(path)

        # Gather input variables
        #input_feature = path.iloc[:,[1,2,3,4,5]].values # LAT, LON, SOG, COG, Heading
        input_feature = path.iloc[:,[2,3,4,5,6]].values # offset by one because I goofed the preprocessing here
        input_data = input_feature

        print("placeholder")
        lookback = 30 # Number of past points to use in prediction
        # Collect path for prediction, and real path to compare against
        X = []
        Y = []
        
        for i in range(len(path)-lookback-1):
            t = []
            for j in range(0, lookback):
                t.append(input_data[(i+j), :])
            X.append(t)
            Y.append(input_data[i + lookback, :])

        # Reshape into proper format (samples, timestep, # of features
        X, Y = np.array(X), np.array(Y)
        X = X.reshape(X.shape[0], lookback, 5)
        Y = Y.reshape(Y.shape[0], 5)
        
        prediction = model.predict(X)

        print(prediction)
    
        result = plot_prediction(prediction, Y, mmsi)
        
        output = io.BytesIO() # Not sure what's happening here
        FigureCanvas(result).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
    else:
        return "You chose an invalid MMSI!"

@app.route('/tug')
def list_tug_MMSI():
    list_of_mmsi = "<h1>Here is a list of accepted Tugboat Vessel MMSIs</h1>"
    vessel_data = full_vessel_data[full_vessel_data.VesselType.isin(utils.tug)]
    mmsi_groups = vessel_data.groupby("MMSI")
    for MMSI in mmsi_groups.groups:
        list_of_mmsi = list_of_mmsi + "<p>"+str(MMSI)+"</p>"
    return list_of_mmsi

@app.route('/tug/<int:mmsi>')
def predict_tug_path(mmsi):
    model = load_model("NO_Models/tug_model.h5")
    vessel_data = full_vessel_data#pd.read_csv("tug_cargo.csv")
    mmsi_groups = vessel_data.groupby("MMSI")
    
    # If the chosen MMSI exists
    if (mmsi in mmsi_groups.groups):
        print("Valid MMSI input!")
        path = vessel_data[vessel_data.MMSI == mmsi].sort_values(by="BaseDateTime").reset_index(drop=True)

        print(path)

        # Gather input variables
        #input_feature = path.iloc[:,[1,2,3,4,5]].values # LAT, LON, SOG, COG, Heading
        input_feature = path.iloc[:,[2,3,4,5,6]].values # offset by one because I goofed the preprocessing here
        input_data = input_feature

        print("placeholder")
        lookback = 30 # Number of past points to use in prediction
        # Collect path for prediction, and real path to compare against
        X = []
        Y = []
        
        for i in range(len(path)-lookback-1):
            t = []
            for j in range(0, lookback):
                t.append(input_data[(i+j), :])
            X.append(t)
            Y.append(input_data[i + lookback, :])

        # Reshape into proper format (samples, timestep, # of features
        X, Y = np.array(X), np.array(Y)
        X = X.reshape(X.shape[0], lookback, 5)
        Y = Y.reshape(Y.shape[0], 5)
        
        prediction = model.predict(X)

        print(prediction)
    
        result = plot_prediction(prediction, Y, mmsi)
        
        output = io.BytesIO() # Not sure what's happening here
        FigureCanvas(result).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
    else:
        return "You chose an invalid MMSI!"

if __name__ == "__main__":
    bounds = utils.get_boundaries()

    LAT_MIN = bounds[0]
    LAT_MIN = bounds[1]
    LON_MIN = bounds[2]
    LON_MAX = bounds[3]
    
    app.run(use_reloader=True, debug=True, port=8080)
