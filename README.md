# ais_flask_server
A small flask server to show vessel path predictions, based on the same dataset as in the AIS Prediction project from Summer of 2019.

# Basic Rundown
The trimmed data (derived from AIS_Trim_CSV.py) is loaded into a persistent pandas dataframe.
This dataframe has the following format, in order of columns:
1. MMSI: a 9-digit ship identification number
2. Timestamps: seconds since 1/1/1970
3. Latitude
4. Longitude
5. Speed over Ground (speed in knots)
6. Course over Ground (direction of motion)
7. Heading (direction the vessel is facing)

# Files Included
iot_server.py - the actual Flask server, run with 'Python3 iot_server.py' or 'Flask run'
AIS_RNN.py - file used to train, test, plot, etc. the recurrent neural network models 
AIS_Trim_CSV.py - file used to trim Marine Cadastre CSV files to a usable format
.h5 files - Pre-trained tensorflow models, from the previous project
.csv files - Pre-trimmed files, separated into individual types: ULTRATRIM is an amalgam of all types, before this split.
Note that the flask server only loads up ULTRATRIM, and splits it on a per-call basis, as it is more performant than having all of them loaded.

AIS_RNN and AIS_Trim_CSV are not directly used, but some code is lifted from them so they are left here for reference.

# User Operation
As shown in the main page, the instructions for use are simple once they're set up, dependent on the several vessel types:

Navigating to host:/[vessel_type] will show a list of all MMSIs of the chosen vessel type

Navigating to host:/[vessel_type]/[mmsi] will do the following:
1. The model trained on [vessel_type] specifically will be loaded
2. Pull out all entries with the requested MMSI and compile them into a path, to be predicted by the model.
3. Run a prediction
4. Display a prediction, with a legend showing the actual points, predicted points, and any points with descrepancy between actual and predicted (these are deemed 'anomalous')
