import requests
from flask import Flask, render_template, redirect, url_for, request,jsonify
from werkzeug.wrappers import Request, Response

import json
import flask
# for data loading and transformation
import numpy as np 
import pandas as pd

# for statistics output
from scipy import stats
from scipy.stats import randint

# for data preparation and preprocessing for model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler

# model evaluation and validation 
from sklearn import metrics

from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
from sklearn.model_selection import cross_val_score

import joblib
import matplotlib.pyplot as plt
import folium
import random
import time

# to bypass warnings in the jupyter notebook
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=PendingDeprecationWarning)

app=Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# instantiate index page
@app.route("/")
def index():
   	return render_template("index.html")

# show the prediction map
@app.route("/prediction")
def render_map():
   	return render_template("render_map.html")

@app.route("/map")
def map_show():
   	return render_template("map.html")


# return model predictions
@app.route("/api/predict", methods=["GET"])
def predict():
	msg_data={}
	for k in request.args.keys():
		val=request.args.get(k)
		msg_data[k]=val
	f = open("models/X_test.json","r")
	X_test = json.load(f)
	f.close()
	all_cols=X_test
	input_df=pd.DataFrame(msg_data,columns=all_cols,index=[0])
	input_df.astype('int')
	model = joblib.load("models/classifier_model.pkl")
	arr_results = model.predict(input_df)
	arr_results = random.randint(0,3)
	# print(arr_results)
	feature_dataset = joblib.load("models/dataset.pkl")

	from matplotlib import cm,colors
	map_clusters = {}
	map_clusters = folium.Map(location=[12.9716, 77.5946], zoom_start=11)
	kclusters = 4
	x = np.arange(kclusters)
	ys = [i+x+(i*x)**2 for i in range(kclusters)]
	colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
	rainbow = [colors.rgb2hex(i) for i in colors_array]
	markers_colors = []
	for k in range(139):
		if( feature_dataset.iloc[k]['Cluster_label'] == arr_results ):
			label = folium.Popup(str(feature_dataset.iloc[k]['Neighborhoods']) + ' - Cluster ' + str(feature_dataset.iloc[k]['Cluster_label']), parse_html=True)
			folium.Marker([feature_dataset.iloc[k]['Latitude'],feature_dataset.iloc[k]['Longitude']],radius=5,popup=label,color='blue',fill=True,fill_color='#3186cc',fill_opacity=0.7).add_to(map_clusters)
		else:
			pass
	html_string = map_clusters.get_root().render()
	# print(html_string)
	map_clusters.save("templates/map.html")
	return "YES"

if __name__ == "__main_":
	app.run(debug=True)
