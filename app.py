from flask import Flask, render_template,request

import json
# for data loading and transformation
import pandas as pd

# for statistics output

# for data preparation and preprocessing for model
# model evaluation and validation 


import joblib


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
@app.route("/map")
def render_map():
	msg_data={}
	for k in request.args.keys():
		val=request.args.get(k)
		msg_data[k]=val
	if( msg_data != None ):
		return render_template("render_map0.html")
	else:
		return render_template("notfound.html")

@app.route("/Map")
def render_Map():
	msg_data={}
	for k in request.args.keys():
		val=request.args.get(k)
		msg_data[k]=val
	if( msg_data != None ):
		return render_template("render_map1.html")
	else:
		return render_template("notfound.html")

@app.route("/mAp")
def render_mAp():
	msg_data={}
	for k in request.args.keys():
		val=request.args.get(k)
		msg_data[k]=val
	if( msg_data != None ):
		return render_template("render_map2.html")
	else:
		return render_template("notfound.html")

@app.route("/MAP")
def render_MAP():
	msg_data={}
	for k in request.args.keys():
		val=request.args.get(k)
		msg_data[k]=val
	if( msg_data != None ):
		return render_template("render_map3.html")
	else:
		return render_template("notfound.html")

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
	model = joblib.load("models/clf_classifier.pkl")
	arr_results = model.predict(input_df)
	# arr_results = random.randint(0,3)

	# "work_interfere_non_effective_treatment":allElements.work_interfere_non_effective_treatment.value,
	# "work_interfere_non_effective_treatment", 

	
	# print(arr_results)
	# feature_dataset = joblib.load("models/dataset.pkl")

	# from matplotlib import cm,colors
	# map_clusters = {}
	# map_clusters = folium.Map(location=[12.9716, 77.5946], zoom_start=11)
	# kclusters = 4
	# x = np.arange(kclusters)
	# ys = [i+x+(i*x)**2 for i in range(kclusters)]
	# colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
	# rainbow = [colors.rgb2hex(i) for i in colors_array]
	# markers_colors = []
	# for k in range(139):
	# 	if( feature_dataset.iloc[k]['Cluster_label'] == arr_results ):
	# 		label = folium.Popup(str(feature_dataset.iloc[k]['Neighborhoods']) + ' - Cluster ' + str(feature_dataset.iloc[k]['Cluster_label']), parse_html=True)
	# 		folium.Marker([feature_dataset.iloc[k]['Latitude'],feature_dataset.iloc[k]['Longitude']],radius=5,popup=label,color='blue',fill=True,fill_color='#3186cc',fill_opacity=0.7).add_to(map_clusters)
	# 	else:
	# 		pass
	# html_string = map_clusters.get_root().render()
	# print(html_string)
	#map_clusters.save("templates/map.html")
	if( arr_results == 0 ):
		return "0"
	if( arr_results == 1 ):
		return "1"
	if( arr_results == 2 ):
		return "2"
	if( arr_results == 3 ):
		return "3"
	
	

if __name__ == "__main_":
	app.run(debug=True)
