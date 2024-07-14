from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open("BaggingRegressorModel.pkl",'rb'))
mobile=pd.read_csv("Cleaned Mobile Price.csv")

@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(mobile['company'].unique())
    Ram=sorted(mobile['Ram'].unique())
    Battery=sorted(mobile['Battery'].unique())
    Display=sorted(mobile['Display'].unique())
    memory=sorted(mobile['Inbuilt_memory'].unique())
    charger=sorted(mobile['fast_charging'].unique())
    resolution=sorted(mobile['Screen_resolution'].unique())
    Processor=sorted(mobile['Processor'].unique())
    Processor_name=sorted(mobile['Processor_name'].unique())
    rear_c=sorted(mobile['rear'].unique())
    front_c=sorted(mobile['front'].unique())	
    no_of_sim=sorted(mobile['no_of_sim'].unique())
    Is_5G=sorted(mobile['Is_5G'].unique())
    return render_template('index.html',companies=companies,Ram=Ram,Battery=Battery,Display=Display,memory=memory,charger=charger,resolution=resolution,Processor=Processor,Processor_name=Processor_name,rear_c=rear_c,front_c=front_c,no_of_sim=no_of_sim,Is_5G=Is_5G)

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    company=request.form.get('company')
    ram=float(request.form.get('ram'))
    battery=int(request.form.get('battery'))
    display=float(request.form.get('display'))
    memo=int(request.form.get('memory'))
    charge=float(request.form.get('charger'))
    resol=request.form.get('resolution')
    process=request.form.get('processor')
    process_n=request.form.get('processorn')
    rear=int(request.form.get('rear'))
    front=int(request.form.get('front'))
    sim =request.form.get('sim')
    five_g=request.form.get('g')
    prediction=model.predict(pd.DataFrame([[ram,battery,display,company,memo,charge,resol,process,process_n,rear,front,sim,five_g]],columns=['Ram','Battery','Display','company','Inbuilt_memory','fast_charging','Screen_resolution','Processor','Processor_name','rear','front','no_of_sim','Is_5G']))
    print(prediction)
    return str(np.round(prediction[0],2))

if __name__=="__main__":
    app.run(debug=True)