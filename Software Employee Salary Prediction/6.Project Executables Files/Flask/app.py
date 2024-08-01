from flask import Flask,render_template,request
#import joblib
import numpy as np
import pandas as pd
import pickle
app=Flask(__name__)
#model=joblib.load('random_forest_model.pkl')
model=pickle.load(open('software Industry Salary prediction.pkl','rb'))
app=Flask(__name__,template_folder='templates')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
   input_feature=[x for x in request.form.values()]
   input_feature=np.transpose(input_feature)
   input_feature=[np.array(input_feature)]
   print(input_feature)
   names=['Rating','Company Name', 'Job Title', 'Salaries Reported', 'Location',
       'Employment Status', 'Job Roles']
   data=pd.DataFrame(input_feature,columns=names)
   prediction=model.predict(data)
   result=int(prediction[0])
   print(result)
 
   return render_template('result.html', prediction_text='{}'.format(result))
if __name__=='__main__':
  app.run(debug=True)