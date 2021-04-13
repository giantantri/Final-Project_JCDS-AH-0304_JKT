from flask import Flask, render_template, request
import joblib
import pickle
import pandas as pd
import csv

app = Flask(__name__)

#halaman home
@app.route('/')
def home():
    return render_template('home.html')

#halaman dataset
@app.route('/database', methods=['POST', 'GET']) 
def dataset():
    with open('data.csv') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        first_line = True
        places = []
        for row in data:           
            if not first_line:
                places.append({
                "id": row[0], 
                "age": row[1],  
                "income": row[3],
                "family": row[5],
                "ccavg": row[6],
                "education": row[7],
                "mortgage": row[8],
                "securities": row[9],
                "cdaccount": row[10],
                "online": row[11],
                "creditcard": row[12],
                "personalloan": row[13]
                })
            else:
                first_line = False
  
    return render_template('dataset.html', places=places)

# #halaman visualisasi
@app.route('/visualize', methods=['POST', 'GET'])
def visual():
    return render_template('plot.html')

# #halaman input prediksi
@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    return render_template('predict.html')

# #halaman hasil prediksi
@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        input = request.form

        df_predict = pd.DataFrame({
            'Age':[input['Age']],
            'Income':[input['Income']],
            'Family':[input['Family']],
            'CCAvg':[input['CCAvg']],
            'Education':[input['Education']],
            'Mortgage':[input['Mortgage']],
            'Securities Account':[input['Securities Account']],
            'CD Account':[input['CD Account']],
            'Online':[input['Online']],
            'CreditCard':[input['CreditCard']]
        })


        prediksi = model.predict_proba(df_predict)[0][1]

        if prediksi > 0.75:
            result = "YES"
        else:
            result = "NO"

        return render_template('result.html',
            data=input, pred=result)

if __name__ == '__main__':
    # model = joblib.load('model_joblib')

    filename = 'Model_Final.sav'
    model = pickle.load(open(filename,'rb'))

    app.run(debug=True)