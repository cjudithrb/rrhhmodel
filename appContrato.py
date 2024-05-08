from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# Importar los modelos
model = pickle.load(open('J:/Education/Postgrado/C2024-I/CS8130_DataAnalyticsParaTomaDecisiones/DA_Semana06/model.pkl','rb'))
sc = pickle.load(open('J:/Education/Postgrado/C2024-I/CS8130_DataAnalyticsParaTomaDecisiones/DA_Semana06/standscaler.pkl','rb'))

# crear flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    # Obtener los valores del formulario
    sl_no = int(request.form['sl_no'])
    gender = request.form['gender']
    ssc_p = float(request.form['ssc_p'])
    hsc_p = float(request.form['hsc_p'])
    degree_p = float(request.form['degree_p'])
    workex = request.form['workex']
    etest_p = float(request.form['etest_p'])
    specialisation = request.form['specialisation']
    mba_p = float(request.form['mba_p'])

    # Convertir variables categóricas a valores numéricos
    gender_numeric = 1 if gender == 'F' else 0
    workex_numeric = 1 if workex == 'Yes' else 0
    specialisation_numeric = 1 if specialisation == 'Mkt&Fin' else 0

    # Realizar la predicción utilizando la función prediction
    features = np.array([[sl_no, gender_numeric, ssc_p, hsc_p, degree_p, workex_numeric, etest_p, specialisation_numeric, mba_p]])
    print(features)
    scaler_features = sc.fit_transform(features)
    print(scaler_features)
    prediction = model.predict(scaler_features).reshape(1,-1)

    if prediction == 1:
        message = 'El Candidato sera Contratado'
    else:
        message = 'El Candidato NO sera Contratado'

    # Devolver el resultado a la página HTML
    return render_template('index.html', result=message)

# python main
if __name__ == "__main__":
    app.run(debug=True)