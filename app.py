from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
model = pickle.load(open('models/random_forest.pkl', 'rb'))


# encoder = pickle.load(open('encoder.pickle', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('main.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        print('age: ', age)
        sex = float(request.form['sex'])
        print('sex: ', sex)
        TSH = float(request.form['TSH'])
        print('TSH: ', TSH)
        TT4 = float(request.form['TT4'])
        print('TT4: ', TT4)
        FTI = float(request.form['FTI'])
        FTI_measured = float(request.form['FTI_measured'])

        print('FTI: ', FTI)
        T3 = request.form['T3']
        print('T3: ', T3)
        T4U = float(request.form['T4U'])
        print('T4U: ', T4U)
        on_thyroxine = float(request.form['on_thyroxine'])
        print('on_thyroxine: ', on_thyroxine)
        on_antithyroid_medication = float(request.form['on_antithyroid_medication'])
        print('on_antithyroid_medication: ', on_antithyroid_medication)
        goitre = float(request.form['goitre'])
        print('goitre: ', goitre)
        hypopituitary = float(request.form['hypopituitary'])
        print('hypopituitary: ', hypopituitary)
        psych = float(request.form['psych'])
        print(psych)
        lithium = float(request.form['lithium'])
        print(lithium)
        TSH_measured = float(request.form['TSH_measured'])
        print(TSH_measured)
        TT4_measured = float(request.form['TT4_measured'])
        print(TT4_measured)
        T4U_measured = float(request.form['T4U_measured'])
        print(T4U_measured)
        T3_measured = float(request.form['T3_measured'])
        print(T3_measured)
        query_on_thyroxine = float(request.form['query_on_thyroxine'])
        print(query_on_thyroxine)
        query_hyperthyroid = float(request.form['query_hyperthyroid'])
        print(query_hyperthyroid)
        query_hypothyroid = float(request.form['query_hypothyroid'])
        print(query_hypothyroid)
        I131 = float(request.form['I131'])
        print(I131)
        thyroid_surgery = float(request.form['thyroid_surgery'])
        print(thyroid_surgery)
        pregnant = float(request.form['pregnant'])
        print(pregnant)
        sick = float(request.form['sick'])
        print(sick)
        tumor = float(request.form['tumor'])
        query = [age, sex, TSH, TT4, FTI, T3, T4U, on_thyroxine, on_antithyroid_medication, goitre, hypopituitary,
                 psych, lithium, tumor, sick, pregnant, thyroid_surgery, I131, query_hypothyroid, query_hyperthyroid,
                 query_on_thyroxine, T3_measured, T4U_measured, TT4_measured, TSH_measured,FTI_measured]
        for i in query:
            print(": ", i)
        print('query: ', query)
        predicted_class = model.predict(np.array(query).reshape(1, 26))[0]
        if predicted_class =='P':
            predicted_class = "Present"
        else:
            predicted_class = "Not present"
        #predicted_class = int(predicted_class)
        print(predicted_class)
        # class_name = encoder.inverse_transform([predicted_class])[0]

        return render_template('main.html', prediction_text=predicted_class)


if __name__ == "__main__":
    app.run(debug=True)
