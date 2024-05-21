from flask import Flask, render_template, request, redirect, url_for
from ml_model import DecisionTree, randomforest, NaiveBayes, KNN

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    Symptom1 = request.form['Symptom1']
    Symptom2 = request.form['Symptom2']
    Symptom3 = request.form['Symptom3']
    Symptom4 = request.form['Symptom4']
    Symptom5 = request.form['Symptom5']
    Name = request.form['Name']

    # Call your machine learning functions
    DecisionTree()
    randomforest()
    NaiveBayes()
    KNN()

    # Redirect to result page
    return redirect(url_for('result'))

@app.route('/result')
def result():
    # Render result page with prediction data
    return render_template('result.html', prediction1=pred1.get(), prediction2=pred2.get(), prediction3=pred3.get(), prediction4=pred4.get())

if __name__ == '__main__':
    app.run(debug=True)
