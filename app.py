from flask import Flask, render_template, request
import FinalFinal
import joblib

model = joblib.load('trained_model.joblib')
app = Flask(__name__)



@app.route('/')
def main():
    return render_template('inner-page.html')


@app.route('/predict', methods=['POST'])
def home():
    sex = request.form['gender']
    pred = FinalFinal.main()
    if pred>13.7 and sex=='male':
        line="You are not Anemic"
    if pred>12.1 and sex=='female':
        line="You are not Anemic"
    else:
        line ="You are Anemic ,please visit neareast doctor soon"
    return render_template('after-page.html', data=pred,gender=sex,val=line)

if __name__ == "__main__":
    app.run(debug=True)
