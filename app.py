from flask import Flask, request, jsonify, render_template
import pickle
filename = 'nlp_model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods =['POST'])
def predict():
    message = request.form["pred-text"]
    data =[message]
    vect = cv.transform(data).toarray()
    my_pred = classifier.predict(vect)
    return render_template("index.html", prediction = "Positive" if my_pred else "Negative")
if __name__ == "__main__":
    app.run(debug=True)
