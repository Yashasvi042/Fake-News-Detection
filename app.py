from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"].strip()

    if not news:
        return render_template(
            "index.html",
            prediction="Please enter some text."
        )

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    if prediction[0] == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)