
from flask import Flask, request, jsonify, render_template
import pickle, re
from datetime import datetime

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

def detect_type(text):
    if "http" in text:
        return "Link"
    elif "@" in text:
        return "Email"
    elif re.search(r'\d{10}', text):
        return "Phone"
    return "Text"

def risk_level(conf):
    if conf > 0.9: return "High"
    elif conf > 0.7: return "Medium"
    return "Low"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    vec = vectorizer.transform([text])
    prob = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    confidence = max(prob)

    explanation = []
    if "win" in text.lower(): explanation.append("Contains 'win'")
    if "http" in text: explanation.append("Contains link")

    return jsonify({
        "input": text,
        "type": detect_type(text),
        "result": pred,
        "confidence": round(confidence*100,2),
        "risk": risk_level(confidence),
        "explanation": explanation,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == "__main__":
    app.run(debug=True)
