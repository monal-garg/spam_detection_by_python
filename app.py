from flask import Flask, request, jsonify, render_template
import joblib
import os
import re

app = Flask(__name__)

# Load the trained model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'spam_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.pkl'))

# --- Extra rule-based checks (links, phone numbers, shorteners) ---

URL_REGEX = re.compile(
    r'((?:https?://|www\.)[^\s]+|\b[a-zA-Z0-9-]+\.[a-zA-Z]{2,6}(?:/[^\s]*)?)',
    re.IGNORECASE
)
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-\s]?)?\d{10}\b')

KNOWN_SHORTENERS = [
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
    'buff.ly', 'shorte.st', 'cutt.ly', 'rb.gy', 'adf.ly'
]

SUSPICIOUS_KEYWORDS = [
    'free', 'winner', 'urgent', 'claim now', 'click here', 'verify account',
    'limited time', 'congratulations', 'you have won', 'act now', 'lottery',
    'prize', 'password', 'suspended', 'update payment'
]


def analyze_links(text):
    urls = URL_REGEX.findall(text)
    urls = [u.strip('.,!?)') for u in urls]

    suspicious_links = []
    for url in urls:
        lower_url = url.lower()
        is_shortener = any(shortener in lower_url for shortener in KNOWN_SHORTENERS)
        has_no_https = not lower_url.startswith('https://')
        if is_shortener or has_no_https:
            suspicious_links.append({
                'url': url,
                'reason': 'Shortened/masked link' if is_shortener else 'Not using secure (https) link'
            })

    return {
        'links_found': urls,
        'suspicious_links': suspicious_links,
        'has_phone_number': bool(PHONE_REGEX.search(text)),
        'suspicious_keywords_found': [kw for kw in SUSPICIOUS_KEYWORDS if kw in text.lower()]
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or request.form
    text = data.get('message', '')

    if not text or not text.strip():
        return jsonify({'error': 'Please enter a message'}), 400

    # ML model prediction (based on text content)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    ml_result = 'Spam' if prediction == 1 else 'Not Spam'
    confidence = round(float(probability[prediction]) * 100, 2)

    # Rule-based link/keyword analysis
    link_analysis = analyze_links(text)

    # Combine: if model says "Not Spam" but message has suspicious links/keywords,
    # flag it as a warning instead of blindly trusting the text-only model.
    final_verdict = ml_result
    warning = None

    risk_signals = len(link_analysis['suspicious_links']) + len(link_analysis['suspicious_keywords_found'])
    if ml_result == 'Not Spam' and risk_signals >= 2:
        final_verdict = 'Suspicious'
        warning = 'Message text looks normal, but it contains suspicious links/keywords. Be cautious.'
    elif ml_result == 'Spam':
        warning = 'This message matches common spam patterns.'

    return jsonify({
        'prediction': final_verdict,
        'ml_prediction': ml_result,
        'confidence': confidence,
        'links_found': link_analysis['links_found'],
        'suspicious_links': link_analysis['suspicious_links'],
        'has_phone_number': link_analysis['has_phone_number'],
        'suspicious_keywords_found': link_analysis['suspicious_keywords_found'],
        'warning': warning
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


