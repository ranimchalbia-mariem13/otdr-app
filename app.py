from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

with open('svm_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def generate_signal(n=500, dist=10000, sig_type='normal'):
    np.random.seed(None)
    x   = np.linspace(0, dist, n)
    sig = -0.2e-3 * x + 0.01 * np.random.randn(n)

    if sig_type == 'anomalie':
        pos = np.random.randint(int(n*0.3), int(n*0.7))
        t   = np.random.randint(3)
        if t == 0:
            sig[pos:pos+5]    = sig[pos] + 0.8
            sig[pos+6:pos+15] = sig[pos] - 0.3
        elif t == 1:
            sig[pos]    = sig[pos-1] + 0.1
            sig[pos+1:] = sig[pos+1] - 3
        else:
            extra    = np.linspace(0, -0.8, n-pos)
            sig[pos:] = sig[pos:] + extra

    elif sig_type == 'attaque':
        pos = np.random.randint(int(n*0.3), int(n*0.6))
        if np.random.randint(2) == 0:
            fake      = 0.3*np.sin(np.linspace(0,4*np.pi,n-pos))
            sig[pos:] = sig[pos:] + fake
        else:
            pe          = min(pos+np.random.randint(20,60), n)
            sig[pos:pe] = sig[pos] + 0.8

    sig = (sig-sig.min()) / (sig.max()-sig.min()+1e-9)
    return sig


def extract_features(sig, dist):
    d      = np.diff(sig)
    jidx   = np.argmax(np.abs(d))
    half   = len(sig) // 2
    xn     = np.linspace(0, 1, len(sig))
    coeffs = np.polyfit(xn, sig, 1)
    fitted = np.polyval(coeffs, xn)
    lin    = 1 - np.sum((sig-fitted)**2) / \
                 (np.sum((sig-np.mean(sig))**2)+1e-9)

    return [
        np.mean(sig),
        np.std(sig),
        np.sum(sig**2),
        np.max(sig),
        np.min(sig),
        np.max(sig)-np.min(sig),
        float(np.mean((sig-np.mean(sig))**3)/
              (np.std(sig)**3+1e-9)),
        float(np.mean((sig-np.mean(sig))**4)/
              (np.std(sig)**4+1e-9)),
        (sig[-1]-sig[0])/len(sig),
        np.max(np.abs(d)),
        int(np.sum(np.abs(d)>3*np.std(d))),
        dist/20000,
        jidx/len(sig),
        float(np.var(sig)),
        np.mean(sig[:half]),
        np.mean(sig[half:]),
        float(lin),
        float(np.sum(np.abs(np.diff(d))))
    ]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data     = request.json
    dist_km  = float(data.get('distance', 10))
    dist_m   = dist_km * 1000
    scenario = data.get('scenario', 'normal')

    sig  = generate_signal(500, dist_m, scenario)
    feat = extract_features(sig, dist_m)

    feat_scaled = scaler.transform([feat])
    pred        = model.predict(feat_scaled)[0]
    prob        = model.predict_proba(feat_scaled)[0]
    confidence  = f'{max(prob)*100:.1f}%'

    if pred == 0:
        result = 'Normal'
        emoji  = '✅'
        color  = '#2ecc71'
        msg    = 'La liaison fibre est en bon état.'
    elif pred == 1:
        result = 'Anomalie Physique'
        emoji  = '🔴'
        color  = '#e74c3c'
        msg    = 'Défaut physique détecté sur la fibre !'
    else:
        result = 'Attaque Logique'
        emoji  = '⚠️'
        color  = '#e67e22'
        msg    = 'Attaque Rogue ONT ou DoS détectée !'

    return jsonify({
        'prediction' : result,
        'emoji'      : emoji,
        'color'      : color,
        'confidence' : confidence,
        'message'    : msg,
        'signal'     : sig.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)